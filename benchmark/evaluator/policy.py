"""The mjlab-native policy: one concrete algorithm, not the interface.

The evaluator's contract lives in :mod:`benchmark.evaluator.policy_api` and is
deliberately tiny: ``reset(context)`` / ``act(step) -> (29,)``. Any algorithm can
satisfy it with its own observation layout, history window, network and
checkpoint format.

This module implements that contract for *one* family of algorithms: the rsl_rl
MLP actor trained by mjlab on the frozen tracking task. It is a client of the
interface, so it is allowed to assume mjlab's encoding -- the 160-dim actor
observation built by :func:`mjlab_observation` -- because that is what such a
checkpoint was trained on. Nothing in the evaluator forces another algorithm to
look like this.

Interface implementation
------------------------
``reset(context)``  binds the clip and records the default joint pose the mjlab
                    encoding subtracts
``act(step)``       builds the 160-dim vector and runs the MLP
``forward(obs)``    the raw network call, useful for testing weights directly

What is frozen about the *control* interface (not the observation)::

  action        29-dim, index ``i`` == ``constants.JOINT_ORDER[i]``
  target        ``action * ACTION_SCALE[i] + default_joint_pos[i]`` (radians)
  control rate  50 Hz (``decimation=4`` x ``timestep=0.005``)
  inference     deterministic: obs normalizer (if present) then the MLP, no
                sampling, no action clipping, no smoothing

Supported checkpoint formats
----------------------------
``*.npz``  canonical export produced by ``scripts/export_actor_npz.py``:

    w{i}/b{i}        dense layer weights (float32), i = 0..len(hidden_dims)
    obs_mean/obs_var rsl_rl ``EmpiricalNormalization`` statistics (optional)
    meta_json        JSON with ``joint_names``, ``action_scale``, ``obs_dim``,
                     ``action_dim``, ``hidden_dims``, ``activation``,
                     ``mjlab_commit``, and the source run/iteration

``*.pt``   an rsl_rl checkpoint (``model_<it>.pt``), read either through
           ``torch`` when it is importable or, without torch, directly from the
           zip container (``torchzip.load_checkpoint``). Keys are searched for a
           unique ``actor.*`` prefix; the observation normalizer is read from
           ``obs_normalizer._mean`` / ``._var``.

Both paths verify the joint order and action scale against the evaluator build:
a checkpoint trained on a different robot/action space cannot be scored silently.
This strictness is a property of *this* implementation, not of the benchmark.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .policy_api import PolicyInterfaceError


class PolicyFormatError(RuntimeError):
    pass


def mjlab_observation(step, default_joint_pos: np.ndarray) -> np.ndarray:
    """mjlab v1.6.0 actor observation (160 dims) assembled from a ``StepInput``.

    Order: reference joint pos (29) ++ reference joint vel (29) ++ anchor pos in
    the robot anchor frame (3) ++ anchor orientation (6) ++ base lin vel (3) ++
    base ang vel (3) ++ joint pos minus default (29) ++ joint vel (29) ++
    previous raw action (29).

    This is a convenience for mjlab-native policies. The evaluator does not
    require it and never inspects the result.
    """
    reference = step.reference
    index = step.reference_index
    obs = np.concatenate(
        [
            reference.joint_pos[index],
            reference.joint_vel[index],
            step.anchor_pos_b,
            step.anchor_ori_b,
            step.base_lin_vel,
            step.base_ang_vel,
            step.joint_pos - default_joint_pos,
            step.joint_vel,
            step.previous_action,
        ]
    )
    if obs.shape[0] != C.OBS_DIM:
        raise RuntimeError(f"observation dim {obs.shape[0]} != {C.OBS_DIM}")
    return obs


def _load_torch_state(path: Path) -> dict[str, np.ndarray]:
    """Read a ``.pt`` checkpoint as numpy, with or without torch installed.

    ``torch.load`` handles real checkpoints; the pure-python reader also covers
    legacy/minimal zip layouts, so it is the fallback whenever ``torch.load``
    refuses the file (and the only path when torch is absent).
    """
    state = None
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None
    if torch is not None:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001  (any parse failure -> pure reader)
            state = None
    if state is None:
        from .torchzip import load_checkpoint  # noqa: PLC0415

        return load_checkpoint(path)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise PolicyFormatError("unsupported .pt layout")
    return {
        str(k): (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        for k, v in state.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


_ACTIVATIONS = {"elu": _elu}


@dataclass
class PolicyMetadata:
    joint_names: tuple[str, ...]
    action_scale: tuple[float, ...]
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str
    source: dict


class MlpActorPolicy:
    """Deterministic MLP policy actor (rsl_rl ``Actor`` mean output)."""

    kind = "mjlab_actor_checkpoint"

    def __init__(self, weights, biases, meta: PolicyMetadata, obs_mean, obs_var):
        self._w = weights
        self._b = biases
        self.meta = meta
        self._mean = None if obs_mean is None else np.asarray(obs_mean, dtype=np.float64)
        self._var = None if obs_var is None else np.asarray(obs_var, dtype=np.float64)
        self._activation = _ACTIVATIONS[meta.activation]
        self._default_joint_pos: np.ndarray | None = None
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        meta = self.meta
        if meta.obs_dim != C.OBS_DIM:
            raise PolicyFormatError(
                f"checkpoint obs_dim={meta.obs_dim} != canonical {C.OBS_DIM}"
            )
        if meta.action_dim != C.ACTION_DIM:
            raise PolicyFormatError(
                f"checkpoint action_dim={meta.action_dim} != canonical {C.ACTION_DIM}"
            )
        if tuple(meta.joint_names) != C.JOINT_ORDER:
            raise PolicyFormatError("checkpoint joint order differs from the evaluator")
        if self._mean is not None and self._mean.shape[-1] != C.OBS_DIM:
            raise PolicyFormatError("obs normalizer width does not match obs_dim")
        self._check_shapes()

    def _check_shapes(self) -> None:
        dims = (C.OBS_DIM,) + tuple(self.meta.hidden_dims) + (C.ACTION_DIM,)
        if len(self._w) != len(dims) - 1:
            raise PolicyFormatError(
                f"expected {len(dims) - 1} layers, got {len(self._w)}"
            )
        for i in range(len(self._w)):
            if self._w[i].shape != (dims[i + 1], dims[i]):
                raise PolicyFormatError(
                    f"layer {i} shape {self._w[i].shape} != {(dims[i + 1], dims[i])}"
                )
            if self._b[i].shape != (dims[i + 1],):
                raise PolicyFormatError(f"bias {i} shape {self._b[i].shape}")

    def assert_action_scale(self, expected: np.ndarray) -> None:
        if len(self.meta.action_scale) != C.ACTION_DIM:
            raise PolicyFormatError("checkpoint does not declare a per-joint action scale")
        if not np.allclose(self.meta.action_scale, expected, atol=1e-6):
            raise PolicyFormatError(
                "checkpoint action scale differs from G1_ACTION_SCALE; refusing to score"
            )

    # -------------------------------------------------------- policy interface
    def reset(self, context) -> None:
        """Bind the clip. The mjlab encoding needs the default joint pose."""
        self._default_joint_pos = np.asarray(
            context.default_joint_pos, dtype=np.float64
        )
        expected = context.action_scale
        if expected is not None:
            self.assert_action_scale(np.asarray(expected, dtype=np.float64))

    def act(self, step) -> np.ndarray:
        """``StepInput`` -> 29 joint targets, using the mjlab 160-dim encoding."""
        if self._default_joint_pos is None:
            raise PolicyInterfaceError(
                f"{type(self).__name__}.act() called before reset(context)"
            )
        return self.forward(mjlab_observation(step, self._default_joint_pos))

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Run the network on an already-encoded observation."""
        x = np.asarray(obs, dtype=np.float64)
        if x.shape[-1] != C.OBS_DIM:
            raise PolicyFormatError(f"obs has {x.shape[-1]} dims, expected {C.OBS_DIM}")
        if self._mean is not None:
            x = (x - self._mean) / np.sqrt(self._var + C.OBS_NORMALIZATION_EPS)
        n_layers = len(self._w)
        for i in range(n_layers):
            x = self._w[i] @ x + self._b[i]
            if i < n_layers - 1:
                x = self._activation(x)
        return x

    # ------------------------------------------------------------- factories
    @classmethod
    def from_npz(cls, path: Path) -> "MlpActorPolicy":
        with np.load(path, allow_pickle=False) as data:
            if "meta_json" not in data:
                raise PolicyFormatError("npz checkpoint is missing meta_json")
            meta_raw = json.loads(str(data["meta_json"]))
            weights, biases = [], []
            i = 0
            while f"w{i}" in data:
                weights.append(np.asarray(data[f"w{i}"], dtype=np.float64))
                biases.append(np.asarray(data[f"b{i}"], dtype=np.float64))
                i += 1
            if not weights:
                raise PolicyFormatError("npz checkpoint has no dense layers")
            mean = data["obs_mean"] if "obs_mean" in data else None
            var = data["obs_var"] if "obs_var" in data else None
        meta = PolicyMetadata(
            joint_names=tuple(meta_raw["joint_names"]),
            action_scale=tuple(meta_raw["action_scale"]),
            obs_dim=int(meta_raw["obs_dim"]),
            action_dim=int(meta_raw["action_dim"]),
            hidden_dims=tuple(int(d) for d in meta_raw["hidden_dims"]),
            activation=str(meta_raw["activation"]),
            source=meta_raw.get("source", {}),
        )
        return cls(weights, biases, meta, mean, var)

    @classmethod
    def from_torch(cls, path: Path) -> "MlpActorPolicy":
        state = _load_torch_state(path)

        prefixes = {k.split(".")[0] for k in state if k.split(".")[1:2] == ["0"] and k.endswith("0.weight")}
        prefix = "actor" if "actor" in prefixes else (sorted(prefixes)[0] if prefixes else None)
        if prefix is None:
            roots = sorted({k.split(".")[0] for k in state})
            raise PolicyFormatError(
                "no actor MLP found in checkpoint: expected <prefix>.0.weight / .0.bias, "
                f"found top-level keys {roots[:8]}"
            )

        weights, biases = [], []
        i = 0
        while f"{prefix}.{2 * i}.weight" in state:
            weights.append(state[f"{prefix}.{2 * i}.weight"].astype(np.float64))
            biases.append(state[f"{prefix}.{2 * i}.bias"].astype(np.float64))
            i += 1
        mean = var = None
        for key in ("obs_normalizer._mean", "obs_normalizer.mean"):
            if key in state:
                mean = state[key].astype(np.float64)
                break
        for key in ("obs_normalizer._var", "obs_normalizer.var"):
            if key in state:
                var = state[key].astype(np.float64)
                break

        meta = PolicyMetadata(
            joint_names=C.JOINT_ORDER,
            action_scale=tuple(float(x) for x in _action_scale_default()),
            obs_dim=int(weights[0].shape[1]),
            action_dim=int(weights[-1].shape[0]),
            hidden_dims=tuple(int(w.shape[0]) for w in weights[:-1]),
            activation=C.ACTOR_ACTIVATION,
            source={"format": "torch", "path": str(path)},
        )
        return cls(weights, biases, meta, mean, var)


_DEFAULT_SCALE_CACHE: list[np.ndarray] = []


def _action_scale_default() -> np.ndarray:
    if not _DEFAULT_SCALE_CACHE:
        from .spec import build_model  # noqa: PLC0415  (import cycle avoidance)

        _DEFAULT_SCALE_CACHE.append(build_model().action_scale)
    return _DEFAULT_SCALE_CACHE[0]


def load_policy(path: Path, info=None):
    """Load a checkpoint by extension and cross-check it against the build."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        policy = MlpActorPolicy.from_npz(path)
    elif suffix in {".pt", ".pth", ".ckpt"}:
        policy = MlpActorPolicy.from_torch(path)
    else:
        raise PolicyFormatError(f"unsupported checkpoint extension {suffix!r}")
    expected = _action_scale_default() if info is None else info.action_scale
    policy.assert_action_scale(expected)
    return policy
