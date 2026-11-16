"""Official ``HybridRobotics/supermimic`` inference, driven through the canonical evaluator.

This adapter drives checkpoints saved by UC Berkeley HybridRobotics' official
``supermimic`` repository (``SuperMimic-GenCoord-G1 --runner.actor-obs proprio
--runner.history-length 10``) through Benchmark V2's ``reset(context)`` /
``act(step)`` contract without modifying evaluator physics or scoring.

Source of truth (``HybridRobotics/supermimic``)
------------------------------------------------
* ``supermimic/tasks/env_cfgs.py`` — ``_sensor_actor_group()`` & ``_goal_obs_group()``
* ``supermimic/representations.py`` — ``proprio_term_widths()``, ``NUM_GOAL_FRAMES=10``,
  ``GOAL_FRAME_STRIDE=5``, ``motion_goal_window()``, ``assemble_anchor_window()``
* ``supermimic/models.py`` — ``MultiEncoderAE`` & ``build_mlp()``
  (``Linear -> LayerNorm(1e-5) -> GELU -> ... -> Linear``, ``tanh`` latent)
* ``supermimic/runners/supermimic_runner.py`` — ``_seed_sim_history()``,
  ``_advance_sim_history()``, ``step()`` (`motion_times = time_steps`)
* ``rsl_rl.modules.EmpiricalNormalization`` (`v5.4.0`) —
  ``(x - _mean) / (_std + 1e-2)``

Observation & Goal Layout
-------------------------
Per-frame sensor proprio (``OBS_DIM = 93``), history length ``HIST = 10``::

     0:3    projected_gravity    -R_pelvis[2, :]
     3:6    base_ang_vel         robot/imu_ang_vel (pelvis frame)
     6:35   joint_pos            q - default_joint_pos
    35:64   joint_vel            qd
    64:93   actions              previous raw action

Actor goal for the ``motion`` encoder (``GOAL_DIM = 640``), window indices
``idx = [min(k + 5 * m, last) for m in range(10)]`` at ``k = step.reference_index``::

      0:580  content             concat([q_ref[idx], qd_ref[idx]], dim=-1).reshape(-1)
    580:640  rel_ori_b           mat_to_6d(R_robot_pelvis^T R_ref_pelvis[idx]).reshape(-1)

Frame indexing vs ``supermimic_vibed``
--------------------------------------
Unlike ``zita-ch/supermimic_vibed`` (which tracked ``f = reference_index - 1`` and
``anchor_ori_b_prev``), official ``HybridRobotics/supermimic`` runs inside mjlab
and reads ``command.time_steps`` directly in both ``SuperMimicRunner.step()`` and
``goal_window_indices()``. This adapter therefore reads ``k = step.reference_index``
and ``step.anchor_ori_b`` (= ``R_robot_torso^T @ R_ref_torso[k]``), recovering
``R_robot_pelvis^T @ R_ref_pelvis[k] = A(q) @ six_to_mat(anchor_ori_b) @ A(q_ref[k])^T``
and ``projected_gravity = pelvis_rel_k @ g_ref[k]`` algebraically with zero
privileged state.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from benchmark.evaluator import mathx
from benchmark.evaluator.torchzip import load_checkpoint
from benchmark.policies.supermimic import (
    JOINT_ORDER,
    SuperMimicContractError,
    action_scale_reference,
    default_joint_pos_reference,
    mat_to_6d,
    six_to_mat,
    waist_rotation,
)

try:
    from scipy.special import erf as _scipy_erf
except ImportError:  # pragma: no cover - stdlib fallback when scipy is absent
    _scipy_erf = None

NUM_J = 29
OBS_DIM = 3 + 3 + NUM_J + NUM_J + NUM_J  # 93
Z_DIM = 32
HIST = 10
GOAL_FRAMES = 10
GOAL_STRIDE = 5
GOAL_CONTENT_PER_FRAME = 2 * NUM_J  # 58 ([q_ref, qd_ref])
GOAL_ORI_PER_FRAME = 6
GOAL_DIM = GOAL_FRAMES * (GOAL_CONTENT_PER_FRAME + GOAL_ORI_PER_FRAME)  # 640
NORM_EPS = np.float32(1e-2)
LAYERNORM_EPS = np.float32(1e-5)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact ``torch.nn.GELU(approximate='none')`` in float32."""
    x32 = np.asarray(x, dtype=np.float32)
    if _scipy_erf is not None:
        erf_val = _scipy_erf(x32 * np.float32(_INV_SQRT2)).astype(np.float32, copy=False)
    else:
        flat = [math.erf(float(v) * _INV_SQRT2) for v in x32.ravel()]
        erf_val = np.asarray(flat, dtype=np.float32).reshape(x32.shape)
    return np.float32(0.5) * x32 * (np.float32(1.0) + erf_val)


def _layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Exact ``torch.nn.LayerNorm(..., eps=1e-5, elementwise_affine=True)`` in float32."""
    x32 = np.asarray(x, dtype=np.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = np.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
    inv_std = np.float32(1.0) / np.sqrt(var + LAYERNORM_EPS)
    return (x32 - mean) * inv_std * weight + bias


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ weight.T + bias


def _parse_ln_mlp(
    tensors: dict[str, np.ndarray], prefix: str
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Extract ``(Linear, LayerNorm)* -> Linear`` layers from ``build_mlp`` state_dict."""
    lin_ids = sorted(
        {
            int(rest.split(".")[0])
            for key in tensors
            if key.startswith(prefix + ".")
            for rest in (key[len(prefix) + 1 :],)
            if rest.endswith(".weight") and rest.split(".")[0].isdigit() and int(rest.split(".")[0]) % 3 == 0
        }
    )
    if len(lin_ids) < 1 or lin_ids != [3 * i for i in range(len(lin_ids))]:
        raise SuperMimicContractError(
            f"checkpoint prefix {prefix!r} linear layer indices {lin_ids} do not match build_mlp layout"
        )

    linears: list[tuple[np.ndarray, np.ndarray]] = []
    lns: list[tuple[np.ndarray, np.ndarray]] = []
    for pos, idx in enumerate(lin_ids):
        w_key, b_key = f"{prefix}.{idx}.weight", f"{prefix}.{idx}.bias"
        if w_key not in tensors or b_key not in tensors:
            raise SuperMimicContractError(f"missing {w_key}/{b_key}")
        w = np.asarray(tensors[w_key], dtype=np.float32)
        b = np.asarray(tensors[b_key], dtype=np.float32)
        linears.append((w, b))

        if pos < len(lin_ids) - 1:
            ln_idx = idx + 1
            ln_w_key, ln_b_key = f"{prefix}.{ln_idx}.weight", f"{prefix}.{ln_idx}.bias"
            if ln_w_key not in tensors or ln_b_key not in tensors:
                raise SuperMimicContractError(f"missing LayerNorm tensors {ln_w_key}/{ln_b_key}")
            ln_w = np.asarray(tensors[ln_w_key], dtype=np.float32)
            ln_b = np.asarray(tensors[ln_b_key], dtype=np.float32)
            if ln_w.shape != (w.shape[0],) or ln_b.shape != (w.shape[0],):
                raise SuperMimicContractError(
                    f"LayerNorm {prefix}.{ln_idx} shape {ln_w.shape} != {(w.shape[0],)}"
                )
            lns.append((ln_w, ln_b))
    return linears, lns


class OfficialSuperMimicPolicy:
    """``reset(context)`` / ``act(step)`` around ``HybridRobotics/supermimic`` ``MultiEncoderAE``."""

    kind = "official-supermimic-ae"

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors
        self.enc_lin, self.enc_ln = _parse_ln_mlp(tensors, "encoders.motion")
        self.dec_lin, self.dec_ln = _parse_ln_mlp(tensors, "decoder")

        for name in (
            "obs_normalizer._mean",
            "obs_normalizer._std",
            "goal_normalizers.motion._mean",
            "goal_normalizers.motion._std",
        ):
            if name not in tensors:
                raise SuperMimicContractError(f"checkpoint is missing normalizer buffer {name!r}")

        self.obs_mean = np.asarray(tensors["obs_normalizer._mean"], dtype=np.float32).reshape(-1)
        self.obs_std = np.asarray(tensors["obs_normalizer._std"], dtype=np.float32).reshape(-1)
        self.goal_mean = np.asarray(tensors["goal_normalizers.motion._mean"], dtype=np.float32).reshape(-1)
        self.goal_std = np.asarray(tensors["goal_normalizers.motion._std"], dtype=np.float32).reshape(-1)
        self.obs_denom = self.obs_std + NORM_EPS
        self.goal_denom = self.goal_std + NORM_EPS

        self.hist, self.goal_frames = self._check_shapes()

        self._context = None
        self._derived: dict[str, np.ndarray] | None = None
        self._hist: np.ndarray | None = None

    def _check_shapes(self) -> tuple[int, int]:
        enc_in = int(self.enc_lin[0][0].shape[1])
        z_dim = int(self.enc_lin[-1][0].shape[0])
        dec_in = int(self.dec_lin[0][0].shape[1])
        act_dim = int(self.dec_lin[-1][0].shape[0])

        if act_dim != NUM_J or z_dim != Z_DIM:
            raise SuperMimicContractError(
                f"checkpoint outputs {act_dim} actions / {z_dim} latent dims, expected {NUM_J}/{Z_DIM}"
            )
        if enc_in != GOAL_DIM:
            raise SuperMimicContractError(
                f"motion encoder input width {enc_in} != expected {GOAL_DIM}"
            )
        if (dec_in - Z_DIM) <= 0 or (dec_in - Z_DIM) % OBS_DIM != 0:
            raise SuperMimicContractError(
                f"decoder input width {dec_in} is not {Z_DIM} + H * {OBS_DIM}"
            )
        hist = (dec_in - Z_DIM) // OBS_DIM
        if hist != HIST:
            raise SuperMimicContractError(
                f"checkpoint encodes history_length={hist}; this adapter expects history_length={HIST}"
            )
        if self.obs_mean.shape != (OBS_DIM,) or self.obs_std.shape != (OBS_DIM,):
            raise SuperMimicContractError(
                f"obs_normalizer shape {self.obs_mean.shape} != {(OBS_DIM,)}"
            )
        if self.goal_mean.shape != (GOAL_DIM,) or self.goal_std.shape != (GOAL_DIM,):
            raise SuperMimicContractError(
                f"goal_normalizers.motion shape {self.goal_mean.shape} != {(GOAL_DIM,)}"
            )
        return hist, GOAL_FRAMES

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "OfficialSuperMimicPolicy":
        """Load ``MultiEncoderAE`` ``actor.pt`` weights without importing torch."""
        tensors = load_checkpoint(Path(path))
        return cls(tensors)

    def reset(self, context) -> None:
        """Clear observation history and precompute reference pelvis rotations."""
        if tuple(context.joint_order) != JOINT_ORDER:
            raise SuperMimicContractError(
                "evaluator joint order differs from G1_JOINT_NAMES"
            )
        scale_err = float(np.abs(np.asarray(context.action_scale) - action_scale_reference()).max())
        if scale_err > 1e-8:
            raise SuperMimicContractError(
                f"action_scale differs from official G1_ACTION_SCALE by {scale_err:.3e}"
            )
        pose_err = float(
            np.abs(np.asarray(context.default_joint_pos) - default_joint_pos_reference()).max()
        )
        if pose_err > 1e-12:
            raise SuperMimicContractError(
                f"default_joint_pos differs from KNEES_BENT keyframe by {pose_err:.3e}"
            )

        ref = context.reference
        q_ref = np.asarray(ref.joint_pos, dtype=np.float64)
        qd_ref = np.asarray(ref.joint_vel, dtype=np.float64)
        quat_ref = np.asarray(ref.root_quat_w, dtype=np.float64)
        r_ref = mathx.matrix_from_quat(quat_ref)
        a_ref = np.stack([waist_rotation(q) for q in q_ref])
        g_ref = mathx.projected_gravity(quat_ref)

        self._context = context
        self._derived = {
            "num_frames": int(ref.num_frames),
            "q": q_ref,
            "qd": qd_ref,
            "R": r_ref,
            "A_ref": a_ref,
            "g_ref": g_ref,
        }
        self._hist = None

    def _ref_idx(self, step) -> int:
        k = int(step.reference_index)
        if k < 0:
            raise SuperMimicContractError(f"negative reference_index {k}")
        return min(k, self._derived["num_frames"] - 1)

    def _pelvis_rel_at(self, step, k: int) -> np.ndarray:
        """``R_robot_pelvis^T @ R_ref_pelvis[k]`` from torso anchor + waist angles."""
        q = np.asarray(step.joint_pos, dtype=np.float64)
        torso_rel = six_to_mat(np.asarray(step.anchor_ori_b, dtype=np.float64))
        a_rob = waist_rotation(q)
        a_ref = self._derived["A_ref"][k]
        return a_rob @ torso_rel @ a_ref.T

    def observation(self, step) -> np.ndarray:
        """93-dim sensor proprio: ``[projected_gravity, base_ang_vel, joint_pos, joint_vel, actions]``."""
        k = self._ref_idx(step)
        pelvis_rel = self._pelvis_rel_at(step, k)
        gravity_b = pelvis_rel @ self._derived["g_ref"][k]
        q = np.asarray(step.joint_pos, dtype=np.float64)
        return np.concatenate(
            [
                gravity_b,
                np.asarray(step.base_ang_vel, dtype=np.float64),
                q - np.asarray(self._context.default_joint_pos, dtype=np.float64),
                np.asarray(step.joint_vel, dtype=np.float64),
                np.asarray(step.previous_action, dtype=np.float64),
            ]
        )

    def goal(self, k: int, pelvis_rel: np.ndarray) -> np.ndarray:
        """640-dim ``actor_goal`` = ``[content(580), rel_ori_b(60)]`` at stride 5."""
        d = self._derived
        last = d["num_frames"] - 1
        offsets = np.arange(self.goal_frames, dtype=np.int64) * GOAL_STRIDE
        idx = np.minimum(k + offsets, last)

        content = np.concatenate([d["q"][idx], d["qd"][idx]], axis=-1).reshape(-1)
        r_ref_rel = np.einsum("ji,kjl->kil", d["R"][k], d["R"][idx])
        r_rel = np.einsum("ij,kjl->kil", pelvis_rel, r_ref_rel)
        rel_ori_b = mat_to_6d(r_rel).reshape(-1)
        return np.concatenate([content, rel_ori_b], axis=-1)

    def forward(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Deterministic ``MultiEncoderAE`` forward pass in IEEE-754 float32."""
        goal_f32 = np.asarray(goal, dtype=np.float32)
        x = (goal_f32 - self.goal_mean) / self.goal_denom
        for (w, b), (ln_w, ln_b) in zip(self.enc_lin[:-1], self.enc_ln):
            x = _gelu(_layernorm(_linear(x, w, b), ln_w, ln_b))
        w_last, b_last = self.enc_lin[-1]
        z = np.tanh(_linear(x, w_last, b_last))

        obs_f32 = np.asarray(obs_hist, dtype=np.float32)
        obs_n = ((obs_f32 - self.obs_mean) / self.obs_denom).reshape(-1)
        x = np.concatenate([z, obs_n], axis=-1)
        for (w, b), (ln_w, ln_b) in zip(self.dec_lin[:-1], self.dec_ln):
            x = _gelu(_layernorm(_linear(x, w, b), ln_w, ln_b))
        w_last, b_last = self.dec_lin[-1]
        return _linear(x, w_last, b_last)

    def act(self, step) -> np.ndarray:
        k = self._ref_idx(step)
        pelvis_rel = self._pelvis_rel_at(step, k)
        obs = self.observation(step)
        if self._hist is None:
            self._hist = np.repeat(obs[None, :], self.hist, axis=0)
        else:
            self._hist = np.concatenate([self._hist[1:], obs[None, :]], axis=0)

        goal = self.goal(k, pelvis_rel)
        action = self.forward(self._hist, goal)
        if not np.all(np.isfinite(action)):
            raise SuperMimicContractError(
                f"non-finite action at step {step.step} (ref_idx {k}) of {self._context.clip}"
            )
        return action.astype(np.float64)

    def parameter_count(self) -> int:
        """Total learnable weights + normalizer stats used by inference."""
        total = self.obs_mean.size + self.obs_std.size + self.goal_mean.size + self.goal_std.size
        for w, b in self.enc_lin + self.enc_ln + self.dec_lin + self.dec_ln:
            total += w.size + b.size
        return int(total)


def make_policy(checkpoint: str | Path | None = None) -> OfficialSuperMimicPolicy:
    """Factory for ``--policy-file benchmark/policies/official_supermimic.py:make_policy``."""
    if checkpoint is None:
        raise SystemExit(
            "OfficialSuperMimicPolicy needs a checkpoint: pass --checkpoint <actor.pt>"
        )
    return OfficialSuperMimicPolicy.from_checkpoint(checkpoint)
