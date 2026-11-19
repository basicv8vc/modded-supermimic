"""InstantTrack / BlinkMimic native inference adapter for the V2 evaluator.

Pure-NumPy (torch-free) implementation of ``reset(context)`` / ``act(step)``
for checkpoints produced by ``Huy-Quang-Dao/InstantTrack`` (v1.0–v1.4) and
``fuyichen0725/blinkmimic``.

Differences from :mod:`benchmark.policies.supermimic`:
1. Supports arbitrary ``enc`` / ``dec`` MLP depths (e.g. 3 hidden layers in
   InstantTrack v1.2 or 4 hidden layers in v1.3/v1.4 and BlinkMimic).
2. Supports ``--pi_ref_skip`` (``a = (q_ref(f+1) - ref_q0) / ref_alpha + dec(...)``)
   when ``pi.ref_q0`` and ``pi.ref_alpha`` buffers are present in the checkpoint.
3. Matches ``MotionLib.goal`` end-of-clip clamping (reads ``root_v_local``,
   ``root_w_local``, ``qd`` at ``min(frame + k, last)`` rather than zeroing past
   the clip end).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from benchmark.evaluator import mathx
from benchmark.evaluator.torchzip import load_checkpoint
from benchmark.policies.supermimic import (
    CONTROL_DT,
    GOAL_FRAMES,
    GOAL_PER_FRAME,
    JOINT_ORDER,
    NUM_J,
    OBS_DIM,
    Z_DIM,
    SuperMimicContractError,
    _anchor_ori_at_state_frame,
    _elu,
    _linear,
    action_scale_reference,
    default_joint_pos_reference,
    mat_to_6d,
    reference_derivatives,
    six_to_mat,
    waist_rotation,
    yaw_matrix,
)

#: Offset of ``q(f+1)`` inside the first future frame of ``MotionLib.goal``:
#: ``dp (3) + rot_6d (6) + root_v_local (3) + root_w_local (3) = 15``.
GOAL_Q_OFFSET = 15

_LAYER_WEIGHT_RE = re.compile(r"^pi\.(enc|dec)\.(\d+)\.weight$")


def _discover_layers(tensors: dict[str, np.ndarray], head: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(weights, biases)`` for ``pi.<head>.<idx>.{weight,bias}`` in index order."""
    indices: list[int] = []
    for key in tensors:
        match = _LAYER_WEIGHT_RE.match(key)
        if match and match.group(1) == head:
            indices.append(int(match.group(2)))
    if not indices:
        raise SuperMimicContractError(f"checkpoint has no pi.{head}.*.weight tensors")
    indices.sort()
    missing = [f"pi.{head}.{idx}.bias" for idx in indices if f"pi.{head}.{idx}.bias" not in tensors]
    if missing:
        raise SuperMimicContractError(f"checkpoint is missing bias tensors: {missing}")
    weights = [tensors[f"pi.{head}.{idx}.weight"] for idx in indices]
    biases = [tensors[f"pi.{head}.{idx}.bias"] for idx in indices]
    return weights, biases


def _run_mlp(weights: list[np.ndarray], biases: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    for idx, (w, b) in enumerate(zip(weights, biases)):
        x = _linear(x, w, b)
        if idx < len(weights) - 1:
            x = _elu(x)
    return x


class InstantTrackPolicy:
    """``reset(context)`` / ``act(step)`` around InstantTrack / BlinkMimic ``pi``."""

    kind = "instanttrack-native-pi"

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors
        self.enc_w, self.enc_b = _discover_layers(tensors, "enc")
        self.dec_w, self.dec_b = _discover_layers(tensors, "dec")
        self.hist, self.goal_frames = self._check_shapes(tensors, self.enc_w, self.dec_w)

        self.obs_mean = tensors["pi.obs_norm.mean"]
        self.obs_std = np.sqrt(tensors["pi.obs_norm.var"] + 1e-2**2)
        self.goal_mean = tensors["pi.goal_norm.mean"]
        self.goal_std = np.sqrt(tensors["pi.goal_norm.var"] + 1e-2**2)

        has_q0 = "pi.ref_q0" in tensors
        has_alpha = "pi.ref_alpha" in tensors
        if has_q0 != has_alpha:
            raise SuperMimicContractError("checkpoint has only one of pi.ref_q0 / pi.ref_alpha")
        self.ref_q0 = tensors["pi.ref_q0"].astype(np.float32) if has_q0 else None
        self.ref_alpha = tensors["pi.ref_alpha"].astype(np.float32) if has_alpha else None
        if has_q0 and (self.ref_q0.shape != (NUM_J,) or self.ref_alpha.shape != (NUM_J,)):
            raise SuperMimicContractError(
                f"pi.ref_q0/pi.ref_alpha must have shape ({NUM_J},), "
                f"got {self.ref_q0.shape}/{self.ref_alpha.shape}"
            )

        self._context = None
        self._derived: dict[str, np.ndarray] | None = None
        self._hist: np.ndarray | None = None

    @staticmethod
    def _check_shapes(
        t: dict[str, np.ndarray],
        enc_w: list[np.ndarray],
        dec_w: list[np.ndarray],
    ) -> tuple[int, int]:
        enc_in = enc_w[0].shape[1]
        dec_in = dec_w[0].shape[1]
        z_dim, act_dim = enc_w[-1].shape[0], dec_w[-1].shape[0]
        if act_dim != NUM_J or z_dim != Z_DIM:
            raise SuperMimicContractError(
                f"checkpoint outputs {act_dim} actions / {z_dim} latent dims, expected {NUM_J}/{Z_DIM}"
            )
        if enc_in % GOAL_PER_FRAME or dec_in - Z_DIM <= 0:
            raise SuperMimicContractError(
                f"checkpoint input widths enc={enc_in} dec={dec_in} are not SuperMimic-shaped"
            )
        goal_frames = enc_in // GOAL_PER_FRAME
        hist = (dec_in - Z_DIM) // OBS_DIM
        if (dec_in - Z_DIM) % OBS_DIM:
            raise SuperMimicContractError(f"decoder width {dec_in} is not {Z_DIM} + k * {OBS_DIM}")
        if t["pi.obs_norm.mean"].shape != (OBS_DIM,) or t["pi.goal_norm.mean"].shape != (enc_in,):
            raise SuperMimicContractError("normaliser shapes do not match the network widths")
        if (goal_frames, hist) != (GOAL_FRAMES, 10):
            raise SuperMimicContractError(
                f"checkpoint encodes goal_frames={goal_frames} hist={hist}; "
                f"this implementation expects {GOAL_FRAMES}/10"
            )
        return hist, goal_frames

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "InstantTrackPolicy":
        """Load ``pi.*`` out of a ``torch.save`` zip file without importing torch."""
        tensors = load_checkpoint(Path(path))
        missing = [
            f"pi.{name}.{stat}"
            for name in ("obs_norm", "goal_norm")
            for stat in ("mean", "var")
            if f"pi.{name}.{stat}" not in tensors
        ]
        if missing:
            raise SuperMimicContractError(f"checkpoint is missing normaliser tensors: {missing}")
        return cls(tensors)

    def reset(self, context) -> None:
        """Clear history and precompute reference derivatives for this clip."""
        if tuple(context.joint_order) != JOINT_ORDER:
            raise SuperMimicContractError(
                "evaluator joint order differs from InstantTrack's G1_JOINT_NAMES"
            )
        scale_err = np.abs(np.asarray(context.action_scale) - action_scale_reference()).max()
        if scale_err > 1e-8:
            raise SuperMimicContractError(
                f"action_scale differs from 0.25*effort/stiffness by {scale_err:.3e}"
            )
        pose_err = np.abs(
            np.asarray(context.default_joint_pos) - default_joint_pos_reference()
        ).max()
        if pose_err > 1e-12:
            raise SuperMimicContractError(
                f"default_joint_pos differs from keyframe by {pose_err:.3e}"
            )
        if self.ref_q0 is not None:
            ref_q0_err = np.abs(self.ref_q0.astype(np.float64) - np.asarray(context.default_joint_pos)).max()
            ref_alpha_err = np.abs(self.ref_alpha.astype(np.float64) - np.asarray(context.action_scale)).max()
            if ref_q0_err > 1e-5 or ref_alpha_err > 1e-5:
                raise SuperMimicContractError(
                    f"checkpoint ref_q0/ref_alpha diverges from evaluator contract "
                    f"(max err {ref_q0_err:.3e}/{ref_alpha_err:.3e})"
                )

        ref = context.reference
        self._context = context
        self._derived = reference_derivatives(
            ref.joint_pos, ref.root_pos_w, ref.root_quat_w, dt=context.control_dt
        )
        self._derived["yaw"] = np.stack([yaw_matrix(R) for R in self._derived["R"]])
        self._hist = None

    def observation(self, step) -> np.ndarray:
        """``MDP.obs`` for one control step from ``StepInput``."""
        k = int(step.reference_index)
        frame = k - 1
        if frame < 0:
            raise SuperMimicContractError(
                f"reference_index {k} does not leave a current state frame"
            )
        d = self._derived
        q = np.asarray(step.joint_pos, dtype=np.float64)
        q_ref = d["q"][frame]

        a_robot = waist_rotation(q)
        a_ref = waist_rotation(q_ref)
        torso_rel = six_to_mat(_anchor_ori_at_state_frame(step))
        g_ref = mathx.projected_gravity(np.asarray(step.reference.root_quat_w[frame]))

        pelvis_rel = a_robot @ torso_rel @ a_ref.T
        gravity_b = pelvis_rel @ g_ref

        return np.concatenate(
            [
                q - np.asarray(self._context.default_joint_pos),
                np.asarray(step.joint_vel, dtype=np.float64),
                np.asarray(step.base_ang_vel, dtype=np.float64),
                gravity_b,
                np.asarray(step.previous_action, dtype=np.float64),
                mat_to_6d(pelvis_rel),
            ]
        )

    def goal(self, frame: int) -> np.ndarray:
        """``MotionLib.goal(frame, 10)`` matching InstantTrack's end-of-clip clamping."""
        d = self._derived
        last = len(d["p"]) - 1
        idx = np.minimum(frame + np.arange(1, self.goal_frames + 1), last)

        ht = d["yaw"][frame].T
        dp = (d["p"][idx] - d["p"][frame]) @ ht.T
        rot = np.einsum("ij,kjl->kil", ht, d["R"][idx])
        feats = np.concatenate(
            [
                dp,
                mat_to_6d(rot),
                d["root_v_local"][idx],
                d["root_w_local"][idx],
                d["q"][idx],
                d["qd"][idx] * 0.1,
            ],
            axis=-1,
        )
        return feats.reshape(-1)

    def forward(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """``Policy.forward``: ``dec(tanh(enc(goal)), obs_hist) [+ ref_skip]``."""
        goal_f32 = np.asarray(goal, dtype=np.float32)
        goal_n = ((goal - self.goal_mean) / self.goal_std).astype(np.float32)
        z = np.tanh(_run_mlp(self.enc_w, self.enc_b, goal_n))

        obs_n = ((obs_hist - self.obs_mean) / self.obs_std).reshape(-1).astype(np.float32)
        a = _run_mlp(self.dec_w, self.dec_b, np.concatenate([z, obs_n]))
        if self.ref_q0 is not None:
            q_ref = goal_f32[GOAL_Q_OFFSET : GOAL_Q_OFFSET + NUM_J]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        return a

    def act(self, step) -> np.ndarray:
        k = int(step.reference_index)
        frame = k - 1
        if frame < 0:
            raise SuperMimicContractError(
                f"reference_index {k} does not leave a current state frame"
            )
        obs = self.observation(step)
        if self._hist is None:
            self._hist = np.repeat(obs[None, :], self.hist, axis=0)
        else:
            self._hist = np.concatenate([self._hist[1:], obs[None, :]], axis=0)

        action = self.forward(self._hist, self.goal(frame))
        if not np.all(np.isfinite(action)):
            raise SuperMimicContractError(
                f"non-finite action at step {k} (frame {frame}) of {self._context.clip}"
            )
        return action.astype(np.float64)


def make_policy(checkpoint: str | Path | None = None) -> InstantTrackPolicy:
    """Factory for ``--policy-file baselines/instanttrack/policy.py:make_policy``."""
    if checkpoint is None:
        raise SystemExit(
            "InstantTrackPolicy needs a checkpoint: pass --checkpoint <ckpt_XXXXXX.pt>"
        )
    return InstantTrackPolicy.from_checkpoint(checkpoint)
