"""Option B: vectorized batched V2 evaluation over a single CUDA ManagerBasedRlEnv.

Instead of spawning N separate Python processes each running ``num_envs=1`` with
~20 GPU<->CPU syncs per step (Option A), this module packs all pending clips into
one contiguous GPU ``MotionLoader`` bank and evaluates ``num_envs=B`` clips in
parallel inside a single ``ManagerBasedRlEnv``:

1. **Packed multi-clip MotionCommand**: all clips are concatenated along the
   frame axis on the GPU (~423 MB for all 40 LAFAN1 clips). Each env slot ``e``
   tracks its own ``[clip_start, clip_end)`` window via ``_clip_starts`` and
   ``_clip_ends``; partial ``env.reset(env_ids=...)`` resets finished slots to
   their next clip without touching other active environments.
2. **LPT dynamic scheduling**: when ``batch_envs < len(clips)``, clips are
   scheduled longest-first (LPT) so tail idle time is minimized, while final
   results are always returned in protocol clip order.
3. **Zero-copy GPU policy path (optional)**: ``BatchedSuperMimicPolicy``,
   ``BatchedNablaMimicPolicy`` and ``BatchedReplayPolicy`` run observation
   reconstruction + MLP inference directly on GPU tensors (or vectorized NumPy
   on CPU via ``policy_device="cpu"``), eliminating per-step observation
   GPU->CPU roundtrips. Arbitrary custom policies implementing ``reset`` /
   ``act`` still work via ``SlotPolicyAdapter`` with one batched GPU->CPU
   observation pack per step.
4. **Protocol & scoring parity**: reuses ``V2CFG.build_v2_env_cfg(play=True)``,
   official mjlab reward/termination managers, ``S.rollout_normalized_return``,
   nonfinite guards, CUDA-only enforcement (``RT.probe_cuda`` /
   ``RT.inspect_runtime``), and per-clip atomic JSON persistence + resume.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from ..evaluator import constants as C
from ..evaluator import mathx, policy_api
from ..evaluator.motion import NPZ_CACHE_DIR, MotionReference, get_reference
from ..evaluator.policy_api import ClipContext, StepInput
from . import config as V2CFG
from . import runtime as RT
from . import spec as S

if TYPE_CHECKING:
    from ..evaluator.spec import ModelInfo
    from .runner import V2ClipResult

# Waist joints for algebraic pelvis<->torso reconstruction (matches supermimic.py / nablamimic.py)
_WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_WAIST_IDX = tuple(C.JOINT_ORDER.index(n) for n in _WAIST_NAMES)


# ---------------------------------------------------------------------------
# Packed multi-clip motion bank + MotionCommand wiring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackedClipMeta:
    clip_id: str
    ref: MotionReference
    start: int
    num_frames: int
    n_steps: int

    @property
    def end(self) -> int:
        return self.start + self.num_frames


class PackedMotionBank:
    """Concatenated reference clips in a single frame-major layout."""

    def __init__(self, refs: Sequence[tuple[str, MotionReference]], max_steps: int | None = None):
        if not refs:
            raise ValueError("PackedMotionBank requires at least one clip")
        self.metas: list[PackedClipMeta] = []
        self.by_clip: dict[str, int] = {}
        offset = 0
        for idx, (clip_id, ref) in enumerate(refs):
            t_len = int(ref.num_frames)
            n_steps = t_len - 1
            if max_steps is not None:
                n_steps = min(n_steps, int(max_steps))
            if n_steps < 1:
                raise ValueError(f"{clip_id}: clip shorter than two frames")
            self.metas.append(PackedClipMeta(
                clip_id=clip_id, ref=ref, start=offset, num_frames=t_len, n_steps=n_steps,
            ))
            self.by_clip[clip_id] = idx
            offset += t_len
        self.total_frames = offset

    def __len__(self) -> int:
        return len(self.metas)


def install_packed_motion_command(env, bank: PackedMotionBank) -> Any:
    """Replace ``command.motion`` with all packed clips and patch clip-windowed stepping."""
    import torch  # noqa: PLC0415

    command = env.command_manager.get_term("motion")
    device = env.device
    num_envs = env.num_envs

    body_idx_np = command.body_indexes.detach().cpu().numpy()
    refs = [m.ref for m in bank.metas]

    def _pack(arrs: list[np.ndarray]) -> torch.Tensor:
        cat = np.concatenate(arrs, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(cat).to(device=device)

    motion = command.motion
    motion.joint_pos = _pack([r.joint_pos for r in refs])
    motion.joint_vel = _pack([r.joint_vel for r in refs])
    motion.body_pos_w = _pack([r.body_pos_w[:, body_idx_np] for r in refs])
    motion.body_quat_w = _pack([r.body_quat_w[:, body_idx_np] for r in refs])
    motion.body_lin_vel_w = _pack([r.body_lin_vel_w[:, body_idx_np] for r in refs])
    motion.body_ang_vel_w = _pack([r.body_ang_vel_w[:, body_idx_np] for r in refs])
    motion.time_step_total = int(bank.total_frames)
    for attr in ("_body_pos_w", "_body_quat_w", "_body_lin_vel_w", "_body_ang_vel_w"):
        if hasattr(motion, attr):
            setattr(motion, attr, None)

    command._clip_starts = torch.zeros(num_envs, dtype=torch.long, device=device)
    command._clip_ends = torch.full((num_envs,), max(bank.metas[0].end, 2), dtype=torch.long, device=device)

    def _packed_resample_command(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = self._clip_starts[env_ids]
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        self._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel,
        )
        self._pending_forward = True

    def _packed_update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1
        # Keep indices inside each env's assigned [clip_start, clip_end) window;
        # the rollout driver completes/resets finished envs immediately after step().
        torch.minimum(self.time_steps, self._clip_ends - 1, out=self.time_steps)
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()

    command._resample_command = types.MethodType(_packed_resample_command, command)
    command._update_command = types.MethodType(_packed_update_command, command)
    return command


# ---------------------------------------------------------------------------
# Batched geometric helpers (NumPy + PyTorch)
# ---------------------------------------------------------------------------

def _waist_rotation_np(q: np.ndarray) -> np.ndarray:
    """Vectorized ``A(q) = Rz(yaw) @ Rx(roll) @ Ry(pitch)`` for ``q`` shape ``(B, 29)``."""
    yaw = q[:, _WAIST_IDX[0]]
    roll = q[:, _WAIST_IDX[1]]
    pitch = q[:, _WAIST_IDX[2]]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    b = q.shape[0]
    rz = np.zeros((b, 3, 3), dtype=np.float64)
    rz[:, 0, 0] = cy
    rz[:, 0, 1] = -sy
    rz[:, 1, 0] = sy
    rz[:, 1, 1] = cy
    rz[:, 2, 2] = 1.0

    rx = np.zeros((b, 3, 3), dtype=np.float64)
    rx[:, 0, 0] = 1.0
    rx[:, 1, 1] = cr
    rx[:, 1, 2] = -sr
    rx[:, 2, 1] = sr
    rx[:, 2, 2] = cr

    ry = np.zeros((b, 3, 3), dtype=np.float64)
    ry[:, 0, 0] = cp
    ry[:, 0, 2] = sp
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0] = -sp
    ry[:, 2, 2] = cp
    return rz @ rx @ ry


def _waist_rotation_torch(q):
    """Vectorized ``A(q) = Rz(yaw) @ Rx(roll) @ Ry(pitch)`` for ``q`` shape ``(B, 29)``."""
    import torch  # noqa: PLC0415

    yaw = q[:, _WAIST_IDX[0]]
    roll = q[:, _WAIST_IDX[1]]
    pitch = q[:, _WAIST_IDX[2]]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    z = torch.zeros_like(cy)
    o = torch.ones_like(cy)
    rz = torch.stack([
        torch.stack([cy, -sy, z], dim=-1),
        torch.stack([sy, cy, z], dim=-1),
        torch.stack([z, z, o], dim=-1),
    ], dim=-2)
    rx = torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, cr, -sr], dim=-1),
        torch.stack([z, sr, cr], dim=-1),
    ], dim=-2)
    ry = torch.stack([
        torch.stack([cp, z, sp], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-sp, z, cp], dim=-1),
    ], dim=-2)
    return rz @ rx @ ry


def _six_to_mat_np(v6: np.ndarray) -> np.ndarray:
    c0, c1 = v6[..., 0::2], v6[..., 1::2]
    return np.stack([c0, c1, np.cross(c0, c1, axis=-1)], axis=-1)


def _six_to_mat_torch(v6):
    import torch  # noqa: PLC0415

    c0, c1 = v6[..., 0::2], v6[..., 1::2]
    return torch.stack([c0, c1, torch.linalg.cross(c0, c1, dim=-1)], dim=-1)


def _mat_to_6d_np(r: np.ndarray) -> np.ndarray:
    return r[..., :, :2].reshape(r.shape[:-2] + (6,))


def _mat_to_6d_torch(r):
    return r[..., :, :2].reshape(r.shape[:-2] + (6,))


def _elu_np(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


# ---------------------------------------------------------------------------
# Batched policy adapters
# ---------------------------------------------------------------------------

def _validate_control_contract(policy_mod_name: str, default_joint_pos: np.ndarray,
                               action_scale: np.ndarray, expected_q0: np.ndarray,
                               expected_scale: np.ndarray) -> None:
    scale_err = float(np.abs(action_scale - expected_scale).max())
    if scale_err > 1e-8:
        raise RuntimeError(
            f"{policy_mod_name}: action_scale differs from reference by {scale_err:.3e}"
        )
    pose_err = float(np.abs(default_joint_pos - expected_q0).max())
    if pose_err > 1e-12:
        raise RuntimeError(
            f"{policy_mod_name}: default_joint_pos differs from reference by {pose_err:.3e}"
        )


class BatchedReplayPolicy:
    """Vectorized smoke replay / hold policy (GPU or CPU)."""

    is_gpu_native = True

    def __init__(self, mode: str, bank: PackedMotionBank, default_joint_pos: np.ndarray,
                 action_scale: np.ndarray, device: str, use_torch: bool = True):
        self.mode = mode
        self.use_torch = bool(use_torch)
        self.device = device
        self.bank = bank
        if self.use_torch:
            import torch  # noqa: PLC0415

            q0 = torch.as_tensor(default_joint_pos, dtype=torch.float64, device=device)
            sc = torch.as_tensor(action_scale, dtype=torch.float64, device=device)
            all_q = torch.from_numpy(
                np.concatenate([m.ref.joint_pos for m in bank.metas], axis=0).astype(np.float64)
            ).to(device=device)
            self._replay_actions = ((all_q - q0) / sc).to(torch.float32)
        else:
            all_q = np.concatenate([m.ref.joint_pos for m in bank.metas], axis=0).astype(np.float64)
            self._replay_actions_np = (all_q - default_joint_pos) / action_scale

    def reset_slots(self, slot_ids: Sequence[int], clip_indices: Sequence[int]) -> None:
        return None

    def act_gpu(self, *, global_frames, **kwargs):
        import torch  # noqa: PLC0415

        if self.mode == "hold":
            return torch.zeros((global_frames.shape[0], C.ACTION_DIM), dtype=torch.float32, device=self.device)
        return self._replay_actions[global_frames]

    def act_cpu(self, *, global_frames_np: np.ndarray, **kwargs) -> np.ndarray:
        if self.mode == "hold":
            return np.zeros((global_frames_np.shape[0], C.ACTION_DIM), dtype=np.float64)
        return self._replay_actions_np[global_frames_np]


class BatchedSuperMimicPolicy:
    """Exact vectorized SuperMimic inference over packed reference clips."""

    def __init__(self, base_policy, bank: PackedMotionBank, num_slots: int,
                 default_joint_pos: np.ndarray, action_scale: np.ndarray,
                 control_dt: float, device: str, use_torch: bool = True):
        from ..policies import supermimic as sm  # noqa: PLC0415

        _validate_control_contract(
            "SuperMimicPolicy",
            np.asarray(default_joint_pos, dtype=np.float64),
            np.asarray(action_scale, dtype=np.float64),
            sm.default_joint_pos_reference(),
            sm.action_scale_reference(),
        )
        self.is_gpu_native = bool(use_torch)
        self.use_torch = bool(use_torch)
        self.device = device
        self.num_slots = int(num_slots)
        self.hist_len = int(base_policy.hist)
        self.goal_frames = int(base_policy.goal_frames)
        self.clip_names = [m.clip_id for m in bank.metas]

        # Vectorized reference derivatives across all clips (same formulas as supermimic.py)
        q_list, qd_list, p_list, r_list, yaw_list = [], [], [], [], []
        rv_list, rw_list, a_ref_list, g_ref_list = [], [], [], []
        for meta in bank.metas:
            ref = meta.ref
            q = np.asarray(ref.joint_pos, dtype=np.float64)
            p = np.asarray(ref.body_pos_w[:, 0], dtype=np.float64)
            quat = np.asarray(ref.body_quat_w[:, 0], dtype=np.float64)
            r = mathx.matrix_from_quat(quat)
            root_v = sm.gradient(p, control_dt)
            joint_v = sm.gradient(q, control_dt)
            root_w = np.empty_like(p)
            root_w[1:-1] = np.stack(
                [sm.so3_log(r[i + 1] @ r[i - 1].T) for i in range(1, len(r) - 1)]
            ) / (2.0 * control_dt)
            root_w[0] = sm.so3_log(r[1] @ r[0].T) / control_dt
            root_w[-1] = sm.so3_log(r[-1] @ r[-2].T) / control_dt
            rt = np.transpose(r, (0, 2, 1))
            rv_local = np.einsum("tij,tj->ti", rt, root_v)
            rw_local = np.einsum("tij,tj->ti", rt, root_w)

            # Vectorized yaw_matrix(R)
            fx, fy = r[:, 0, 0], r[:, 1, 0]
            n = np.sqrt(fx * fx + fy * fy + 1e-9)
            c, s = fx / n, fy / n
            yaw = np.zeros((len(r), 3, 3), dtype=np.float64)
            yaw[:, 0, 0] = c
            yaw[:, 0, 1] = -s
            yaw[:, 1, 0] = s
            yaw[:, 1, 1] = c
            yaw[:, 2, 2] = 1.0

            q_list.append(q)
            qd_list.append(joint_v)
            p_list.append(p)
            r_list.append(r)
            yaw_list.append(yaw)
            rv_list.append(rv_local)
            rw_list.append(rw_local)
            a_ref_list.append(_waist_rotation_np(q))
            g_ref_list.append(mathx.projected_gravity(quat))

        if self.use_torch:
            import torch  # noqa: PLC0415

            def _t64(arrs):
                return torch.from_numpy(np.concatenate(arrs, axis=0)).to(device=device, dtype=torch.float64)

            def _t32(arr):
                return torch.as_tensor(arr, dtype=torch.float32, device=device)

            self._q_ref = _t64(q_list)
            self._qd_ref = _t64(qd_list)
            self._p_ref = _t64(p_list)
            self._r_ref = _t64(r_list)
            self._yaw_ref = _t64(yaw_list)
            self._rv_local = _t64(rv_list)
            self._rw_local = _t64(rw_list)
            self._a_ref = _t64(a_ref_list)
            self._g_ref = _t64(g_ref_list)
            self._q0 = _t64([np.asarray(default_joint_pos, dtype=np.float64)])

            self._enc_w = [_t32(w) for w in base_policy.enc_w]
            self._enc_b = [_t32(b) for b in base_policy.enc_b]
            self._dec_w = [_t32(w) for w in base_policy.dec_w]
            self._dec_b = [_t32(b) for b in base_policy.dec_b]
            self._obs_mean = _t32(base_policy.obs_mean)
            self._obs_std = _t32(base_policy.obs_std)
            self._goal_mean = _t32(base_policy.goal_mean)
            self._goal_std = _t32(base_policy.goal_std)
            self._offsets = torch.arange(1, self.goal_frames + 1, dtype=torch.long, device=device)

            self._hist = torch.zeros((num_slots, self.hist_len, sm.OBS_DIM), dtype=torch.float64, device=device)
            self._hist_init = torch.zeros(num_slots, dtype=torch.bool, device=device)
            self._slot_starts = torch.zeros(num_slots, dtype=torch.long, device=device)
            self._slot_ends = torch.ones(num_slots, dtype=torch.long, device=device)
            self._clip_starts_all = torch.tensor([m.start for m in bank.metas], dtype=torch.long, device=device)
            self._clip_ends_all = torch.tensor([m.end for m in bank.metas], dtype=torch.long, device=device)
        else:
            self._q_ref_np = np.concatenate(q_list, axis=0)
            self._qd_ref_np = np.concatenate(qd_list, axis=0)
            self._p_ref_np = np.concatenate(p_list, axis=0)
            self._r_ref_np = np.concatenate(r_list, axis=0)
            self._yaw_ref_np = np.concatenate(yaw_list, axis=0)
            self._rv_local_np = np.concatenate(rv_list, axis=0)
            self._rw_local_np = np.concatenate(rw_list, axis=0)
            self._a_ref_np = np.concatenate(a_ref_list, axis=0)
            self._g_ref_np = np.concatenate(g_ref_list, axis=0)
            self._q0_np = np.asarray(default_joint_pos, dtype=np.float64)

            self._enc_w_np = [np.asarray(w, dtype=np.float32) for w in base_policy.enc_w]
            self._enc_b_np = [np.asarray(b, dtype=np.float32) for b in base_policy.enc_b]
            self._dec_w_np = [np.asarray(w, dtype=np.float32) for w in base_policy.dec_w]
            self._dec_b_np = [np.asarray(b, dtype=np.float32) for b in base_policy.dec_b]
            self._obs_mean_np = np.asarray(base_policy.obs_mean, dtype=np.float32)
            self._obs_std_np = np.asarray(base_policy.obs_std, dtype=np.float32)
            self._goal_mean_np = np.asarray(base_policy.goal_mean, dtype=np.float32)
            self._goal_std_np = np.asarray(base_policy.goal_std, dtype=np.float32)
            self._offsets_np = np.arange(1, self.goal_frames + 1, dtype=np.int64)

            self._hist_np = np.zeros((num_slots, self.hist_len, sm.OBS_DIM), dtype=np.float64)
            self._hist_init_np = np.zeros(num_slots, dtype=bool)
            self._slot_starts_np = np.zeros(num_slots, dtype=np.int64)
            self._slot_ends_np = np.ones(num_slots, dtype=np.int64)
            self._clip_starts_all_np = np.array([m.start for m in bank.metas], dtype=np.int64)
            self._clip_ends_all_np = np.array([m.end for m in bank.metas], dtype=np.int64)

    def reset_slots(self, slot_ids: Sequence[int], clip_indices: Sequence[int]) -> None:
        if not slot_ids:
            return
        if self.use_torch:
            import torch  # noqa: PLC0415

            s_idx = torch.as_tensor(list(slot_ids), dtype=torch.long, device=self.device)
            c_idx = torch.as_tensor(list(clip_indices), dtype=torch.long, device=self.device)
            self._hist_init[s_idx] = False
            self._slot_starts[s_idx] = self._clip_starts_all[c_idx]
            self._slot_ends[s_idx] = self._clip_ends_all[c_idx]
        else:
            s_idx = np.asarray(list(slot_ids), dtype=np.int64)
            c_idx = np.asarray(list(clip_indices), dtype=np.int64)
            self._hist_init_np[s_idx] = False
            self._slot_starts_np[s_idx] = self._clip_starts_all_np[c_idx]
            self._slot_ends_np[s_idx] = self._clip_ends_all_np[c_idx]

    def act_gpu(self, *, global_frames, joint_pos, joint_vel, base_ang_vel,
                anchor_ori_b_prev, prev_actions, **kwargs):
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        frame = torch.maximum(global_frames - 1, self._slot_starts)
        q = joint_pos.to(torch.float64)
        qd = joint_vel.to(torch.float64)
        ang_vel = base_ang_vel.to(torch.float64)
        prev_a = prev_actions.to(torch.float64)
        torso_rel = _six_to_mat_torch(anchor_ori_b_prev.to(torch.float64))

        a_rob = _waist_rotation_torch(q)
        a_ref = self._a_ref[frame]
        g_ref = self._g_ref[frame]
        pelvis_rel = a_rob @ torso_rel @ a_ref.transpose(-1, -2)
        gravity_b = torch.einsum("bij,bj->bi", pelvis_rel, g_ref)

        obs = torch.cat([
            q - self._q0,
            qd,
            ang_vel,
            gravity_b,
            prev_a,
            _mat_to_6d_torch(pelvis_rel),
        ], dim=-1)  # (B, 99)

        # Update per-slot history: repeat first obs H times on slot init, else shift left
        if not torch.all(self._hist_init):
            init_mask = ~self._hist_init
            self._hist[init_mask] = obs[init_mask].unsqueeze(1).expand(-1, self.hist_len, -1)
            self._hist_init[init_mask] = True
            cont_mask = ~init_mask
            if torch.any(cont_mask):
                self._hist[cont_mask, :-1] = self._hist[cont_mask, 1:].clone()
                self._hist[cont_mask, -1] = obs[cont_mask]
        else:
            self._hist[:, :-1] = self._hist[:, 1:].clone()
            self._hist[:, -1] = obs

        # Batched goal(frame): future frames f+1 .. f+10 relative to current state frame f
        last = (self._slot_ends - 1).unsqueeze(1)
        req = frame.unsqueeze(1) + self._offsets.unsqueeze(0)
        idx = torch.minimum(req, last)
        recorded = (req <= last).unsqueeze(-1)

        ht = self._yaw_ref[frame].transpose(-1, -2)  # (B, 3, 3)
        dp = torch.einsum("bkj,bij->bki", self._p_ref[idx] - self._p_ref[frame].unsqueeze(1), ht)
        rot = torch.einsum("bij,bkjl->bkil", ht, self._r_ref[idx])
        v = torch.where(recorded, self._rv_local[idx], 0.0)
        w = torch.where(recorded, self._rw_local[idx], 0.0)
        qd_fut = torch.where(recorded, self._qd_ref[idx], 0.0)
        goal = torch.cat([
            dp, _mat_to_6d_torch(rot), v, w, self._q_ref[idx], qd_fut * 0.1,
        ], dim=-1).reshape(global_frames.shape[0], -1)

        # Forward MLP in full IEEE-754 float32 (no TF32 truncation on A100)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x = (goal.to(torch.float32) - self._goal_mean) / self._goal_std
            for i, (wt, b) in enumerate(zip(self._enc_w, self._enc_b)):
                x = F.linear(x, wt, b)
                if i < len(self._enc_w) - 1:
                    x = F.elu(x)
            z = torch.tanh(x)

            obs_n = ((self._hist.to(torch.float32) - self._obs_mean) / self._obs_std).reshape(
                global_frames.shape[0], -1
            )
            x = torch.cat([z, obs_n], dim=-1)
            for i, (wt, b) in enumerate(zip(self._dec_w, self._dec_b)):
                x = F.linear(x, wt, b)
                if i < len(self._dec_w) - 1:
                    x = F.elu(x)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return x

    def act_cpu(self, *, global_frames_np: np.ndarray, joint_pos: np.ndarray,
                joint_vel: np.ndarray, base_ang_vel: np.ndarray,
                anchor_ori_b_prev: np.ndarray, prev_actions: np.ndarray, **kwargs) -> np.ndarray:
        frame = np.maximum(global_frames_np - 1, self._slot_starts_np)
        q = np.asarray(joint_pos, dtype=np.float64)
        qd = np.asarray(joint_vel, dtype=np.float64)
        ang_vel = np.asarray(base_ang_vel, dtype=np.float64)
        prev_a = np.asarray(prev_actions, dtype=np.float64)
        torso_rel = _six_to_mat_np(np.asarray(anchor_ori_b_prev, dtype=np.float64))

        a_rob = _waist_rotation_np(q)
        a_ref = self._a_ref_np[frame]
        g_ref = self._g_ref_np[frame]
        pelvis_rel = a_rob @ torso_rel @ np.transpose(a_ref, (0, 2, 1))
        gravity_b = np.einsum("bij,bj->bi", pelvis_rel, g_ref)

        obs = np.concatenate([
            q - self._q0_np, qd, ang_vel, gravity_b, prev_a, _mat_to_6d_np(pelvis_rel),
        ], axis=-1)

        for e in range(global_frames_np.shape[0]):
            if not self._hist_init_np[e]:
                self._hist_np[e] = obs[e:e + 1]
                self._hist_init_np[e] = True
            else:
                self._hist_np[e, :-1] = self._hist_np[e, 1:]
                self._hist_np[e, -1] = obs[e]

        last = (self._slot_ends_np - 1)[:, None]
        req = frame[:, None] + self._offsets_np[None, :]
        idx = np.minimum(req, last)
        recorded = (req <= last)[:, :, None]

        ht = np.transpose(self._yaw_ref_np[frame], (0, 2, 1))
        dp = np.einsum("bkj,bij->bki", self._p_ref_np[idx] - self._p_ref_np[frame][:, None, :], ht)
        rot = np.einsum("bij,bkjl->bkil", ht, self._r_ref_np[idx])
        v = np.where(recorded, self._rv_local_np[idx], 0.0)
        w = np.where(recorded, self._rw_local_np[idx], 0.0)
        qd_fut = np.where(recorded, self._qd_ref_np[idx], 0.0)
        goal = np.concatenate([
            dp, _mat_to_6d_np(rot), v, w, self._q_ref_np[idx], qd_fut * 0.1,
        ], axis=-1).reshape(global_frames_np.shape[0], -1)

        x = ((goal - self._goal_mean_np) / self._goal_std_np).astype(np.float32)
        for i, (wt, b) in enumerate(zip(self._enc_w_np, self._enc_b_np)):
            x = x @ wt.T + b
            if i < len(self._enc_w_np) - 1:
                x = _elu_np(x)
        z = np.tanh(x)

        obs_n = ((self._hist_np - self._obs_mean_np) / self._obs_std_np).reshape(
            global_frames_np.shape[0], -1
        ).astype(np.float32)
        x = np.concatenate([z, obs_n], axis=-1)
        for i, (wt, b) in enumerate(zip(self._dec_w_np, self._dec_b_np)):
            x = x @ wt.T + b
            if i < len(self._dec_w_np) - 1:
                x = _elu_np(x)
        return x.astype(np.float64)


class BatchedNablaMimicPolicy:
    """Exact vectorized NablaMimic inference over packed reference clips."""

    def __init__(self, base_policy, bank: PackedMotionBank, num_slots: int,
                 default_joint_pos: np.ndarray, action_scale: np.ndarray,
                 control_dt: float, device: str, use_torch: bool = True):
        from ..policies import nablamimic as nm  # noqa: PLC0415

        _validate_control_contract(
            "NablaMimicPolicy",
            np.asarray(default_joint_pos, dtype=np.float64),
            np.asarray(action_scale, dtype=np.float64),
            nm.default_joint_pos_reference(),
            nm.action_scale_reference(),
        )
        self.is_gpu_native = bool(use_torch)
        self.use_torch = bool(use_torch)
        self.device = device
        self.num_slots = int(num_slots)
        self.hist_len = nm.HIST
        self.future = nm.FUTURE
        self.clip_names = [m.clip_id for m in bank.metas]

        feat_list, q_list, a_ref_list, g_ref_list = [], [], [], []
        for meta in bank.metas:
            ref = meta.ref
            ref_view = policy_api.ReferenceClip.from_motion(ref)
            feat = base_policy._goal_features(ref_view)  # (T, 68) float64
            q = np.asarray(ref.joint_pos, dtype=np.float64)
            quat = np.asarray(ref.body_quat_w[:, 0], dtype=np.float64)
            feat_list.append(feat)
            q_list.append(q)
            a_ref_list.append(_waist_rotation_np(q))
            g_ref_list.append(mathx.projected_gravity(quat))

        if self.use_torch:
            import torch  # noqa: PLC0415

            def _t64(arrs):
                return torch.from_numpy(np.concatenate(arrs, axis=0)).to(device=device, dtype=torch.float64)

            def _t32(arr):
                return torch.as_tensor(arr, dtype=torch.float32, device=device)

            self._feat = _t64(feat_list)
            self._q_ref = _t64(q_list)
            self._a_ref = _t64(a_ref_list)
            self._g_ref = _t64(g_ref_list)
            self._q0 = torch.as_tensor(default_joint_pos, dtype=torch.float64, device=device)
            self._scale = torch.as_tensor(action_scale, dtype=torch.float64, device=device)

            self._enc = [(_t32(w), _t32(b)) for w, b in base_policy.enc]
            self._dec = [(_t32(w), _t32(b)) for w, b in base_policy.dec]
            self._offsets = torch.arange(self.future, dtype=torch.long, device=device)

            self._hist = torch.zeros((num_slots, self.hist_len, nm.PROPRIO_DIM), dtype=torch.float64, device=device)
            self._hist_init = torch.zeros(num_slots, dtype=torch.bool, device=device)
            self._prev_action_raw = torch.zeros((num_slots, nm.NUM_J), dtype=torch.float64, device=device)
            self._slot_starts = torch.zeros(num_slots, dtype=torch.long, device=device)
            self._slot_ends = torch.ones(num_slots, dtype=torch.long, device=device)
            self._clip_starts_all = torch.tensor([m.start for m in bank.metas], dtype=torch.long, device=device)
            self._clip_ends_all = torch.tensor([m.end for m in bank.metas], dtype=torch.long, device=device)
        else:
            self._feat_np = np.concatenate(feat_list, axis=0)
            self._q_ref_np = np.concatenate(q_list, axis=0)
            self._a_ref_np = np.concatenate(a_ref_list, axis=0)
            self._g_ref_np = np.concatenate(g_ref_list, axis=0)
            self._q0_np = np.asarray(default_joint_pos, dtype=np.float64)
            self._scale_np = np.asarray(action_scale, dtype=np.float64)

            self._enc_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.enc]
            self._dec_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.dec]
            self._offsets_np = np.arange(self.future, dtype=np.int64)

            self._hist_np = np.zeros((num_slots, self.hist_len, nm.PROPRIO_DIM), dtype=np.float64)
            self._hist_init_np = np.zeros(num_slots, dtype=bool)
            self._prev_action_raw_np = np.zeros((num_slots, nm.NUM_J), dtype=np.float64)
            self._slot_starts_np = np.zeros(num_slots, dtype=np.int64)
            self._slot_ends_np = np.ones(num_slots, dtype=np.int64)
            self._clip_starts_all_np = np.array([m.start for m in bank.metas], dtype=np.int64)
            self._clip_ends_all_np = np.array([m.end for m in bank.metas], dtype=np.int64)

    def reset_slots(self, slot_ids: Sequence[int], clip_indices: Sequence[int]) -> None:
        if not slot_ids:
            return
        if self.use_torch:
            import torch  # noqa: PLC0415

            s_idx = torch.as_tensor(list(slot_ids), dtype=torch.long, device=self.device)
            c_idx = torch.as_tensor(list(clip_indices), dtype=torch.long, device=self.device)
            self._hist_init[s_idx] = False
            self._prev_action_raw[s_idx] = 0.0
            self._slot_starts[s_idx] = self._clip_starts_all[c_idx]
            self._slot_ends[s_idx] = self._clip_ends_all[c_idx]
        else:
            s_idx = np.asarray(list(slot_ids), dtype=np.int64)
            c_idx = np.asarray(list(clip_indices), dtype=np.int64)
            self._hist_init_np[s_idx] = False
            self._prev_action_raw_np[s_idx] = 0.0
            self._slot_starts_np[s_idx] = self._clip_starts_all_np[c_idx]
            self._slot_ends_np[s_idx] = self._clip_ends_all_np[c_idx]

    def act_gpu(self, *, global_frames, joint_pos, joint_vel, base_ang_vel,
                anchor_ori_b_prev, **kwargs):
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from ..policies import nablamimic as nm  # noqa: PLC0415

        frame = torch.maximum(global_frames - 1, self._slot_starts)
        last = self._slot_ends - 1

        q = joint_pos.to(torch.float64)
        qd = joint_vel.to(torch.float64)
        ang_vel = base_ang_vel.to(torch.float64)
        ori_prev = anchor_ori_b_prev.to(torch.float64)

        torso_rel = _six_to_mat_torch(ori_prev)
        a_rob = _waist_rotation_torch(q)
        a_ref = self._a_ref[frame]
        g_ref = self._g_ref[frame]
        gravity_b = torch.einsum("bij,bj->bi", a_rob @ torso_rel @ a_ref.transpose(-1, -2), g_ref)

        proprio = torch.cat([
            q - self._q0,
            qd * nm.JOINT_VEL_SCALE,
            ang_vel * nm.ANG_VEL_SCALE,
            gravity_b,
            self._prev_action_raw,
        ], dim=-1)  # (B, 93)

        if not torch.all(self._hist_init):
            init_mask = ~self._hist_init
            self._hist[init_mask] = proprio[init_mask].unsqueeze(1).expand(-1, self.hist_len, -1)
            self._hist_init[init_mask] = True
            cont_mask = ~init_mask
            if torch.any(cont_mask):
                self._hist[cont_mask, :-1] = self._hist[cont_mask, 1:].clone()
                self._hist[cont_mask, -1] = proprio[cont_mask]
        else:
            self._hist[:, :-1] = self._hist[:, 1:].clone()
            self._hist[:, -1] = proprio

        idx = torch.minimum(frame.unsqueeze(1) + self._offsets.unsqueeze(0), last.unsqueeze(1))
        goal_feats = self._feat[idx].reshape(global_frames.shape[0], -1)
        anchor_stacked = ori_prev[:, [0, 2, 4, 1, 3, 5]]

        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x = goal_feats.to(torch.float32)
            for wt, b in self._enc[:-1]:
                x = F.elu(x @ wt + b)
            wt, b = self._enc[-1]
            z = torch.tanh(x @ wt + b)

            dec_in = torch.cat([
                z,
                self._hist.to(torch.float32).reshape(global_frames.shape[0], -1),
                anchor_stacked.to(torch.float32),
            ], dim=-1)
            for wt, b in self._dec[:-1]:
                dec_in = F.elu(dec_in @ wt + b)
            wt, b = self._dec[-1]
            action_raw = (dec_in @ wt + b).to(torch.float64)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

        self._prev_action_raw = action_raw
        base = self._q_ref[torch.minimum(global_frames, last)]
        return (action_raw + (base - self._q0) / self._scale).to(torch.float32)

    def act_cpu(self, *, global_frames_np: np.ndarray, joint_pos: np.ndarray,
                joint_vel: np.ndarray, base_ang_vel: np.ndarray,
                anchor_ori_b_prev: np.ndarray, **kwargs) -> np.ndarray:
        from ..policies import nablamimic as nm  # noqa: PLC0415

        frame = np.maximum(global_frames_np - 1, self._slot_starts_np)
        last = self._slot_ends_np - 1

        q = np.asarray(joint_pos, dtype=np.float64)
        qd = np.asarray(joint_vel, dtype=np.float64)
        ang_vel = np.asarray(base_ang_vel, dtype=np.float64)
        ori_prev = np.asarray(anchor_ori_b_prev, dtype=np.float64)

        torso_rel = _six_to_mat_np(ori_prev)
        a_rob = _waist_rotation_np(q)
        a_ref = self._a_ref_np[frame]
        g_ref = self._g_ref_np[frame]
        gravity_b = np.einsum("bij,bj->bi", a_rob @ torso_rel @ np.transpose(a_ref, (0, 2, 1)), g_ref)

        proprio = np.concatenate([
            q - self._q0_np,
            qd * nm.JOINT_VEL_SCALE,
            ang_vel * nm.ANG_VEL_SCALE,
            gravity_b,
            self._prev_action_raw_np,
        ], axis=-1)

        for e in range(global_frames_np.shape[0]):
            if not self._hist_init_np[e]:
                self._hist_np[e] = proprio[e:e + 1]
                self._hist_init_np[e] = True
            else:
                self._hist_np[e, :-1] = self._hist_np[e, 1:]
                self._hist_np[e, -1] = proprio[e]

        idx = np.minimum(frame[:, None] + self._offsets_np[None, :], last[:, None])
        goal_feats = self._feat_np[idx].reshape(global_frames_np.shape[0], -1).astype(np.float32)
        anchor_stacked = ori_prev[:, [0, 2, 4, 1, 3, 5]].astype(np.float32)

        x = goal_feats
        for wt, b in self._enc_np[:-1]:
            x = _elu_np(x @ wt + b)
        wt, b = self._enc_np[-1]
        z = np.tanh(x @ wt + b)

        dec_in = np.concatenate([
            z,
            self._hist_np.astype(np.float32).reshape(global_frames_np.shape[0], -1),
            anchor_stacked,
        ], axis=-1)
        for wt, b in self._dec_np[:-1]:
            dec_in = _elu_np(dec_in @ wt + b)
        wt, b = self._dec_np[-1]
        action_raw = (dec_in @ wt + b).astype(np.float64)

        self._prev_action_raw_np = action_raw
        base = self._q_ref_np[np.minimum(global_frames_np, last)]
        return action_raw + (base - self._q0_np) / self._scale_np


class BatchedOfficialSuperMimicPolicy:
    """Exact vectorized official ``HybridRobotics/supermimic`` inference over packed clips."""

    def __init__(self, base_policy, bank: PackedMotionBank, num_slots: int,
                 default_joint_pos: np.ndarray, action_scale: np.ndarray,
                 control_dt: float, device: str, use_torch: bool = True):
        from ..policies import official_supermimic as osm  # noqa: PLC0415

        del control_dt
        _validate_control_contract(
            "OfficialSuperMimicPolicy",
            np.asarray(default_joint_pos, dtype=np.float64),
            np.asarray(action_scale, dtype=np.float64),
            osm.default_joint_pos_reference(),
            osm.action_scale_reference(),
        )
        self.is_gpu_native = bool(use_torch)
        self.use_torch = bool(use_torch)
        self.device = device
        self.num_slots = int(num_slots)
        self.hist_len = int(base_policy.hist)
        self.goal_frames = int(base_policy.goal_frames)
        self.goal_stride = osm.GOAL_STRIDE
        self.clip_names = [m.clip_id for m in bank.metas]

        q_list, qd_list, r_list, a_ref_list, g_ref_list = [], [], [], [], []
        for meta in bank.metas:
            ref = meta.ref
            q = np.asarray(ref.joint_pos, dtype=np.float64)
            qd = np.asarray(ref.joint_vel, dtype=np.float64)
            quat = np.asarray(ref.body_quat_w[:, 0], dtype=np.float64)
            r = mathx.matrix_from_quat(quat)
            q_list.append(q)
            qd_list.append(qd)
            r_list.append(r)
            a_ref_list.append(_waist_rotation_np(q))
            g_ref_list.append(mathx.projected_gravity(quat))

        if self.use_torch:
            import torch  # noqa: PLC0415

            def _t64(arrs):
                return torch.from_numpy(np.concatenate(arrs, axis=0)).to(device=device, dtype=torch.float64)

            def _t32(arr):
                return torch.as_tensor(arr, dtype=torch.float32, device=device)

            self._q_ref = _t64(q_list)
            self._qd_ref = _t64(qd_list)
            self._r_ref = _t64(r_list)
            self._a_ref = _t64(a_ref_list)
            self._g_ref = _t64(g_ref_list)
            self._q0 = torch.as_tensor(default_joint_pos, dtype=torch.float64, device=device)

            self._enc_lin = [(_t32(w), _t32(b)) for w, b in base_policy.enc_lin]
            self._enc_ln = [(_t32(w), _t32(b)) for w, b in base_policy.enc_ln]
            self._dec_lin = [(_t32(w), _t32(b)) for w, b in base_policy.dec_lin]
            self._dec_ln = [(_t32(w), _t32(b)) for w, b in base_policy.dec_ln]
            self._obs_mean = _t32(base_policy.obs_mean)
            self._obs_denom = _t32(base_policy.obs_denom)
            self._goal_mean = _t32(base_policy.goal_mean)
            self._goal_denom = _t32(base_policy.goal_denom)
            self._offsets = torch.arange(self.goal_frames, dtype=torch.long, device=device) * self.goal_stride

            self._hist = torch.zeros((num_slots, self.hist_len, osm.OBS_DIM), dtype=torch.float64, device=device)
            self._hist_init = torch.zeros(num_slots, dtype=torch.bool, device=device)
            self._slot_starts = torch.zeros(num_slots, dtype=torch.long, device=device)
            self._slot_ends = torch.ones(num_slots, dtype=torch.long, device=device)
            self._clip_starts_all = torch.tensor([m.start for m in bank.metas], dtype=torch.long, device=device)
            self._clip_ends_all = torch.tensor([m.end for m in bank.metas], dtype=torch.long, device=device)
        else:
            self._q_ref_np = np.concatenate(q_list, axis=0)
            self._qd_ref_np = np.concatenate(qd_list, axis=0)
            self._r_ref_np = np.concatenate(r_list, axis=0)
            self._a_ref_np = np.concatenate(a_ref_list, axis=0)
            self._g_ref_np = np.concatenate(g_ref_list, axis=0)
            self._q0_np = np.asarray(default_joint_pos, dtype=np.float64)

            self._enc_lin_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.enc_lin]
            self._enc_ln_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.enc_ln]
            self._dec_lin_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.dec_lin]
            self._dec_ln_np = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in base_policy.dec_ln]
            self._obs_mean_np = np.asarray(base_policy.obs_mean, dtype=np.float32)
            self._obs_denom_np = np.asarray(base_policy.obs_denom, dtype=np.float32)
            self._goal_mean_np = np.asarray(base_policy.goal_mean, dtype=np.float32)
            self._goal_denom_np = np.asarray(base_policy.goal_denom, dtype=np.float32)
            self._offsets_np = np.arange(self.goal_frames, dtype=np.int64) * self.goal_stride

            self._hist_np = np.zeros((num_slots, self.hist_len, osm.OBS_DIM), dtype=np.float64)
            self._hist_init_np = np.zeros(num_slots, dtype=bool)
            self._slot_starts_np = np.zeros(num_slots, dtype=np.int64)
            self._slot_ends_np = np.ones(num_slots, dtype=np.int64)
            self._clip_starts_all_np = np.array([m.start for m in bank.metas], dtype=np.int64)
            self._clip_ends_all_np = np.array([m.end for m in bank.metas], dtype=np.int64)

    def reset_slots(self, slot_ids: Sequence[int], clip_indices: Sequence[int]) -> None:
        if not slot_ids:
            return
        if self.use_torch:
            import torch  # noqa: PLC0415

            s_idx = torch.as_tensor(list(slot_ids), dtype=torch.long, device=self.device)
            c_idx = torch.as_tensor(list(clip_indices), dtype=torch.long, device=self.device)
            self._hist_init[s_idx] = False
            self._slot_starts[s_idx] = self._clip_starts_all[c_idx]
            self._slot_ends[s_idx] = self._clip_ends_all[c_idx]
        else:
            s_idx = np.asarray(list(slot_ids), dtype=np.int64)
            c_idx = np.asarray(list(clip_indices), dtype=np.int64)
            self._hist_init_np[s_idx] = False
            self._slot_starts_np[s_idx] = self._clip_starts_all_np[c_idx]
            self._slot_ends_np[s_idx] = self._clip_ends_all_np[c_idx]

    def act_gpu(self, *, global_frames, joint_pos, joint_vel, base_ang_vel,
                anchor_ori_b, prev_actions, **kwargs):
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        k = torch.minimum(global_frames, self._slot_ends - 1)
        q = joint_pos.to(torch.float64)
        qd = joint_vel.to(torch.float64)
        ang_vel = base_ang_vel.to(torch.float64)
        prev_a = prev_actions.to(torch.float64)
        torso_rel = _six_to_mat_torch(anchor_ori_b.to(torch.float64))

        a_rob = _waist_rotation_torch(q)
        a_ref = self._a_ref[k]
        g_ref = self._g_ref[k]
        pelvis_rel = a_rob @ torso_rel @ a_ref.transpose(-1, -2)
        gravity_b = torch.einsum("bij,bj->bi", pelvis_rel, g_ref)

        obs = torch.cat([gravity_b, ang_vel, q - self._q0, qd, prev_a], dim=-1)  # (B, 93)

        if not torch.all(self._hist_init):
            init_mask = ~self._hist_init
            self._hist[init_mask] = obs[init_mask].unsqueeze(1).expand(-1, self.hist_len, -1)
            self._hist_init[init_mask] = True
            cont_mask = ~init_mask
            if torch.any(cont_mask):
                self._hist[cont_mask, :-1] = self._hist[cont_mask, 1:].clone()
                self._hist[cont_mask, -1] = obs[cont_mask]
        else:
            self._hist[:, :-1] = self._hist[:, 1:].clone()
            self._hist[:, -1] = obs

        last = (self._slot_ends - 1).unsqueeze(1)
        idx = torch.minimum(k.unsqueeze(1) + self._offsets.unsqueeze(0), last)
        content = torch.cat([self._q_ref[idx], self._qd_ref[idx]], dim=-1).reshape(global_frames.shape[0], -1)
        r_ref_rel = torch.einsum("bji,bkjl->bkil", self._r_ref[k], self._r_ref[idx])
        r_rel = torch.einsum("bij,bkjl->bkil", pelvis_rel, r_ref_rel)
        rel_ori_b = _mat_to_6d_torch(r_rel).reshape(global_frames.shape[0], -1)
        goal = torch.cat([content, rel_ori_b], dim=-1)

        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x = (goal.to(torch.float32) - self._goal_mean) / self._goal_denom
            for (wt, b), (lw, lb) in zip(self._enc_lin[:-1], self._enc_ln):
                x = F.gelu(F.layer_norm(F.linear(x, wt, b), (wt.shape[0],), lw, lb, 1e-5))
            wt, b = self._enc_lin[-1]
            z = torch.tanh(F.linear(x, wt, b))

            obs_n = ((self._hist.to(torch.float32) - self._obs_mean) / self._obs_denom).reshape(
                global_frames.shape[0], -1
            )
            x = torch.cat([z, obs_n], dim=-1)
            for (wt, b), (lw, lb) in zip(self._dec_lin[:-1], self._dec_ln):
                x = F.gelu(F.layer_norm(F.linear(x, wt, b), (wt.shape[0],), lw, lb, 1e-5))
            wt, b = self._dec_lin[-1]
            x = F.linear(x, wt, b)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return x

    def act_cpu(self, *, global_frames_np: np.ndarray, joint_pos: np.ndarray,
                joint_vel: np.ndarray, base_ang_vel: np.ndarray,
                anchor_ori_b: np.ndarray, prev_actions: np.ndarray, **kwargs) -> np.ndarray:
        from ..policies import official_supermimic as osm  # noqa: PLC0415

        k = np.minimum(global_frames_np, self._slot_ends_np - 1)
        q = np.asarray(joint_pos, dtype=np.float64)
        qd = np.asarray(joint_vel, dtype=np.float64)
        ang_vel = np.asarray(base_ang_vel, dtype=np.float64)
        prev_a = np.asarray(prev_actions, dtype=np.float64)
        torso_rel = _six_to_mat_np(np.asarray(anchor_ori_b, dtype=np.float64))

        a_rob = _waist_rotation_np(q)
        a_ref = self._a_ref_np[k]
        g_ref = self._g_ref_np[k]
        pelvis_rel = a_rob @ torso_rel @ np.transpose(a_ref, (0, 2, 1))
        gravity_b = np.einsum("bij,bj->bi", pelvis_rel, g_ref)

        obs = np.concatenate([gravity_b, ang_vel, q - self._q0_np, qd, prev_a], axis=-1)

        for e in range(global_frames_np.shape[0]):
            if not self._hist_init_np[e]:
                self._hist_np[e] = obs[e:e + 1]
                self._hist_init_np[e] = True
            else:
                self._hist_np[e, :-1] = self._hist_np[e, 1:]
                self._hist_np[e, -1] = obs[e]

        last = (self._slot_ends_np - 1)[:, None]
        idx = np.minimum(k[:, None] + self._offsets_np[None, :], last)
        content = np.concatenate([self._q_ref_np[idx], self._qd_ref_np[idx]], axis=-1).reshape(
            global_frames_np.shape[0], -1
        )
        r_ref_rel = np.einsum("bji,bkjl->bkil", self._r_ref_np[k], self._r_ref_np[idx])
        r_rel = np.einsum("bij,bkjl->bkil", pelvis_rel, r_ref_rel)
        rel_ori_b = _mat_to_6d_np(r_rel).reshape(global_frames_np.shape[0], -1)
        goal = np.concatenate([content, rel_ori_b], axis=-1)

        x = (goal.astype(np.float32) - self._goal_mean_np) / self._goal_denom_np
        for (wt, b), (lw, lb) in zip(self._enc_lin_np[:-1], self._enc_ln_np):
            x = osm._gelu(osm._layernorm(x @ wt.T + b, lw, lb))
        wt, b = self._enc_lin_np[-1]
        z = np.tanh(x @ wt.T + b)

        obs_n = ((self._hist_np.astype(np.float32) - self._obs_mean_np) / self._obs_denom_np).reshape(
            global_frames_np.shape[0], -1
        )
        x = np.concatenate([z, obs_n], axis=-1)
        for (wt, b), (lw, lb) in zip(self._dec_lin_np[:-1], self._dec_ln_np):
            x = osm._gelu(osm._layernorm(x @ wt.T + b, lw, lb))
        wt, b = self._dec_lin_np[-1]
        return (x @ wt.T + b).astype(np.float64)


class SlotPolicyAdapter:
    """Fallback batched wrapper around ``B`` independent ``reset/act`` policy instances."""

    is_gpu_native = False

    def __init__(self, policy_factory: Callable[[], Any], bank: PackedMotionBank,
                 num_slots: int, default_joint_pos: np.ndarray, action_scale: np.ndarray):
        self.slots = [policy_factory() for _ in range(num_slots)]
        for pol in self.slots:
            policy_api.require_interface(pol)
        self.bank = bank
        self.default_joint_pos = np.array(default_joint_pos, dtype=np.float64)
        self.action_scale = np.array(action_scale, dtype=np.float64)
        self._ref_views = [policy_api.ReferenceClip.from_motion(m.ref) for m in bank.metas]
        self._clip_names = [m.clip_id for m in bank.metas]
        self._slot_clip_names = [""] * num_slots

    def reset_slots(self, slot_ids: Sequence[int], clip_indices: Sequence[int]) -> None:
        for slot, cidx in zip(slot_ids, clip_indices):
            meta = self.bank.metas[cidx]
            self._slot_clip_names[slot] = meta.clip_id
            self.slots[slot].reset(ClipContext(
                clip=meta.clip_id,
                joint_order=C.JOINT_ORDER,
                default_joint_pos=self.default_joint_pos.copy(),
                action_scale=self.action_scale.copy(),
                control_dt=C.STEP_DT,
                reference=self._ref_views[cidx],
            ))

    def act_cpu(self, *, active_slots: Sequence[int], slot_steps: np.ndarray,
                slot_clip_idx: np.ndarray, local_ref_indices: np.ndarray,
                joint_pos: np.ndarray, joint_vel: np.ndarray,
                base_lin_vel: np.ndarray, base_ang_vel: np.ndarray,
                anchor_pos_b: np.ndarray, anchor_ori_b: np.ndarray,
                anchor_pos_b_prev: np.ndarray, anchor_ori_b_prev: np.ndarray,
                prev_actions: np.ndarray, **kwargs) -> np.ndarray:
        actions = np.zeros((len(self.slots), C.ACTION_DIM), dtype=np.float64)
        for e in active_slots:
            cidx = int(slot_clip_idx[e])
            step_num = int(slot_steps[e])
            step_in = StepInput(
                step=step_num,
                reference_index=int(local_ref_indices[e]),
                time_s=step_num * C.STEP_DT,
                joint_pos=joint_pos[e].copy(),
                joint_vel=joint_vel[e].copy(),
                base_lin_vel=base_lin_vel[e].copy(),
                base_ang_vel=base_ang_vel[e].copy(),
                anchor_pos_b=anchor_pos_b[e].copy(),
                anchor_ori_b=anchor_ori_b[e].copy(),
                previous_action=prev_actions[e].copy(),
                reference=self._ref_views[cidx],
                anchor_pos_b_prev=anchor_pos_b_prev[e].copy(),
                anchor_ori_b_prev=anchor_ori_b_prev[e].copy(),
            )
            actions[e] = policy_api.check_action(
                self.slots[e].act(step_in),
                clip=self._clip_names[cidx],
                step=step_num,
            )
        return actions


def make_batched_policy(policy_or_factory, bank: PackedMotionBank, num_slots: int,
                        default_joint_pos: np.ndarray, action_scale: np.ndarray,
                        device: str, policy_device: str = "auto"):
    """Select the fastest exact batched policy adapter for ``policy_or_factory``."""
    if policy_device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"unsupported policy_device={policy_device!r}")
    use_torch = (policy_device in {"auto", "cuda"})

    if isinstance(policy_or_factory, type) or (
        callable(policy_or_factory) and not hasattr(policy_or_factory, "act")
    ):
        sample = policy_or_factory()
        factory = policy_or_factory
    else:
        sample = policy_or_factory
        factory = None

    kind = getattr(sample, "kind", "")
    if kind == "supermimic-native-pi":
        return BatchedSuperMimicPolicy(
            sample, bank, num_slots, default_joint_pos, action_scale, C.STEP_DT,
            device=device, use_torch=use_torch,
        ), ("supermimic_gpu" if use_torch else "supermimic_cpu_vec")

    if kind == "official-supermimic-ae":
        return BatchedOfficialSuperMimicPolicy(
            sample, bank, num_slots, default_joint_pos, action_scale, C.STEP_DT,
            device=device, use_torch=use_torch,
        ), ("official_supermimic_gpu" if use_torch else "official_supermimic_cpu_vec")

    if kind == "nablamimic-native-actor-ema":
        return BatchedNablaMimicPolicy(
            sample, bank, num_slots, default_joint_pos, action_scale, C.STEP_DT,
            device=device, use_torch=use_torch,
        ), ("nablamimic_gpu" if use_torch else "nablamimic_cpu_vec")

    if kind == "v2_smoke_scripted":
        mode = getattr(sample, "_mode", "replay")
        return BatchedReplayPolicy(
            mode, bank, default_joint_pos, action_scale, device=device, use_torch=use_torch,
        ), (f"smoke_{mode}_gpu" if use_torch else f"smoke_{mode}_cpu_vec")

    if factory is None:
        import copy  # noqa: PLC0415
        slots_factory = lambda _s=sample: copy.deepcopy(_s)
        return SlotPolicyAdapter(
            slots_factory, bank, num_slots, default_joint_pos, action_scale,
        ), "slot_fallback_cpu"

    # Reuse `sample` as slot 0 so we only construct `num_slots - 1` extra copies.
    pool = [sample] + [factory() for _ in range(max(0, num_slots - 1))]
    idx_iter = iter(pool)
    return SlotPolicyAdapter(
        lambda: next(idx_iter), bank, num_slots, default_joint_pos, action_scale,
    ), "slot_fallback_cpu"


# ---------------------------------------------------------------------------
# Batched environment state / observation extraction
# ---------------------------------------------------------------------------

def _sensor_adr(mj_model, name: str) -> int:
    import mujoco  # noqa: PLC0415

    for candidate in (name, f"robot/{name}"):
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, candidate)
        if sid >= 0:
            return int(mj_model.sensor_adr[sid])
    raise KeyError(f"sensor {name!r} not found")


def resolve_batch_envs(device: str, n_clips: int, requested: str | int = "auto") -> dict[str, Any]:
    """Choose ``num_envs`` for batched simulation (capped at ``n_clips`` and VRAM budget)."""
    if n_clips < 1:
        raise ValueError("n_clips must be >= 1")
    if isinstance(requested, str) and requested.lower() not in {"auto", "0"}:
        try:
            requested = int(requested)
        except ValueError as exc:
            raise ValueError(f"--batch-envs must be an integer or 'auto', got {requested!r}") from exc

    if isinstance(requested, int) and requested > 0:
        b = min(int(requested), n_clips)
        return {"batch_envs": b, "mode": "explicit", "n_clips": n_clips}

    # Auto: 40 envs fits comfortably in < 3 GB (motion bank ~423 MiB + ~50 MiB/env)
    # on A100/RTX PRO 6000/L4/T4; scale down only on tight-VRAM GPUs.
    free_mib = float("nan")
    try:
        import torch  # noqa: PLC0415
        free_b, _ = torch.cuda.mem_get_info(device)
        free_mib = free_b / 2**20
        avail = max(free_mib - 1536.0, 256.0)
        b_vram = max(1, int(avail // 64.0))
    except Exception:
        b_vram = 40
    b = min(40, n_clips, b_vram)
    return {"batch_envs": int(b), "mode": "auto", "vram_free_mib": free_mib, "n_clips": n_clips}


_ORIG_GET_REF = get_reference


def _warm_reference_worker(csv_path_str: str) -> None:
    _ORIG_GET_REF(Path(csv_path_str), use_cache=True)


def _load_references(
    pending_ids: Sequence[str], root: Path, use_cache: bool,
) -> list[tuple[str, MotionReference]]:
    # Parallelize cold CSV -> NPZ conversion across CPU workers (OfficialConverter
    # runs on CPU and takes ~20 min serially for all 40 clips; parallelizing drops
    # cold-cache startup to ~1-2 min while producing bit-identical NPZ files).
    if use_cache and get_reference is _ORIG_GET_REF and len(pending_ids) > 1:
        missing = [
            str(root / f"{clip}.csv")
            for clip in pending_ids
            if not (NPZ_CACHE_DIR / f"{clip}.npz").exists()
        ]
        if len(missing) > 1:
            n_workers = min(len(missing), os.cpu_count() or 1, 12)
            if n_workers > 1:
                ctx = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                    list(ex.map(_warm_reference_worker, missing))
    return [(clip, get_reference(root / f"{clip}.csv", use_cache=use_cache)) for clip in pending_ids]


# ---------------------------------------------------------------------------
# Batched rollout driver
# ---------------------------------------------------------------------------

def evaluate_batched(
    info: ModelInfo,
    policy_or_factory,
    clip_ids: Sequence[str] | None = None,
    *,
    data_dir: Path | None = None,
    use_cache: bool = False,
    progress: Callable[[V2ClipResult], None] | None = None,
    device: str = RT.DEFAULT_DEVICE,
    batch_envs: str | int = "auto",
    policy_device: str = "auto",
    checkpoint_dir: Path | None = None,
    recovery_metadata: dict | None = None,
    max_steps: int | None = None,
    warmup_steps: int = 5,
) -> tuple[list[V2ClipResult], dict[str, Any]]:
    """Evaluate all ``clip_ids`` in a single vectorized CUDA ``ManagerBasedRlEnv``."""
    import torch  # noqa: PLC0415
    from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
    from mjlab.tasks.tracking.mdp.commands import MotionCommand  # noqa: PLC0415
    from mjlab.tasks.tracking.mdp.observations import (  # noqa: PLC0415
        motion_anchor_ori_b, motion_anchor_pos_b,
    )
    from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms  # noqa: PLC0415
    from .runner import V2ClipResult  # noqa: PLC0415

    probe = RT.probe_cuda(device)
    device = RT.require_cuda_device(device)
    ids = list(clip_ids or S.load_protocol_v2()["clip_ids"])
    root = data_dir or C.DATA_DIR
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check per-clip resume cache first
    cached: dict[str, V2ClipResult] = {}
    pending_ids: list[str] = []
    for clip in ids:
        if ckpt_dir:
            part_path = ckpt_dir / f"{clip}.json"
            if part_path.exists():
                try:
                    payload = json.loads(part_path.read_text(encoding="utf-8"))
                    meta = payload.get("metadata", {})
                    match = True
                    if recovery_metadata:
                        for k, v in recovery_metadata.items():
                            if meta.get(k) != v:
                                match = False
                                break
                    if match and payload.get("result"):
                        res = V2ClipResult.from_dict(payload["result"])
                        cached[clip] = res
                        if progress is not None:
                            progress(res)
                        continue
                except Exception:
                    pass
        pending_ids.append(clip)

    if not pending_ids:
        return [cached[c] for c in ids], {
            "batch_envs": 0, "resumed_all": True, "executed_steps": 0, "rollout_wall_s": 0.0,
        }

    # 2. Load pending references (warming cold NPZ caches in parallel) and sort longest-first (LPT)
    t_ref0 = time.perf_counter()
    loaded_refs = _load_references(pending_ids, root, use_cache=use_cache)
    ref_load_s = time.perf_counter() - t_ref0
    loaded_refs.sort(key=lambda pair: int(pair[1].num_frames), reverse=True)

    bank = PackedMotionBank(loaded_refs, max_steps=max_steps)
    batch_res = resolve_batch_envs(device, len(bank), requested=batch_envs)
    num_envs = int(batch_res["batch_envs"])

    # 3. Build single batched ManagerBasedRlEnv
    cfg = V2CFG.build_v2_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].motion_file = str(bank.metas[0].ref.npz_path)

    t_build0 = time.perf_counter()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env_build_s = time.perf_counter() - t_build0
    runtime_info = RT.inspect_runtime(env)

    command = env.command_manager.get_term("motion")
    assert isinstance(command, MotionCommand)
    action_term = env.action_manager.get_term("joint_pos")

    offset_t = action_term.offset
    if offset_t.ndim == 2:
        offset_t = offset_t[0]
    live_default_q = offset_t.detach().cpu().numpy().reshape(-1).astype(np.float64)
    scale_raw = action_term.scale
    if hasattr(scale_raw, "detach"):
        if scale_raw.ndim == 2:
            scale_raw = scale_raw[0]
        live_action_scale = scale_raw.detach().cpu().numpy().reshape(-1).astype(np.float64)
    else:
        scale_arr = np.asarray(scale_raw, dtype=np.float64)
        if scale_arr.ndim == 2:
            scale_arr = scale_arr[0]
        live_action_scale = scale_arr.reshape(-1)
    scale_err = float(np.abs(live_action_scale - info.action_scale).max())
    pose_err = float(np.abs(live_default_q - info.default_joint_pos).max())
    if scale_err > 1e-6 or pose_err > 1e-6:
        raise RuntimeError(f"live mjlab action mapping drifted ({scale_err:.3e}, {pose_err:.3e})")

    install_packed_motion_command(env, bank)

    mj_model = env.sim.mj_model
    imu_lin_adr = _sensor_adr(mj_model, "imu_lin_vel")
    imu_ang_adr = _sensor_adr(mj_model, "imu_ang_vel")
    robot = env.scene["robot"]
    anchor_idx = int(command.motion_anchor_body_index)

    # Reward & termination layout
    reward_names = list(env.reward_manager.active_terms)
    term_names = [n for n in env.termination_manager._term_names if n != "time_out"]  # noqa: SLF001

    # 4. Build batched policy adapter
    batched_policy, policy_backend = make_batched_policy(
        policy_or_factory, bank, num_envs,
        np.array(info.default_joint_pos, dtype=np.float64),
        np.array(info.action_scale, dtype=np.float64),
        device=device, policy_device=policy_device,
    )

    # 5. Assign initial clips [0..num_envs-1] and reset all envs
    next_clip_queue_idx = num_envs
    slot_active = np.ones(num_envs, dtype=bool)
    slot_clip_idx = np.arange(num_envs, dtype=np.int64)
    slot_steps = np.zeros(num_envs, dtype=np.int64)
    slot_rewards: list[np.ndarray] = [np.zeros(bank.metas[c].n_steps, dtype=np.float64) for c in slot_clip_idx]
    slot_term_sums: list[dict[str, float]] = [{} for _ in range(num_envs)]
    slot_start_wall = [time.perf_counter()] * num_envs

    for e, cidx in enumerate(slot_clip_idx):
        command._clip_starts[e] = bank.metas[cidx].start
        command._clip_ends[e] = bank.metas[cidx].end

    t_reset0 = time.perf_counter()
    env.reset()
    reset_s = time.perf_counter() - t_reset0
    batched_policy.reset_slots(list(range(num_envs)), list(slot_clip_idx))

    prev_actions_gpu = torch.zeros((num_envs, C.ACTION_DIM), dtype=torch.float32, device=device)
    prev_actions_cpu = np.zeros((num_envs, C.ACTION_DIM), dtype=np.float64)

    completed: dict[str, V2ClipResult] = {}
    step_times: list[float] = []
    executed_env_steps = 0
    steady_env_steps = 0
    rollout_t0 = time.perf_counter()
    global_iter = 0

    try:
        while np.any(slot_active):
            t_iter0 = time.perf_counter()
            active_slots = np.flatnonzero(slot_active)
            executed_env_steps += len(active_slots)
            if global_iter >= warmup_steps:
                steady_env_steps += len(active_slots)

            # --- Observation extraction + policy inference ---
            if batched_policy.is_gpu_native:
                with torch.no_grad():
                    cur_pos_b = motion_anchor_pos_b(env, "motion")
                    cur_ori_b = motion_anchor_ori_b(env, "motion")
                    prev_frames = torch.maximum(command.time_steps - 1, command._clip_starts)
                    ref_pos_prev = command.motion.body_pos_w[prev_frames, anchor_idx] + env.scene.env_origins
                    ref_quat_prev = command.motion.body_quat_w[prev_frames, anchor_idx]
                    prev_pos_b, prev_quat_b = subtract_frame_transforms(
                        command.robot_anchor_pos_w, command.robot_anchor_quat_w,
                        ref_pos_prev, ref_quat_prev,
                    )
                    prev_ori_b = matrix_from_quat(prev_quat_b)[..., :2].reshape(num_envs, 6)

                    actions_gpu = batched_policy.act_gpu(
                        global_frames=command.time_steps,
                        joint_pos=robot.data.joint_pos,
                        joint_vel=robot.data.joint_vel,
                        base_lin_vel=env.sim.data.sensordata[:, imu_lin_adr:imu_lin_adr + 3],
                        base_ang_vel=env.sim.data.sensordata[:, imu_ang_adr:imu_ang_adr + 3],
                        anchor_pos_b=cur_pos_b,
                        anchor_ori_b=cur_ori_b,
                        anchor_pos_b_prev=prev_pos_b,
                        anchor_ori_b_prev=prev_ori_b,
                        prev_actions=prev_actions_gpu,
                    )
                    if not np.all(slot_active):
                        idle_mask = torch.as_tensor(~slot_active, device=device)
                        actions_gpu[idle_mask] = 0.0
                    prev_actions_gpu.copy_(actions_gpu)
            else:
                with torch.no_grad():
                    cur_pos_b = motion_anchor_pos_b(env, "motion")
                    cur_ori_b = motion_anchor_ori_b(env, "motion")
                    prev_frames = torch.maximum(command.time_steps - 1, command._clip_starts)
                    ref_pos_prev = command.motion.body_pos_w[prev_frames, anchor_idx] + env.scene.env_origins
                    ref_quat_prev = command.motion.body_quat_w[prev_frames, anchor_idx]
                    prev_pos_b, prev_quat_b = subtract_frame_transforms(
                        command.robot_anchor_pos_w, command.robot_anchor_quat_w,
                        ref_pos_prev, ref_quat_prev,
                    )
                    prev_ori_b = matrix_from_quat(prev_quat_b)[..., :2].reshape(num_envs, 6)
                    obs_pack = torch.cat([
                        robot.data.joint_pos,                                           # 0:29
                        robot.data.joint_vel,                                           # 29:58
                        env.sim.data.sensordata[:, imu_lin_adr:imu_lin_adr + 3],        # 58:61
                        env.sim.data.sensordata[:, imu_ang_adr:imu_ang_adr + 3],        # 61:64
                        cur_pos_b,                                                      # 64:67
                        cur_ori_b,                                                      # 67:73
                        prev_pos_b,                                                     # 73:76
                        prev_ori_b,                                                     # 76:82
                    ], dim=-1).cpu().numpy().astype(np.float64)

                global_frames_np = np.array(
                    [
                        min(
                            bank.metas[slot_clip_idx[e]].start + int(slot_steps[e]) + 1,
                            bank.metas[slot_clip_idx[e]].end - 1,
                        )
                        for e in range(num_envs)
                    ],
                    dtype=np.int64,
                )
                actions_np = batched_policy.act_cpu(
                    active_slots=active_slots,
                    slot_steps=slot_steps,
                    slot_clip_idx=slot_clip_idx,
                    local_ref_indices=slot_steps + 1,
                    global_frames_np=global_frames_np,
                    joint_pos=obs_pack[:, 0:29],
                    joint_vel=obs_pack[:, 29:58],
                    base_lin_vel=obs_pack[:, 58:61],
                    base_ang_vel=obs_pack[:, 61:64],
                    anchor_pos_b=obs_pack[:, 64:67],
                    anchor_ori_b=obs_pack[:, 67:73],
                    anchor_pos_b_prev=obs_pack[:, 73:76],
                    anchor_ori_b_prev=obs_pack[:, 76:82],
                    prev_actions=prev_actions_cpu,
                )
                for e in active_slots:
                    if not np.all(np.isfinite(actions_np[e])):
                        raise policy_api.ActionFormatError(
                            f"{bank.metas[slot_clip_idx[e]].clip_id} step {slot_steps[e]}: non-finite action"
                        )
                actions_np[~slot_active] = 0.0
                prev_actions_cpu[:] = actions_np
                actions_gpu = torch.as_tensor(actions_np, dtype=torch.float32, device=device)

            # --- Environment step ---
            _obs, reward_gpu, _terminated, _truncated, _extras = env.step(actions_gpu)

            # --- Single packed GPU->CPU readback for reward, terms, terminations & finiteness ---
            with torch.no_grad():
                term_cols = [
                    env.termination_manager.get_term(name).to(torch.float32).unsqueeze(-1)
                    for name in term_names
                ]
                is_finite_col = (
                    torch.isfinite(env.sim.data.qpos).all(dim=-1)
                    & torch.isfinite(env.sim.data.qvel).all(dim=-1)
                ).to(torch.float32).unsqueeze(-1)
                act_finite_col = torch.isfinite(actions_gpu).all(dim=-1).to(torch.float32).unsqueeze(-1)
                step_out = torch.cat([
                    reward_gpu.unsqueeze(-1),
                    env.reward_manager._step_reward * C.STEP_DT,  # noqa: SLF001
                    *term_cols,
                    is_finite_col,
                    act_finite_col,
                ], dim=-1).cpu().numpy()

            n_rew = len(reward_names)
            n_term = len(term_names)
            step_rewards = step_out[:, 0]
            term_vals = step_out[:, 1:1 + n_rew]
            fired_flags = step_out[:, 1 + n_rew:1 + n_rew + n_term] > 0.5
            finite_ok = step_out[:, 1 + n_rew + n_term] > 0.5
            action_ok = step_out[:, 1 + n_rew + n_term + 1] > 0.5

            # --- Bookkeeping + dynamic LPT slot assignment ---
            reset_slots: list[int] = []
            reassigned_slots: list[int] = []
            reassigned_clips: list[int] = []

            for e in range(num_envs):
                if not slot_active[e]:
                    # Keep idle envs cleared of _manual_reset_pending if they ever fall
                    if np.any(fired_flags[e]) or not finite_ok[e]:
                        reset_slots.append(e)
                    continue

                cidx = int(slot_clip_idx[e])
                meta = bank.metas[cidx]
                s = int(slot_steps[e])

                if not action_ok[e]:
                    raise policy_api.ActionFormatError(
                        f"{meta.clip_id} step {s}: non-finite action"
                    )

                slot_rewards[e][s] = float(step_rewards[e])
                tsums = slot_term_sums[e]
                for j, tname in enumerate(reward_names):
                    tsums[tname] = tsums.get(tname, 0.0) + float(term_vals[e, j])

                fired = [term_names[j] for j in range(n_term) if fired_flags[e, j]]
                if not finite_ok[e] and "nonfinite" not in fired:
                    fired.append("nonfinite")

                alive_steps = s + 1
                clip_done = bool(fired) or (alive_steps >= meta.n_steps)
                slot_steps[e] = alive_steps

                if clip_done:
                    terminated_at = s if fired else None
                    fall_terms = tuple(fired)
                    if terminated_at is not None:
                        slot_rewards[e][s + 1:] = 0.0
                    norm_ret = S.rollout_normalized_return(
                        slot_rewards[e], meta.n_steps, terminated_at,
                    )
                    res = V2ClipResult(
                        clip=meta.clip_id,
                        normalized_return=norm_ret,
                        failed=terminated_at is not None,
                        steps=meta.n_steps,
                        alive_steps=alive_steps,
                        fall_step=None if terminated_at is None else terminated_at + 1,
                        fall_terms=fall_terms,
                        mean_term_score={k: v / max(alive_steps, 1) for k, v in sorted(tsums.items())},
                        wall_s=time.perf_counter() - slot_start_wall[e],
                        device=device,
                        performance={},
                        runtime=dict(runtime_info),
                    )
                    completed[meta.clip_id] = res
                    if ckpt_dir:
                        part_path = ckpt_dir / f"{meta.clip_id}.json"
                        tmp_path = ckpt_dir / f"{meta.clip_id}.json.tmp"
                        tmp_path.write_text(
                            json.dumps({
                                "clip": meta.clip_id,
                                "metadata": recovery_metadata or {},
                                "result": res.to_dict(),
                            }, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        tmp_path.replace(part_path)
                    if progress is not None:
                        progress(res)

                    if next_clip_queue_idx < len(bank):
                        nxt = next_clip_queue_idx
                        next_clip_queue_idx += 1
                        slot_clip_idx[e] = nxt
                        slot_steps[e] = 0
                        slot_rewards[e] = np.zeros(bank.metas[nxt].n_steps, dtype=np.float64)
                        slot_term_sums[e] = {}
                        slot_start_wall[e] = time.perf_counter()
                        command._clip_starts[e] = bank.metas[nxt].start
                        command._clip_ends[e] = bank.metas[nxt].end
                        prev_actions_gpu[e].zero_()
                        prev_actions_cpu[e].fill(0.0)
                        reset_slots.append(e)
                        reassigned_slots.append(e)
                        reassigned_clips.append(nxt)
                    else:
                        slot_active[e] = False
                        if fired:
                            reset_slots.append(e)

            if reset_slots:
                env.reset(env_ids=torch.tensor(reset_slots, dtype=torch.long, device=device))
                if reassigned_slots:
                    batched_policy.reset_slots(reassigned_slots, reassigned_clips)

            step_times.append(time.perf_counter() - t_iter0)
            global_iter += 1
    finally:
        env.close()

    rollout_wall_s = time.perf_counter() - rollout_t0
    steady_times = step_times[warmup_steps:] if len(step_times) > warmup_steps else step_times
    steady_wall_s = float(sum(steady_times))
    steady_fps = float(steady_env_steps / steady_wall_s) if steady_wall_s > 0 else 0.0
    total_fps = float(executed_env_steps / rollout_wall_s) if rollout_wall_s > 0 else 0.0

    # Stamp shared batched performance metadata on each newly evaluated result
    for clip_id, res in completed.items():
        res.performance = {
            "batched": True,
            "batch_envs": num_envs,
            "policy_backend": policy_backend,
            "cuda_env_build_s": float(env_build_s),
            "reset_wall_s": float(reset_s),
            "reference_load_s": float(ref_load_s),
            "warmup_steps": int(warmup_steps),
            "steady_steps": int(steady_env_steps),
            "steady_wall_s": steady_wall_s,
            "steady_simulation_fps": steady_fps,
            "executed_env_steps": int(executed_env_steps),
            "rollout_wall_s": float(rollout_wall_s),
            "aggregate_executed_fps": total_fps,
        }
        if ckpt_dir:
            part_path = ckpt_dir / f"{clip_id}.json"
            tmp_path = ckpt_dir / f"{clip_id}.json.tmp"
            tmp_path.write_text(
                json.dumps({
                    "clip": clip_id,
                    "metadata": recovery_metadata or {},
                    "result": res.to_dict(),
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(part_path)

    all_results = [cached[c] if c in cached else completed[c] for c in ids]
    meta_summary = {
        "batch_envs": num_envs,
        "batch_resolution": batch_res,
        "policy_backend": policy_backend,
        "cuda_env_build_s": float(env_build_s),
        "reset_wall_s": float(reset_s),
        "reference_load_s": float(ref_load_s),
        "rollout_wall_s": float(rollout_wall_s),
        "executed_steps": int(executed_env_steps),
        "steady_steps": int(steady_env_steps),
        "steady_wall_s": steady_wall_s,
        "steady_aggregate_steps_per_s": steady_fps,
        "executed_steps_per_s": total_fps,
        "resumed_clips": len(cached),
        "evaluated_clips": len(completed),
    }
    return all_results, meta_summary
