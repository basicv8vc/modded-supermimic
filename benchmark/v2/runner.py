"""V2 rollout driver over the official mjlab environment.

This is glue only: physics, PD/actuators, contact set, motion loading and the
8 official reward terms all run in mjlab 1.6.0.  The only V2-specific pieces
are (a) the cfg from :func:`benchmark.v2.config.build_v2_env_cfg`,
(b) collecting the termination reasons, and (c) the **unclamped** full-clip
normalized return from :mod:`benchmark.v2.spec`.

V2's formal evaluator is **CUDA-only** (task prompt §3).  ``V2TrackingEnv``
asks :mod:`benchmark.v2.runtime` to validate the CUDA/Warp/MuJoCo Warp stack
before constructing ``ManagerBasedRlEnv``, then verifies that the constructed
env holds CUDA simulation state and a captured CUDA step graph.  Any missing
piece raises ``CudaRuntimeUnavailable`` -- there is no silent CPU fallback.
Only policy I/O crosses to the host (``numpy`` via ``.cpu()``); physics,
contact, PD, reward and termination stay on the GPU.

``benchmark/evaluator/*`` is imported read-only (motion converter, policy
contract, manifest/version helpers) and never modified.  V1's CPU env glue is
left untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..evaluator import constants as C
from ..evaluator import policy_api
from ..evaluator.manifest import load_manifest
from ..evaluator.motion import CONVERTER_VERSION, MotionReference, get_reference
from ..evaluator.policy_api import ClipContext
from ..evaluator.runner import dependency_versions, source_git_sha
from ..evaluator.spec import ModelInfo
from . import config as V2CFG
from . import runtime as RT
from . import spec as S


@dataclass
class V2StepInfo:
    terminated: bool
    terminated_terms: tuple[str, ...]
    reward: float
    terms: dict[str, float] = field(default_factory=dict)


@dataclass
class V2ClipResult:
    clip: str
    normalized_return: float       # raw, unclamped
    failed: bool
    steps: int
    alive_steps: int
    fall_step: int | None
    fall_terms: tuple[str, ...]
    mean_term_score: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    device: str = ""
    performance: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    def record_row(self) -> dict:
        return {"clip": self.clip, "normalized_return": self.normalized_return, "failed": self.failed}

    def to_dict(self) -> dict:
        return {
            "clip": self.clip,
            "normalized_return": self.normalized_return,
            "failed": self.failed,
            "steps": self.steps,
            "alive_steps": self.alive_steps,
            "fall_step": self.fall_step,
            "fall_terms": list(self.fall_terms),
            "mean_term_score": self.mean_term_score,
            "wall_s": self.wall_s,
            "device": self.device,
            "performance": self.performance,
            "runtime": self.runtime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> V2ClipResult:
        return cls(
            clip=d["clip"],
            normalized_return=float(d["normalized_return"]),
            failed=bool(d["failed"]),
            steps=int(d["steps"]),
            alive_steps=int(d["alive_steps"]),
            fall_step=None if d.get("fall_step") is None else int(d["fall_step"]),
            fall_terms=tuple(d.get("fall_terms") or ()),
            mean_term_score={k: float(v) for k, v in (d.get("mean_term_score") or {}).items()},
            wall_s=float(d.get("wall_s", 0.0)),
            device=str(d.get("device", "")),
            performance=dict(d.get("performance") or {}),
            runtime=dict(d.get("runtime") or {}),
        )


class V2TrackingEnv:
    """One clip, one env: official mjlab tracking task with the V2 cfg.

    Mirrors PR #12's ``benchmark/evaluator/env.py`` glue so that the policy
    contract (including the PR #13 ``anchor_*_prev`` time-index fix) is
    identical.  V2 does not change observations, actions, or physics.
    """

    def __init__(self, reference: MotionReference, device: str = RT.DEFAULT_DEVICE,
                 profile: bool = False):
        import torch  # noqa: PLC0415
        from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
        from mjlab.tasks.tracking.mdp.commands import MotionCommand  # noqa: PLC0415
        from mjlab.tasks.tracking.mdp.observations import (  # noqa: PLC0415
            motion_anchor_ori_b, motion_anchor_pos_b,
        )
        from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms  # noqa: PLC0415

        # P0: hard CUDA-only selection.  probe_cuda raises CudaRuntimeUnavailable
        # rather than falling back to the CPU MuJoCo Warp backend.
        self.probe = RT.probe_cuda(device)
        self.device = RT.require_cuda_device(device)
        self._profile = bool(profile)
        self._torch = torch
        self._obs_anchor_pos_b = motion_anchor_pos_b
        self._obs_anchor_ori_b = motion_anchor_ori_b
        self._matrix_from_quat = matrix_from_quat
        self._subtract_frame_transforms = subtract_frame_transforms
        self.ref = reference
        self.reference_view = policy_api.ReferenceClip.from_motion(reference)
        self.timings: dict[str, Any] = {"step_times": [], "reset_s": 0.0, "env_build_s": 0.0}

        cfg = V2CFG.build_v2_env_cfg(play=True)
        cfg.commands["motion"].motion_file = str(reference.npz_path)
        t0 = time.perf_counter()
        self._env = ManagerBasedRlEnv(cfg=cfg, device=self.device)
        self.timings["env_build_s"] = time.perf_counter() - t0
        # Prove the constructed env holds CUDA simulation state + CUDA graphs.
        self.runtime = RT.inspect_runtime(self._env)
        self._command = self._env.command_manager.get_term("motion")
        assert isinstance(self._command, MotionCommand)
        self._action_term = self._env.action_manager.get_term("joint_pos")

        mj_model = self._env.sim.mj_model
        self._imu_lin_adr = self._sensor_adr(mj_model, "imu_lin_vel")
        self._imu_ang_adr = self._sensor_adr(mj_model, "imu_ang_vel")

        self._action = np.zeros(C.ACTION_DIM)
        self._step_count = 0
        self._obs_buf = None
        # Phase breakdown (only accumulated when profile=True).  Inert for
        # scoring; used by scripts/v2/profile_breakdown.py to attribute the
        # per-control-step wall clock (P0 bottleneck analysis).
        self.timings["phase_times"] = {
            "step_input_s": 0.0, "policy_s": 0.0, "env_step_s": 0.0,
            "reward_read_s": 0.0, "term_read_s": 0.0, "finite_s": 0.0,
        }
        self.reset()

    # ------------------------------------------------------------- contract
    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._action_term.offset.detach().cpu().numpy().reshape(-1).astype(np.float64)

    @property
    def action_scale(self) -> np.ndarray:
        scale = self._action_term.scale
        return np.asarray(
            scale.detach().cpu().numpy() if hasattr(scale, "detach") else scale,
            dtype=np.float64,
        ).reshape(-1)

    @property
    def ref_index(self) -> int:
        return int(self._command.time_steps[0])

    def reset(self) -> np.ndarray:
        t0 = time.perf_counter()
        self._obs_buf, _ = self._env.reset()
        if self._profile:
            self._torch.cuda.synchronize(self.device)
        self.timings["reset_s"] = time.perf_counter() - t0
        self._action = np.zeros(C.ACTION_DIM)
        self._step_count = 0
        return self.observation()

    def step(self, action: np.ndarray) -> V2StepInfo:
        action = np.asarray(action, dtype=np.float64).reshape(C.ACTION_DIM)
        prev_action = self._action
        self._action = action.copy()
        torch_action = self._torch.tensor(
            action, dtype=self._torch.float32, device=self._env.device
        ).view(1, C.ACTION_DIM)
        t0 = time.perf_counter()
        self._obs_buf, reward, _terminated, _truncated, _extras = self._env.step(torch_action)
        if self._profile:
            self._torch.cuda.synchronize(self.device)
        self.timings["step_times"].append(time.perf_counter() - t0)
        self._step_count += 1

        # mjlab iterable terms are ``weight * value``; V2 records ``weight*dt*value``
        _t = time.perf_counter()
        raw_terms = {
            name: float(value[0])
            for name, value in self._env.reward_manager.get_active_iterable_terms(0)
        }
        terms = {name: value * C.STEP_DT for name, value in raw_terms.items()}
        total = float(reward[0])
        if self._profile:
            self.timings["phase_times"]["reward_read_s"] += time.perf_counter() - _t

        _t = time.perf_counter()
        fired: list[str] = []
        for name in self._env.termination_manager._term_names:  # noqa: SLF001
            if name == "time_out":
                continue  # clip end is success
            if bool(self._env.termination_manager.get_term(name)[0]):
                fired.append(name)
        if self._profile:
            self.timings["phase_times"]["term_read_s"] += time.perf_counter() - _t
        # Belt-and-braces nonfinite guard (same semantics as the V2 glue term).
        _t = time.perf_counter()
        if not self._is_finite() and "nonfinite" not in fired:
            fired.append("nonfinite")
        if self._profile:
            self.timings["phase_times"]["finite_s"] += time.perf_counter() - _t
        return V2StepInfo(bool(fired), tuple(fired), total, terms)

    def step_input(self) -> policy_api.StepInput:
        _t = time.perf_counter()
        pos_b = self._obs_anchor_pos_b(self._env, "motion")[0].cpu().numpy()
        ori_b = self._obs_anchor_ori_b(self._env, "motion")[0].cpu().numpy()
        prev_pos_b, prev_ori_b = self._anchor_relative(self.ref_index - 1)
        sensordata = self._env.sim.data.sensordata[0].cpu().numpy().astype(np.float64)
        result = policy_api.StepInput(
            step=self._step_count,
            reference_index=self.ref_index,
            time_s=self._step_count * C.STEP_DT,
            joint_pos=self._joint_pos(),
            joint_vel=self._joint_vel(),
            base_lin_vel=sensordata[self._imu_lin_adr:self._imu_lin_adr + 3].copy(),
            base_ang_vel=sensordata[self._imu_ang_adr:self._imu_ang_adr + 3].copy(),
            anchor_pos_b=np.asarray(pos_b, dtype=np.float64).reshape(3).copy(),
            anchor_ori_b=np.asarray(ori_b, dtype=np.float64).reshape(6).copy(),
            previous_action=self._action.copy(),
            reference=self.reference_view,
            anchor_pos_b_prev=prev_pos_b,
            anchor_ori_b_prev=prev_ori_b,
        )
        if self._profile:
            self.timings["phase_times"]["step_input_s"] += time.perf_counter() - _t
        return result

    def _anchor_relative(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        frame = int(max(0, min(frame, self.ref.num_frames - 1)))
        command = self._command
        idx = int(command.motion_anchor_body_index)
        with self._torch.no_grad():
            ref_pos = command.motion.body_pos_w[frame, idx] + self._env.scene.env_origins[0]
            ref_quat = command.motion.body_quat_w[frame, idx]
            pos_b, quat_b = self._subtract_frame_transforms(
                command.robot_anchor_pos_w[0], command.robot_anchor_quat_w[0], ref_pos, ref_quat,
            )
            mat = self._matrix_from_quat(quat_b)
        return (
            np.asarray(pos_b.detach().cpu().numpy(), dtype=np.float64).reshape(3),
            np.asarray(mat[..., :2].reshape(-1).detach().cpu().numpy(), dtype=np.float64),
        )

    def observation(self) -> np.ndarray:
        from ..evaluator.policy import mjlab_observation  # noqa: PLC0415
        return mjlab_observation(self.step_input(), self.default_joint_pos)

    # ------------------------------------------------- GPU runtime parity I/O
    def runtime_state(self) -> S.V2State:
        """Read the *real* GPU state into the independent V2 oracle's inputs.

        Nothing here reads RewardManager outputs: the arrays come straight from
        ``MotionCommand`` (reference + robot body state), ``sim.data`` (pelvis
        free joint), the action manager (raw action history) and the entity's
        soft joint limits.  This is the A/B split required by task prompt §6.
        """
        def np64(t):
            return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)

        command = self._command
        robot = self._env.scene["robot"]
        qpos = np64(self._env.sim.data.qpos[0])
        qvel = np64(self._env.sim.data.qvel[0])
        pelvis_idx = command.cfg.body_names.index(S.TERMINATION_ROOT)
        ref_pos = np64(command.body_pos_w[0])
        ref_quat = np64(command.body_quat_w[0])
        soft = np64(robot.data.soft_joint_pos_limits[0])
        return S.V2State(
            robot_body_pos=np64(command.robot_body_pos_w[0]),
            robot_body_quat=np64(command.robot_body_quat_w[0]),
            robot_body_lin_vel=np64(command.robot_body_lin_vel_w[0]),
            robot_body_ang_vel=np64(command.robot_body_ang_vel_w[0]),
            ref_body_pos=ref_pos,
            ref_body_quat=ref_quat,
            ref_body_lin_vel=np64(command.body_lin_vel_w[0]),
            ref_body_ang_vel=np64(command.body_ang_vel_w[0]),
            robot_root_z=float(qpos[2]),
            robot_root_quat=qpos[3:7].copy(),
            ref_root_z=float(ref_pos[pelvis_idx, 2]),
            ref_root_quat=ref_quat[pelvis_idx].copy(),
            joint_pos=np64(robot.data.joint_pos[0]),
            action=np64(self._env.action_manager.action[0]),
            prev_action=np64(self._env.action_manager.prev_action[0]),
            soft_lo=soft[:, 0].copy(),
            soft_hi=soft[:, 1].copy(),
            qpos=qpos,
            qvel=qvel,
        )

    def runtime_relative_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """The re-anchored reference the GPU RewardManager actually consumes.

        ``MotionCommand.update_relative_body_poses`` (mjlab
        ``commands.py:379-407``) stores two CUDA tensors,
        ``body_pos_relative_w`` / ``body_quat_relative_w``; the official body
        position / orientation reward terms read them directly
        (``rewards.py:39-70``).  They are **reference state**, not a
        RewardManager output, so the independent V2 oracle may use them as its
        input while still computing all errors / weights / dt itself.  This is
        what makes the runtime A/B comparison use the identical reference
        frame (task prompt §6).
        """
        command = self._command
        return (
            np.asarray(command.body_pos_relative_w[0].detach().cpu().numpy(),
                       dtype=np.float64),
            np.asarray(command.body_quat_relative_w[0].detach().cpu().numpy(),
                       dtype=np.float64),
        )

    def runtime_terms(self) -> dict[str, float]:
        """Weighted-then-dt-scaled RewardManager terms for the *last* step.

        mjlab's ``get_active_iterable_terms`` returns ``weight * value`` (the
        manager stores ``value / scale``); V2's oracle returns
        ``dt * weight * value``.  Multiplying by ``STEP_DT`` is the only
        conversion -- no tolerance fudging, no reading the manager as oracle.
        """
        return {name: float(value[0]) * C.STEP_DT
                for name, value in self._env.reward_manager.get_active_iterable_terms(0)}

    def runtime_terminations(self) -> dict[str, bool]:
        return {name: bool(self._env.termination_manager.get_term(name)[0])
                for name in self._env.termination_manager._term_names}  # noqa: SLF001

    def performance(self, warmup_steps: int = 5) -> dict[str, Any]:
        """Steady-state FPS excludes CUDA init / reference load / JIT warmup."""
        times = list(self.timings["step_times"])
        steady = times[warmup_steps:] if len(times) > warmup_steps else times
        total = float(sum(times))
        return {
            "cuda_env_build_s": float(self.timings["env_build_s"]),
            "reset_wall_s": float(self.timings["reset_s"]),
            "warmup_steps": warmup_steps,
            "warmup_wall_s": float(sum(times[:warmup_steps])),
            "first_step_wall_s": float(times[0]) if times else 0.0,
            "steady_steps": len(steady),
            "steady_wall_s": float(sum(steady)),
            "steady_simulation_fps": float(len(steady) / sum(steady)) if steady and sum(steady) > 0 else 0.0,
            "step_wall_s_mean": float(total / len(times)) if times else 0.0,
            "step_wall_s_max": float(max(times)) if times else 0.0,
            "total_steps": len(times),
            "total_step_wall_s": total,
            "phase_times": dict(self.timings.get("phase_times") or {}) if self._profile else {},
        }

    # ----------------------------------------------------------- internals
    @staticmethod
    def _sensor_adr(mj_model, name: str) -> int:
        import mujoco  # noqa: PLC0415
        for candidate in (name, f"robot/{name}"):
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, candidate)
            if sid >= 0:
                return int(mj_model.sensor_adr[sid])
        raise KeyError(f"sensor {name!r} not found")

    def _joint_pos(self) -> np.ndarray:
        return self._env.scene["robot"].data.joint_pos[0].cpu().numpy().astype(np.float64).copy()

    def _joint_vel(self) -> np.ndarray:
        return self._env.scene["robot"].data.joint_vel[0].cpu().numpy().astype(np.float64).copy()

    def _is_finite(self) -> bool:
        qpos = self._env.sim.data.qpos[0].cpu().numpy()
        qvel = self._env.sim.data.qvel[0].cpu().numpy()
        return bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))


# --------------------------------------------------------------------------
# Rollout + scoring
# --------------------------------------------------------------------------

def evaluate_clip(info: ModelInfo, ref: MotionReference, policy, clip_id: str,
                  device: str = RT.DEFAULT_DEVICE, profile: bool = False,
                  max_steps: int | None = None) -> V2ClipResult:
    started = time.perf_counter()
    policy_api.require_interface(policy)
    env = V2TrackingEnv(ref, device=device, profile=profile)
    scale_err = float(np.abs(env.action_scale - info.action_scale).max())
    pose_err = float(np.abs(env.default_joint_pos - info.default_joint_pos).max())
    if scale_err > 1e-6 or pose_err > 1e-6:
        raise RuntimeError(f"{clip_id}: live mjlab action mapping drifted ({scale_err:.3e}, {pose_err:.3e})")
    policy.reset(ClipContext(
        clip=clip_id,
        joint_order=C.JOINT_ORDER,
        default_joint_pos=np.array(info.default_joint_pos, dtype=np.float64),
        action_scale=np.array(info.action_scale, dtype=np.float64),
        control_dt=C.STEP_DT,
        reference=env.reference_view,
    ))

    n_steps = ref.num_frames - 1     # full clip, T-1 control steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))  # throughput probe only
    if n_steps < 1:
        raise ValueError(f"{clip_id}: clip shorter than two frames")

    rewards = np.zeros(n_steps, dtype=np.float64)
    term_sums: dict[str, float] = {}
    alive_steps = 0
    terminated_at: int | None = None
    fall_terms: tuple[str, ...] = ()

    for step in range(n_steps):
        step_in = env.step_input()
        _t = time.perf_counter()
        action = policy_api.check_action(policy.act(step_in), clip=clip_id, step=step)
        if profile:
            env.timings["phase_times"]["policy_s"] += time.perf_counter() - _t
        result = env.step(action)
        rewards[step] = result.reward
        alive_steps += 1
        for name, value in result.terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + value
        if result.terminated:
            terminated_at = step
            fall_terms = result.terminated_terms
            rewards[step + 1:] = 0.0    # remaining steps contribute exactly zero
            break

    normalized = S.rollout_normalized_return(rewards, n_steps, terminated_at)
    return V2ClipResult(
        clip=clip_id,
        normalized_return=normalized,       # NO [0,1] clamp
        failed=terminated_at is not None,
        steps=n_steps,
        alive_steps=alive_steps,
        fall_step=None if terminated_at is None else terminated_at + 1,
        fall_terms=fall_terms,
        mean_term_score={k: v / max(alive_steps, 1) for k, v in sorted(term_sums.items())},
        wall_s=time.perf_counter() - started,
        device=env.device,
        performance=env.performance(),
        runtime=dict(env.runtime),
    )


def evaluate_all(info: ModelInfo, policy, clip_ids: Sequence[str] | None = None,
                 data_dir: Path | None = None, use_cache: bool = False,
                 progress: Callable[[V2ClipResult], None] | None = None,
                 device: str = RT.DEFAULT_DEVICE, profile: bool = False,
                 checkpoint_dir: Path | None = None,
                 recovery_metadata: dict | None = None,
                 max_steps: int | None = None) -> list[V2ClipResult]:
    import json  # noqa: PLC0415
    ids = list(clip_ids or S.load_protocol_v2()["clip_ids"])
    root = data_dir or C.DATA_DIR
    out: list[V2ClipResult] = []
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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
                        out.append(res)
                        if progress is not None:
                            progress(res)
                        continue
                except Exception:
                    pass  # Corrupted or mismatched part -> re-evaluate

        t_ref = time.perf_counter()
        ref = get_reference(root / f"{clip}.csv", use_cache=use_cache)
        result = evaluate_clip(info, ref, policy, clip, device=device, profile=profile,
                               max_steps=max_steps)
        result.performance["reference_load_s"] = time.perf_counter() - t_ref - result.wall_s
        result.performance["reference_load_s"] = max(result.performance["reference_load_s"], 0.0)

        if ckpt_dir:
            part_path = ckpt_dir / f"{clip}.json"
            tmp_path = ckpt_dir / f"{clip}.json.tmp"
            tmp_path.write_text(
                json.dumps({
                    "clip": clip,
                    "metadata": recovery_metadata or {},
                    "result": result.to_dict(),
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(part_path)

        out.append(result)
        if progress is not None:
            progress(result)
    return out


def build_record(*, algorithm: str, training_commit: str, seed: int, gpu_name: str,
                 gpu_count: int, checkpoint_sha256: str, training_wall_s: float,
                 setup_wall_s: float, results: Sequence[V2ClipResult], mode: str,
                 extra: dict | None = None, provenance: dict | None = None,
                 telemetry: dict | None = None) -> dict:
    """Assemble a V2 record with its own identity.  V1 records are untouched.

    ``gpu_name`` is only a *declared* CLI value.  When ``provenance`` is
    supplied (always, for a real CUDA run) the identity fields come from
    runtime detection instead, so a user-supplied string can never be used as
    proof of which GPU executed the physics (task prompt §9).
    """
    cfg = S.load_protocol_v2()
    manifest = load_manifest()
    versions = dependency_versions()
    identity = S.evaluator_identity()
    prov = provenance or {}
    hardware = {
        "gpu_name": prov.get("gpu_name", gpu_name),
        "gpu_count": prov.get("gpu_count", gpu_count),
        "declared_gpu_name": gpu_name,
        "gpu_compute_capability": prov.get("gpu_compute_capability"),
        "actual_torch_device": prov.get("actual_torch_device"),
        "actual_warp_device": prov.get("actual_warp_device"),
        "actual_simulation_backend": prov.get("actual_simulation_backend"),
        "cuda_graph_step_captured": prov.get("cuda_graph_step_captured"),
    }
    # Eligibility is hardware-qualified: a benchmark run only becomes a
    # leaderboard submission when runtime detection proves the target GPU
    # (RTX PRO 6000 / sm_120).  Non-target hardware (A100/sm_80, L4/sm_89)
    # is a controlled baseline: comparison_valid=True, submission_eligible=False.
    target_ok = bool(prov.get("target_gpu_match")) and bool(prov.get("sm_120_confirmed"))
    submission_ok = bool(mode == "benchmark" and target_ok)
    comparison_ok = bool(mode == "benchmark")
    return {
        "protocol_id": cfg["protocol_id"],
        "algorithm": algorithm,
        "training_commit": training_commit,
        "seed": seed,
        "hardware": hardware,
        "evaluator": {
            "implementation": cfg["evaluator_implementation"],
            "protocol_sha256": identity["protocol_sha256"],
            "evaluator_source_tree": cfg["evaluator_source_tree"],
            "source_git_sha": identity["evaluator_source_git_sha"],
            "self_collision": True,                       # physics: full
            "self_collision_reward_penalty": False,      # scoring: no penalty
            "robot_dof": C.ROBOT_DOF,
            **versions,
            "robot_model_sha256": manifest["robot_model"]["sha256"],
            "dataset_manifest_sha256": manifest["dataset"]["manifest_sha256"],
            "mjlab_commit": C.MJLAB_COMMIT,
            "mjlab_version": C.MJLAB_VERSION,
            "motion_converter_version": CONVERTER_VERSION,
            "reference_implementations": cfg["reference_implementations"],
            "depends_on_prs": cfg["depends_on_prs"],
        },
        "checkpoints": [{
            "checkpoint_sha256": checkpoint_sha256,
            "training_wall_s": training_wall_s,
            "setup_wall_s": setup_wall_s,
            "eval_wall_s": float(sum(r.wall_s for r in results)),
            "per_clip": [r.record_row() for r in results],
        }],
        "target_hardware_met": target_ok,
        "submission_eligible": submission_ok,
        "comparison_valid": comparison_ok,
        "evaluation": {
            "mode": mode,
            "submission_eligible": submission_ok,
            "comparison_valid": comparison_ok,
            "cuda_required": True,
            "provenance": prov or None,
            "telemetry": telemetry or None,
            "performance": {
                r.clip: r.performance for r in results if r.performance
            },
            "normalization": "full-clip (T-1)*0.1 denominator, no clipping",
            "control_hz": 1.0 / C.STEP_DT,
            "clip_steps": {r.clip: r.steps for r in results},
            "normalized_return_denominator": S.MAX_STEP_REWARD,
            "aggregate_normalized_return": S.aggregate(r.normalized_return for r in results),
            "failed_clips": S.failed_clips(r.failed for r in results),
            "per_clip_diagnostics": {
                r.clip: {"steps": r.steps, "alive_steps": r.alive_steps,
                         "fall_step": r.fall_step, "fall_terms": list(r.fall_terms),
                         "mean_term_score": r.mean_term_score}
                for r in results
            },
            **(extra or {}),
        },
    }
