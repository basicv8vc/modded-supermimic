"""Deterministic closed-loop rollout of the official mjlab G1 tracking task.

This module is glue only. Physics, actuator/PD targets, rewards, terminations
and motion tracking all execute mjlab v1.6.0 (commit
``0fb8a681136be94ffc636a3dd423cabb97d91f10``) code: ``ManagerBasedRlEnv`` with
``MotionCommand`` and the tracking MDP's reward/termination managers. Nothing
here re-implements them.

Glue responsibilities (everything the frozen benchmark protocol needs on top
of the official env):

* build ``Mjlab-Tracking-Flat-Unitree-G1`` in play mode with the determinism
  overrides the protocol requires -- ``sampling_mode="start"``, no startup
  events (encoder bias / base com / foot friction), no observation corruption,
  ``auto_reset=False`` (the rollout driver stops at termination and keeps the
  terminal state);
* map the algorithm-agnostic policy contract onto env tensors: a 29-DoF action
  in ``C.JOINT_ORDER`` (the action manager's target order is the entity's
  natural joint order, i.e. the same order) and a :class:`StepInput` assembled
  from robot proprioception + the reference anchor, exactly like mjlab's
  ``motion_anchor_pos_b`` / ``motion_anchor_ori_b`` observation terms;
* accumulate the frozen score: per-term ``weight * dt * value`` contributions,
  the terminating step's reward still counts, ``time_out`` is not a failure.

Frame-index semantics are mjlab's own: ``env.reset()`` ends with
``command_manager.compute(dt=0.0)``, which advances ``MotionCommand.time_steps``
to 1, so the first scored reference frame is 1 and frame 0 is only the initial
condition. The pre-refactor evaluator scored frames 0..T-2 instead; that
difference is quantified in ``reports/alignment_v1_vs_mjlab.json`` and is not
papered over here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .motion import MotionReference
from .policy_api import ReferenceClip, StepInput


@dataclass
class StepInfo:
    terminated: bool
    terminated_terms: tuple[str, ...]
    reward: float
    terms: dict[str, float] = field(default_factory=dict)


class TrackingEnv:
    """One clip, one env: the official tracking task driven step by step."""

    def __init__(self, reference: MotionReference):
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.tracking.config.g1.env_cfgs import (
            unitree_g1_flat_tracking_env_cfg,
        )
        from mjlab.tasks.tracking.mdp.commands import MotionCommand
        from mjlab.tasks.tracking.mdp.observations import (
            motion_anchor_ori_b,
            motion_anchor_pos_b,
        )
        from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

        if reference.npz_path is None:
            raise ValueError(
                "reference has no npz_path; load it through motion.get_reference()"
            )

        self._torch = torch
        self._obs_anchor_pos_b = motion_anchor_pos_b
        self._obs_anchor_ori_b = motion_anchor_ori_b
        self._matrix_from_quat = matrix_from_quat
        self._subtract_frame_transforms = subtract_frame_transforms
        self.ref = reference
        self.reference_view = ReferenceClip.from_motion(reference)

        cfg = unitree_g1_flat_tracking_env_cfg(play=True)
        cfg.scene.num_envs = 1
        cfg.seed = 0
        cfg.auto_reset = False
        motion_cfg = cfg.commands["motion"]
        motion_cfg.motion_file = str(reference.npz_path)
        motion_cfg.sampling_mode = "start"
        # ``play=True`` zeroes pose_range/velocity_range but keeps the RSI
        # joint-position randomization (MotionCommandCfg.joint_position_range
        # defaults to +/-0.52 rad). The frozen protocol resets to the reference
        # frame exactly, so it must be disabled here.
        motion_cfg.joint_position_range = (0.0, 0.0)
        # Frozen protocol: deterministic rollout, no randomization at all.
        cfg.events = {}
        cfg.observations["actor"].enable_corruption = False

        self._env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        self._command = self._env.command_manager.get_term("motion")
        assert isinstance(self._command, MotionCommand)
        self._action_term = self._env.action_manager.get_term("joint_pos")

        mj_model = self._env.sim.mj_model
        self._imu_lin_adr = self._sensor_adr(mj_model, "imu_lin_vel")
        self._imu_ang_adr = self._sensor_adr(mj_model, "imu_ang_vel")

        self._action = np.zeros(C.ACTION_DIM)
        self._prev_action = np.zeros(C.ACTION_DIM)
        self._step_count = 0
        self._obs_buf = None
        self.reset()

    # ------------------------------------------------------------- contract
    @property
    def default_joint_pos(self) -> np.ndarray:
        return (
            self._action_term.offset.detach().cpu().numpy().reshape(-1).astype(np.float64)
        )

    @property
    def action_scale(self) -> np.ndarray:
        scale = self._action_term.scale
        return np.asarray(
            scale.detach().cpu().numpy() if hasattr(scale, "detach") else scale,
            dtype=np.float64,
        ).reshape(-1)

    @property
    def ref_index(self) -> int:
        """Reference frame the *upcoming* reward/termination compare against."""
        return int(self._command.time_steps[0])

    def state_snapshot(self) -> dict[str, np.ndarray]:
        """Read-only kinematic state, for cross-implementation comparisons."""
        qpos = self._env.sim.data.qpos[0].cpu().numpy().astype(np.float64)
        return {
            "root_pos": qpos[0:3].copy(),
            "root_quat": qpos[3:7].copy(),
            "joint_pos": self._joint_pos(),
            "joint_vel": self._joint_vel(),
            "anchor_pos_w": self._command.robot_anchor_pos_w[0].cpu().numpy().astype(np.float64),
        }

    # ---------------------------------------------------------------- reset
    def reset(self) -> np.ndarray:
        self._obs_buf, _ = self._env.reset()
        self._action = np.zeros(C.ACTION_DIM)
        self._prev_action = np.zeros(C.ACTION_DIM)
        self._step_count = 0
        return self.observation()

    # ----------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> StepInfo:
        action = np.asarray(action, dtype=np.float64).reshape(C.ACTION_DIM)
        self._prev_action = self._action
        self._action = action.copy()

        torch_action = self._torch.tensor(
            action, dtype=self._torch.float32, device=self._env.device
        ).view(1, C.ACTION_DIM)
        self._obs_buf, reward, terminated, _truncated, _extras = self._env.step(
            torch_action
        )
        self._step_count += 1

        raw_terms = {
            name: float(value[0])
            for name, value in self._env.reward_manager.get_active_iterable_terms(0)
        }
        # The benchmark records per-term ``weight * dt * value`` contributions;
        # mjlab's iterable terms are ``weight * value``.
        terms = {name: value * C.STEP_DT for name, value in raw_terms.items()}
        total = float(reward[0])

        fired: list[str] = []
        for name in self._env.termination_manager._term_names:  # noqa: SLF001
            if name == "time_out":
                continue
            if bool(self._env.termination_manager.get_term(name)[0]):
                fired.append(name)
        if not self._is_finite():
            fired.append("nonfinite")

        return StepInfo(
            terminated=bool(fired),
            terminated_terms=tuple(fired),
            reward=total,
            terms=terms,
        )

    # -------------------------------------------------------- policy inputs
    def step_input(self) -> StepInput:
        """Everything a policy is allowed to see at this control step.

        This -- not any fixed observation encoding -- is the interface. The
        anchor fields come from mjlab's own ``motion_anchor_pos_b`` /
        ``motion_anchor_ori_b`` terms (the same quantities the official actor
        observation exposes); the IMU velocities are the g1.xml sensors of the
        same names the reference implementation read.

        ``anchor_*_prev`` are the same two quantities one reference frame
        earlier (see :meth:`_anchor_relative`), because mjlab's terms always
        read the anchor at ``time_steps`` -- the *upcoming* frame -- while an
        algorithm whose observation describes its current state needs the
        frame that state belongs to.
        """
        pos_b = self._obs_anchor_pos_b(self._env, "motion")[0].cpu().numpy()
        ori_b = self._obs_anchor_ori_b(self._env, "motion")[0].cpu().numpy()
        prev_pos_b, prev_ori_b = self._anchor_relative(self.ref_index - 1)
        sensordata = self._env.sim.data.sensordata[0].cpu().numpy().astype(np.float64)

        return StepInput(
            step=self._step_count,
            reference_index=self.ref_index,
            time_s=self._step_count * C.STEP_DT,
            joint_pos=self._joint_pos(),
            joint_vel=self._joint_vel(),
            base_lin_vel=sensordata[self._imu_lin_adr : self._imu_lin_adr + 3].copy(),
            base_ang_vel=sensordata[self._imu_ang_adr : self._imu_ang_adr + 3].copy(),
            anchor_pos_b=np.asarray(pos_b, dtype=np.float64).reshape(3).copy(),
            anchor_ori_b=np.asarray(ori_b, dtype=np.float64).reshape(6).copy(),
            previous_action=self._action.copy(),
            reference=self.reference_view,
            anchor_pos_b_prev=prev_pos_b,
            anchor_ori_b_prev=prev_ori_b,
        )

    def _anchor_relative(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Reference anchor in the robot anchor frame at an **explicit** frame.

        ``motion_anchor_pos_b`` / ``motion_anchor_ori_b`` read the command's
        ``anchor_pos_w`` / ``anchor_quat_w`` properties, which are indexed at
        ``time_steps`` and therefore always describe ``reference_index``. There
        is no mjlab observation term for another frame, so this reproduces the
        same transform with mjlab's own helpers and the command's own loader
        (``subtract_frame_transforms`` + ``matrix_from_quat``, the pair the
        observation term uses), only with the reference index passed in.

        Returns ``(pos_b (3,), ori_6 (6,))`` in the same layouts the mjlab terms
        emit, so a policy can use either frame interchangeably.
        """
        frame = int(max(0, min(frame, self.ref.num_frames - 1)))
        command = self._command
        idx = int(command.motion_anchor_body_index)
        with self._torch.no_grad():
            ref_pos = (
                command.motion.body_pos_w[frame, idx]
                + self._env.scene.env_origins[0]
            )
            ref_quat = command.motion.body_quat_w[frame, idx]
            pos_b, quat_b = self._subtract_frame_transforms(
                command.robot_anchor_pos_w[0],
                command.robot_anchor_quat_w[0],
                ref_pos,
                ref_quat,
            )
            mat = self._matrix_from_quat(quat_b)
        return (
            np.asarray(pos_b.detach().cpu().numpy(), dtype=np.float64).reshape(3),
            np.asarray(mat[..., :2].reshape(-1).detach().cpu().numpy(), dtype=np.float64),
        )

    def observation(self) -> np.ndarray:
        """The reference implementation's 160-dim actor encoding of ``step_input``.

        Kept for mjlab-native ``.npz``/``.pt`` actors (``benchmark.evaluator.policy``);
        it is one possible encoding of :meth:`step_input`, not a requirement.
        """
        from .policy import mjlab_observation  # noqa: PLC0415  (import cycle)

        return mjlab_observation(self.step_input(), self.default_joint_pos)

    # ----------------------------------------------------------- internals
    @staticmethod
    def _sensor_adr(mj_model, name: str) -> int:
        """Sensordata offset of a g1.xml sensor (mjlab prefixes entity names)."""
        import mujoco

        for candidate in (name, f"robot/{name}"):
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, candidate)
            if sid >= 0:
                return int(mj_model.sensor_adr[sid])
        raise KeyError(f"sensor {name!r} not found in the compiled model")

    def _joint_pos(self) -> np.ndarray:
        return (
            self._env.scene["robot"]
            .data.joint_pos[0]
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        )

    def _joint_vel(self) -> np.ndarray:
        return (
            self._env.scene["robot"]
            .data.joint_vel[0]
            .cpu()
            .numpy()
            .astype(np.float64)
            .copy()
        )

    def _is_finite(self) -> bool:
        qpos = self._env.sim.data.qpos[0].cpu().numpy()
        qvel = self._env.sim.data.qvel[0].cpu().numpy()
        return bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
