"""Minimal inference interface between the evaluator and an algorithm.

The benchmark fixes the *task*: the robot, its physics, the reference motions,
the control interface and the scoring. It deliberately does **not** fix what a
policy observes, how it encodes history, which network it runs, what its
checkpoint contains, or which simulator trained it.

An algorithm plugs in by exposing two methods::

    def reset(self, context: ClipContext) -> None
    def act(self, step: StepInput) -> array-like, shape (29,)

``reset`` is called once per clip, before the first step. It receives the clip
name, the control contract (joint order, default pose, action scale, control
rate) and a read-only view of the reference motion. ``act`` is called once per
control step (50 Hz) and must return the 29 joint targets; the evaluator turns
them into position targets with ``action * action_scale + default_joint_pos``.

Everything else -- observation layout, history buffers, network, checkpoint
format, training loop -- is the algorithm's business. The evaluator only ever
sees the returned action, and it validates it (shape, dtype, finiteness) before
stepping the simulator.

The policy never receives the ``MjModel``/``MjData``, the reward terms or the
termination functions, so it cannot alter the physics or the score. It also
cannot read simulator state that a real robot would not have: the fields below
are exactly proprioception plus the reference motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C

ACTION_DIM = C.ROBOT_DOF


class ActionFormatError(ValueError):
    """The policy returned something that is not a valid 29-DoF command."""


class PolicyInterfaceError(RuntimeError):
    """The object handed to the evaluator does not implement reset()/act()."""


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True)
class ReferenceClip:
    """Read-only view of the retargeted motion the policy has to track.

    The full clip is exposed (not just the current frame) so that algorithms
    needing look-ahead are not penalised; every algorithm sees the same data,
    so this is not a privilege. It contains no simulator state.
    """

    num_frames: int
    control_dt: float
    joint_pos: np.ndarray  # (T, 29) absolute joint angles, radians
    joint_vel: np.ndarray  # (T, 29) radians / second
    root_pos_w: np.ndarray  # (T, 3) world position of the pelvis
    root_quat_w: np.ndarray  # (T, 4) wxyz

    @classmethod
    def from_motion(cls, reference) -> "ReferenceClip":
        return cls(
            num_frames=int(reference.num_frames),
            control_dt=float(C.STEP_DT),
            joint_pos=_readonly(reference.joint_pos),
            joint_vel=_readonly(reference.joint_vel),
            root_pos_w=_readonly(reference.body_pos_w[:, 0]),
            root_quat_w=_readonly(reference.body_quat_w[:, 0]),
        )


@dataclass(frozen=True)
class ClipContext:
    """Passed to ``reset`` once per clip.

    ``action_scale`` and ``default_joint_pos`` define the control interface and
    are identical for every submission: the final position target for joint ``i``
    is ``action[i] * action_scale[i] + default_joint_pos[i]``.
    """

    clip: str
    joint_order: tuple[str, ...]
    default_joint_pos: np.ndarray  # (29,) radians
    action_scale: np.ndarray  # (29,) radians per unit action
    control_dt: float  # seconds (0.02 -> 50 Hz)
    reference: ReferenceClip


@dataclass(frozen=True)
class StepInput:
    """Passed to ``act`` once per control step.

    All arrays are copies, so a policy may keep them. Values are physical
    quantities in the robot's own frame; how they are normalised, stacked or
    fed to a network is up to the algorithm.

    Two reference frames are live at a given step, and they are not the same
    frame:

    * ``reference_index`` -- the frame the *upcoming* reward and termination
      compare against (mjlab advances ``MotionCommand.time_steps`` at reset,
      so the first step has ``reference_index == 1`` while the robot still
      stands on frame 0);
    * ``reference_index - 1`` -- the frame the robot's *own state* belongs to,
      which is also the frame the previous step was scored at.

    mjlab's tracking observation terms read the anchor at ``reference_index``,
    so ``anchor_pos_b`` / ``anchor_ori_b`` carry that frame; the ``*_prev``
    fields carry the same two quantities one frame earlier, at
    ``reference_index - 1``. An algorithm that anchors at the frame its state
    belongs to must use the ``*_prev`` fields -- combining the frame-``f``
    goal features with the frame-``f+1`` anchor is a mixed-frame observation
    and is what ``benchmark/policies/nablamimic.py`` used to do.
    """

    step: int  # 0-based control step within the clip
    reference_index: int  # index into ``reference`` arrays (== frame + 1)
    time_s: float  # step * control_dt
    joint_pos: np.ndarray  # (29,) absolute qpos, radians
    joint_vel: np.ndarray  # (29,) radians / second
    base_lin_vel: np.ndarray  # (3,) IMU linear velocity, body frame
    base_ang_vel: np.ndarray  # (3,) IMU angular velocity, body frame
    anchor_pos_b: np.ndarray  # (3,) reference anchor at reference_index, robot anchor frame
    anchor_ori_b: np.ndarray  # (6,) first two columns of that rotation matrix
    previous_action: np.ndarray  # (29,) raw action issued on the previous step
    reference: ReferenceClip  # same object as ``context.reference``
    anchor_pos_b_prev: np.ndarray  # (3,) same quantity at reference_index - 1
    anchor_ori_b_prev: np.ndarray  # (6,) same quantity at reference_index - 1


def check_action(action, *, clip: str = "", step: int = -1) -> np.ndarray:
    """Validate a policy's return value and coerce it to ``float64[29]``.

    Rejects wrong width, wrong rank, non-numeric payloads and non-finite values.
    This is the only thing the evaluator requires of an action.
    """
    where = f"{clip or '<clip>'} step {step}"
    try:
        array = np.asarray(action)
    except Exception as exc:  # pragma: no cover - exotic __array__ failures
        raise ActionFormatError(f"{where}: action is not array-like ({exc})") from exc

    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ActionFormatError(
            f"{where}: action dtype {array.dtype} is not numeric"
        )
    if array.shape != (ACTION_DIM,):
        raise ActionFormatError(
            f"{where}: action shape {array.shape} != ({ACTION_DIM},)"
        )
    array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        bad = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise ActionFormatError(f"{where}: action holds {bad} non-finite value(s)")
    return array


def require_interface(policy) -> None:
    """Fail early, with a useful message, if the object is not a policy."""
    missing = [name for name in ("reset", "act") if not callable(getattr(policy, name, None))]
    if missing:
        raise PolicyInterfaceError(
            f"policy {type(policy).__name__} is missing {'/'.join(missing)}(); "
            "an algorithm must expose reset(context) and act(step) -> (29,) action"
        )
