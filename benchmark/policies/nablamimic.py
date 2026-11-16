"""NablaMimic's *native* inference, driven through the canonical evaluator.

This is not a re-encoding of NablaMimic into mjlab's 160-dim observation. It
rebuilds, from the public robot state and the reference motion, the exact
tensors the original implementation fed to its actor, and runs the original
``actor_ema`` weights unchanged.

Source of truth
---------------
``sailab-ai/NablaMimic`` @ ``bfbb2a6773b4776c17402a34a93ec7589d9c7c18`` (the
commit the evaluated checkpoint was trained at; ``config.json`` of run
``ea_s0_20261111T065918Z`` records the same SHA with ``git_dirty=false``):

=================================  ============================================
``src/nablamimic/env.py``          ``proprio`` (93), ``goal`` / ``goal_ref``
                                   (686), ``action_base`` / ``ctrl_of``
``src/nablamimic/nets.py``         ``actor_apply`` = encoder / decoder MLP
``src/nablamimic/evaluate.py``     the eval loop, ``actor_ema`` selection
``configs/ours.args``              ``--residual-action``, 10-frame history
=================================  ============================================

The 93-dim proprioception (``env.proprio``), concatenated in this order::

    0:29    q - q0                     joint angles relative to the keyframe
    29:58   qd * 0.2                   joint velocities
    58:61   w * 0.5                    pelvis angular velocity, body frame
    61:64   R^T g                      projected gravity, pelvis frame
    64:93   a_prev                     raw policy output of the previous step

The 686-dim goal (``env.goal``) is 10 reference frames ``f .. f+9`` (clamped to
the clip end), each 68 wide, plus the 6-d anchor orientation::

     0:29   q[k] - q0                  reference joint angles
     29:58  qd[k] * 0.2                reference joint velocities
     58:61  R_k^T g                    reference projected gravity
     61:64  w_k * 0.5                  reference pelvis angular velocity
     64:67  H_yaw^T v_k                reference linear velocity, yaw frame
     67:68  z_k                        reference pelvis height
    674:680 the above for the 10 frames, flattened
    680:686 anchor orientation        R_robot_torso^T R_ref_torso, first 2 columns

Note that the goal starts at the *current* frame ``f`` (not ``f+1`` as
SuperMimic does) and that the encoder does not see the anchor: with
``--anchor-to-decoder`` the last 6 goal entries bypass the encoder and enter the
decoder beside the history (``nets.actor_apply``).

Control semantics
-----------------
NablaMimic was trained with ``--residual-action``: the commanded setpoint is
``ref[min(f+1, last)] + action_scale * a``, i.e. the *next* reference frame's
joint angles plus the scaled output, then clipped to the setpoint range. The
benchmark's contract is ``default_joint_pos + action_scale * a``. This module
therefore returns

    action_benchmark = a_nablamimic + (ref[k] - q0) / action_scale

with ``k = reference_index`` (which is ``f + 1``), so that both stacks command
the identical setpoint. Nothing else about the action is changed: no clipping,
scaling, smoothing or filtering.

Why the observation has to be reconstructed
-------------------------------------------
The evaluator hands out physical quantities, not NablaMimic's tensors. Three of
them are read directly (joint positions, joint velocities, IMU angular
velocity), one is exactly the previous action, and the rest are rebuilt
algebraically:

* the torso-relative orientation ``M = R_torso^T R_ref_torso`` from
  ``anchor_ori_b_prev`` (frame ``f``, see "Frame indexing");
* the *pelvis* frame quantities, because ``R_torso = R_pelvis @ A(q_waist)``
  with ``A = Rz(yaw) Rx(roll) Ry(pitch)`` (``assets/g1/g1.xml``: pelvis ->
  ``waist_yaw_link`` -> ``waist_roll_link`` -> ``torso_link``, all body offsets
  rotation-free), hence ``R_pelvis^T R_ref_pelvis = A(q) M A(q_ref)^T`` and
  ``gravity_b = A(q) M A(q_ref)^T g_ref_pelvis``;
* the reference root linear / angular velocity, which ``ReferenceClip`` does not
  expose, by the same central differences ``mj_differentiatePos`` uses
  (``_root_velocities`` below).

``tests/test_nablamimic_policy.py`` pins the result against values produced by
the original implementation; ``docs/NABLAMIMIC_POLICY.md`` records the measured
residuals. None of this is a re-encoding of the policy -- it is the arithmetic
the original performed inside its own simulator, redone on the public state.

Frame indexing
--------------
mjlab's ``MotionCommand`` writes the reference state of frame 0 at reset and
then advances ``time_steps`` to 1 (``commands.py::_update_command``), so the
first ``StepInput`` carries ``reference_index == 1`` while the robot stands at
frame 0. NablaMimic's own ``frame`` is therefore ``reference_index - 1``: that
is the frame its goal starts at, the frame its anchor is expressed in, and
``reference_index`` itself is the frame its residual base uses.

Two different frames are live at each step and the adapter must not mix them:

===========================  ==============================================
what                         frame
===========================  ==============================================
robot state / ``proprio``    ``f = reference_index - 1``
goal features ``f .. f+9``   ``f``
anchor orientation           ``f``        (``anchor_ori_b_prev``)
residual base                ``f + 1``    (``reference_index``)
reward / termination         ``f + 1``    (mjlab's, not the policy's)
===========================  ==============================================

mjlab's own observation terms read the anchor at ``reference_index`` (the
upcoming frame), which is right for mjlab's convention but one frame ahead of
NablaMimic's. :class:`StepInput` therefore carries the frame-``f`` anchor as
``anchor_*_prev``; this adapter uses those. Earlier revisions used
``anchor_ori_b`` and documented the resulting mixed-frame observation as "the
protocol's +1 offset"; that was wrong -- it is an adapter bug, and
``docs/NABLAMIMIC_EVAL_AUDIT.md`` records how it was found and what it costs.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from benchmark.evaluator import constants as C
from benchmark.evaluator import mathx

NUM_J = 29
HIST = 10
FUTURE = 10
PROPRIO_DIM = 93
GOAL_FRAME_DIM = 68
ANCHOR_DIM = 6
GOAL_DIM = FUTURE * GOAL_FRAME_DIM + ANCHOR_DIM
JOINT_VEL_SCALE = 1.0 / 5.0
ANG_VEL_SCALE = 1.0 / 2.0
CONTROL_DT = C.STEP_DT

#: ``src/nablamimic/env.py`` ``Q0``: the mjlab knees-bent keyframe, unlisted 0.
Q0_PATTERNS = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": 0.6,
    "left_shoulder_roll": 0.2,
    "right_shoulder_roll": -0.2,
    "shoulder_pitch": 0.2,
}

#: ``assets/g1/g1.xml``: the waist chain pelvis -> torso, in order.
_WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_WAIST_IDX = tuple(C.JOINT_ORDER.index(n) for n in _WAIST)
_WAIST_AXES = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

#: ``lib.qvel[3:6]`` of a MuJoCo free joint is the angular velocity in the *body*
#: frame; ``"world"`` exists only so the verification harness can show that the
#: body-frame reading is the one that reproduces the original reference data.
ROOT_ANG_VEL_FRAME = "body"

CHECKPOINT_ACTOR_KEY = "actor_ema"  # ``--actor-ema 0.999``: evaluate.py prefers it


class NablaMimicContractError(RuntimeError):
    """The evaluator's control contract differs from what NablaMimic was trained on."""


def _anchor_ori_at_state_frame(step) -> np.ndarray:
    """The 6-D anchor orientation at the frame the robot's state belongs to.

    ``StepInput.anchor_ori_b`` is the anchor at ``reference_index`` -- the frame
    the *upcoming* reward compares against, which is ``f + 1``. NablaMimic's
    observation describes the state at frame ``f`` and anchors there
    (``env.goal`` calls ``_anchor_rel(state.dx, state.frame)``, and
    ``env.proprio`` rebuilds gravity from the same frame), so the adapter must
    read ``anchor_ori_b_prev``.

    Mixing the two is not a cosmetic slip: the gravity channel composes
    ``A(q) M A(q_ref)^T g_ref``, so a wrong-frame ``M`` perturbs
    ``proprio[61:64]`` by the reference torso's one-frame rotation, *and*
    the last six goal entries condition the decoder on the wrong orientation.
    See ``docs/NABLAMIMIC_EVAL_AUDIT.md``.

    Fails closed rather than silently falling back to the mismatched field.
    """
    value = getattr(step, "anchor_ori_b_prev", None)
    if value is None:
        raise NablaMimicContractError(
            "StepInput.anchor_ori_b_prev is missing: this evaluator cannot supply the "
            "anchor at the robot's own reference frame, and falling back to "
            "anchor_ori_b (frame reference_index) would feed the actor a mixed-frame "
            "observation. Update benchmark/evaluator/policy_api.py."
        )
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# NablaMimic's own math, reimplemented (numpy) to keep the weights untouched.
# ---------------------------------------------------------------------------


def default_joint_pos_reference() -> np.ndarray:
    """``env.Q0`` resolved onto ``constants.JOINT_ORDER`` (substring match)."""
    out = np.zeros(NUM_J)
    for i, name in enumerate(C.JOINT_ORDER):
        for key, value in Q0_PATTERNS.items():
            if key in name:
                out[i] = value
    return out


def action_scale_reference() -> np.ndarray:
    """``env.action_scale`` = ``0.25 * effort_limit / stiffness`` per actuator.

    Recomputed from the frozen actuator table so the two stacks are compared
    rather than assumed equal; ``stiffness = armature * (2 pi 10 Hz)^2`` is
    exactly ``constants.ACTUATOR_GROUPS[i].stiffness``.
    """
    out = np.zeros(NUM_J)
    for i, name in enumerate(C.JOINT_ORDER):
        for group in C.ACTUATOR_GROUPS:
            if any(re.fullmatch(p, name) for p in group.patterns):
                out[i] = C.ACTION_SCALE * group.effort_limit / group.stiffness
                break
    return out


def axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues rotation about a fixed unit axis."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ],
        dtype=np.float64,
    )


def waist_rotation(q: np.ndarray) -> np.ndarray:
    """``A(q)`` with ``R_torso = R_pelvis @ A(q)`` = ``Rz(yaw) Rx(roll) Ry(pitch)``."""
    yaw, roll, pitch = (float(q[i]) for i in _WAIST_IDX)
    return (
        axis_rotation(_WAIST_AXES[0], yaw)
        @ axis_rotation(_WAIST_AXES[1], roll)
        @ axis_rotation(_WAIST_AXES[2], pitch)
    )


def six_to_mat(v6: np.ndarray) -> np.ndarray:
    """Inverse of mjlab's ``matrix_from_quat(ori)[..., :2].reshape(-1)``.

    ``motion_anchor_ori_b`` slices the first two *columns* of the rotation
    matrix and then flattens **row-major**, so the payload is interleaved:
    ``v6 = [m00, m01, m10, m11, m20, m21]``. Measured, not assumed: reading it
    this way gives ``|det - 1| <= 3e-7`` on every step of a walking clip,
    whereas the column-stacked reading gives ``|det - 1| ~ 0.75`` (it is not a
    rotation at all).
    """
    v6 = np.asarray(v6, dtype=np.float64)
    c0, c1 = v6[0::2], v6[1::2]
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def anchor_ori_stacked(v6: np.ndarray) -> np.ndarray:
    """Re-order the evaluator's 6-D orientation into NablaMimic's convention.

    ``env._anchor_rel_pq`` emits ``concatenate([rot[:, 0], rot[:, 1]])`` -- the
    two **columns** of the relative rotation, stacked. The evaluator hands out
    the interleaved layout of :func:`six_to_mat`, so feeding it through
    unchanged would transpose the rotation the network is conditioned on. The
    mapping is a permutation, not a change of values.
    """
    return np.asarray(v6, dtype=np.float64)[[0, 2, 4, 1, 3, 5]]


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences, one-sided at both ends (``mj_differentiatePos`` rule)."""
    g = np.empty_like(x, dtype=np.float64)
    g[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion, stable near zero."""
    q = np.asarray(q, dtype=np.float64)
    q = q if q[0] >= 0.0 else -q
    v = q[1:]
    sin_half = float(np.linalg.norm(v))
    if sin_half < 1e-9:
        return 2.0 * v / max(q[0], 1e-12)
    return v * (2.0 * np.arctan2(sin_half, q[0]) / sin_half)


def _root_velocities(
    root_pos: np.ndarray, root_quat: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference pelvis linear (world) and angular (body) velocity per frame.

    ``ReferenceClip`` exposes the reference root pose but not its velocity.
    ``nablamimic/motion.py`` gets them from ``mujoco.mj_differentiatePos`` over
    the same 50 Hz frames, which is a central difference: the linear part is
    reproduced exactly by differencing the stored positions, the angular part by
    the SO(3) derivative of the quaternions. See ``docs/NABLAMIMIC_POLICY.md``
    for the measured residual against the original arrays.
    """
    p = np.asarray(root_pos, dtype=np.float64)
    q = np.asarray(root_quat, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)

    lin_w = _gradient(p, dt)
    ang = np.empty_like(p)
    for i in range(len(q)):
        a, b = max(i - 1, 0), min(i + 1, len(q) - 1)
        # Body-frame increment: q_a^{-1} * q_b, scipy-free ordering.
        if ROOT_ANG_VEL_FRAME == "body":
            dq = mathx.quat_mul(mathx.quat_inv(q[a]), q[b])
        else:
            dq = mathx.quat_mul(q[b], mathx.quat_inv(q[a]))
        ang[i] = _quat_rotvec(dq) / ((b - a) * dt)
    return lin_w, ang


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """``mlp_apply``'s layer: JAX stores ``(n_in, n_out)``, so ``x @ W + b``."""
    return x @ weight + bias


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def _mlp(params: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    """``nets.mlp_apply``: ELU between layers, linear on the last one."""
    for w, b in params[:-1]:
        x = _elu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def load_actor_ema(path: str | Path) -> dict:
    """Read ``actor_ema`` out of a ``nablamimic.train`` checkpoint.

    The checkpoints are ``pickle`` files of JAX PyTrees, so reading one requires
    ``jax`` (the algorithm's own checkpoint format; the evaluator only hashes
    the file). ``scripts/export_nablamimic_actor.py`` can turn one into a plain
    ``.npz`` for tests that must not depend on JAX.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            params = {
                "enc": [(z[f"enc.{i}.w"], z[f"enc.{i}.b"]) for i in range(int(z["enc_layers"]))],
                "dec": [(z[f"dec.{i}.w"], z[f"dec.{i}.b"]) for i in range(int(z["dec_layers"]))],
            }
            if "anchor_dec" in z:
                params["anchor_dec"] = z["anchor_dec"]
            return params
    try:
        import jax  # noqa: PLC0415  (only needed to deserialize the PyTree)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NablaMimicContractError(
            "loading a NablaMimic .pkl checkpoint needs `jax` (it stores JAX PyTrees); "
            "install jax or pass the .npz export from scripts/export_nablamimic_actor.py"
        ) from exc
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)  # noqa: S301  (own artifact)
    if CHECKPOINT_ACTOR_KEY not in checkpoint:
        raise NablaMimicContractError(
            f"{path} has no {CHECKPOINT_ACTOR_KEY!r} entry (keys: {sorted(checkpoint)[:8]})"
        )
    return jax.tree_util.tree_map(np.asarray, checkpoint[CHECKPOINT_ACTOR_KEY])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class NablaMimicPolicy:
    """``reset(context)`` / ``act(step)`` around the original ``actor_ema``.

    The network runs in numpy float32 (the weights' own dtype); the observation
    and goal are built in float64 and cast once before the forward pass, which
    is what the original does inside JAX (``jax_default_matmul_precision =
    "highest"`` at evaluation time, i.e. fp32 matmuls).
    """

    kind = "nablamimic-native-actor-ema"

    def __init__(self, params: dict):
        if "anchor_dec" not in params:
            raise NablaMimicContractError(
                "this checkpoint was not trained with --anchor-to-decoder; the "
                "evaluated 8-minute recipe (configs/ours.args) sets it, so the "
                "weights do not match the observation layout used here"
            )
        self.enc = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in params["enc"]]
        self.dec = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in params["dec"]]
        self._check_shapes()

        self._context = None
        self._feat: np.ndarray | None = None
        self._hist: np.ndarray | None = None
        self._prev_action_raw = np.zeros(NUM_J)
        self._last_action = None

    # ---------------------------------------------------------------- setup
    def _check_shapes(self) -> None:
        enc_in, latent = self.enc[0][0].shape[0], self.enc[-1][0].shape[1]
        dec_in, act_dim = self.dec[0][0].shape[0], self.dec[-1][0].shape[1]
        hist_in = dec_in - latent - ANCHOR_DIM
        if act_dim != NUM_J:
            raise NablaMimicContractError(f"actor outputs {act_dim} actions, expected {NUM_J}")
        if enc_in != GOAL_DIM - ANCHOR_DIM:
            raise NablaMimicContractError(
                f"encoder input {enc_in} != {FUTURE}*{GOAL_FRAME_DIM} = {GOAL_DIM - ANCHOR_DIM}"
            )
        if hist_in != HIST * PROPRIO_DIM:
            raise NablaMimicContractError(
                f"decoder expects {hist_in} history values, not {HIST}*{PROPRIO_DIM}"
            )
        self.latent = int(latent)

    def reset(self, context) -> None:
        """Check the control contract and precompute the clip's goal features."""
        if tuple(context.joint_order) != C.JOINT_ORDER:
            raise NablaMimicContractError(
                "evaluator joint order differs from the G1 actuator order NablaMimic trained on"
            )
        scale_err = float(np.abs(np.asarray(context.action_scale) - action_scale_reference()).max())
        if scale_err > 1e-8:
            raise NablaMimicContractError(
                f"action_scale differs from NablaMimic's 0.25*effort/stiffness by {scale_err:.3e}"
            )
        pose_err = float(
            np.abs(np.asarray(context.default_joint_pos) - default_joint_pos_reference()).max()
        )
        if pose_err > 1e-12:
            raise NablaMimicContractError(
                f"default_joint_pos differs from NablaMimic's q0 by {pose_err:.3e}"
            )

        self._context = context
        self._feat = self._goal_features(context.reference)
        self._hist = None
        self._prev_action_raw = np.zeros(NUM_J)
        self._last_action = None

    def _goal_features(self, ref) -> np.ndarray:
        """``env._goal_features``: 68 features per reference frame."""
        q0 = default_joint_pos_reference()
        joint_pos = np.asarray(ref.joint_pos, dtype=np.float64)
        joint_vel = np.asarray(ref.joint_vel, dtype=np.float64)
        root_pos = np.asarray(ref.root_pos_w, dtype=np.float64)
        root_quat = np.asarray(ref.root_quat_w, dtype=np.float64)

        grav = np.stack([mathx.projected_gravity(q) for q in root_quat])
        lin_w, ang = _root_velocities(root_pos, root_quat, CONTROL_DT)
        # ``H_yaw^T v``: the reference linear velocity in the heading frame.
        yaw = np.arctan2(
            2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
            1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
        )
        c, s = np.cos(yaw), np.sin(yaw)
        lin_h = np.stack(
            [c * lin_w[:, 0] + s * lin_w[:, 1], -s * lin_w[:, 0] + c * lin_w[:, 1], lin_w[:, 2]],
            axis=1,
        )
        feats = np.concatenate(
            [
                joint_pos - q0,
                joint_vel * JOINT_VEL_SCALE,
                grav,
                ang * ANG_VEL_SCALE,
                lin_h,
                root_pos[:, 2:3],
            ],
            axis=1,
        )
        if feats.shape[1] != GOAL_FRAME_DIM:
            raise NablaMimicContractError(f"goal frame width {feats.shape[1]} != {GOAL_FRAME_DIM}")
        return feats

    # ------------------------------------------------------------------ act
    def proprioception(self, step, frame: int) -> np.ndarray:
        """``env.proprio`` for one control step, from the evaluator's ``StepInput``."""
        q = np.asarray(step.joint_pos, dtype=np.float64)
        q_ref = np.asarray(self._context.reference.joint_pos[frame], dtype=np.float64)
        torso_rel = six_to_mat(_anchor_ori_at_state_frame(step))  # R_robot_torso^T R_ref_torso(f)
        g_ref = mathx.projected_gravity(self._context.reference.root_quat_w[frame])

        # pelvis frame: R_pelvis^T R_ref_pelvis applied to the reference gravity.
        gravity_b = waist_rotation(q) @ torso_rel @ waist_rotation(q_ref).T @ g_ref

        return np.concatenate(
            [
                q - np.asarray(self._context.default_joint_pos, dtype=np.float64),
                np.asarray(step.joint_vel, dtype=np.float64) * JOINT_VEL_SCALE,
                np.asarray(step.base_ang_vel, dtype=np.float64) * ANG_VEL_SCALE,
                gravity_b,
                self._prev_action_raw,
            ]
        )

    def goal(self, frame: int, step) -> np.ndarray:
        """``env.goal``: 10 reference frames from ``frame`` on, plus the anchor part."""
        last = self._feat.shape[0] - 1
        idx = np.minimum(frame + np.arange(FUTURE), last)
        return np.concatenate(
            [self._feat[idx].reshape(-1), anchor_ori_stacked(_anchor_ori_at_state_frame(step))]
        )

    def forward(self, hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """``nets.actor_apply`` with ``--anchor-to-decoder``, deterministic."""
        g = np.asarray(goal, dtype=np.float32)
        z = np.tanh(_mlp(self.enc, g[:-ANCHOR_DIM]))
        x = np.concatenate([z, np.asarray(hist, np.float32).reshape(-1), g[-ANCHOR_DIM:]])
        return _mlp(self.dec, x)

    def act(self, step) -> np.ndarray:
        k = int(step.reference_index)
        frame = k - 1  # NablaMimic's frame; see "Frame indexing" in the module docstring
        if frame < 0:
            raise NablaMimicContractError(
                f"reference_index {k} does not leave a frame to anchor at"
            )

        obs = self.proprioception(step, frame)
        if self._hist is None:
            # ``evaluate.py``: the first observation is repeated ``hist`` times.
            self._hist = np.repeat(obs[None, :], HIST, axis=0)
        else:
            self._hist = np.concatenate([self._hist[1:], obs[None, :]], axis=0)

        action_raw = self.forward(self._hist.reshape(-1), self.goal(frame, step))
        if not np.all(np.isfinite(action_raw)):
            raise NablaMimicContractError(
                f"non-finite action at step {k} of {self._context.clip}"
            )

        # Residual action: both stacks must command the same setpoint.
        reference = self._context.reference
        base = np.asarray(reference.joint_pos[min(k, reference.num_frames - 1)], dtype=np.float64)
        scale = np.asarray(self._context.action_scale, dtype=np.float64)
        q0 = np.asarray(self._context.default_joint_pos, dtype=np.float64)
        action = action_raw.astype(np.float64) + (base - q0) / scale

        if self._last_action is not None:
            drift = float(np.abs(np.asarray(step.previous_action) - self._last_action).max())
            if drift > 1e-9:
                raise NablaMimicContractError(
                    f"action/history desync at step {k} of {self._context.clip}: "
                    f"evaluator's previous_action differs by {drift:.3e}"
                )
        self._last_action = action.copy()
        self._prev_action_raw = action_raw.astype(np.float64)
        return action

    # ------------------------------------------------------------ reporting
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in self.enc + self.dec))

    def architecture(self) -> dict[str, str]:
        def widths(p):
            return [int(p[0][0].shape[0])] + [int(w.shape[1]) for w, _ in p]

        return {
            "encoder": "Linear+ELU chain " + "->".join(map(str, widths(self.enc))) + ", tanh",
            "decoder": "Linear+ELU chain " + "->".join(map(str, widths(self.dec))),
            "activation": "ELU (alpha=1)",
            "observation": f"({HIST}, {PROPRIO_DIM}) flattened",
            "goal": f"{FUTURE} x {GOAL_FRAME_DIM} + {ANCHOR_DIM} anchor (anchor bypasses the encoder)",
            "action": "residual around ref[k] (no clipping, scaling or smoothing)",
        }


def make_policy(checkpoint: str | Path | None = None) -> NablaMimicPolicy:
    """Factory for ``--policy-file benchmark/policies/nablamimic.py:make_policy``."""
    if checkpoint is None:
        raise SystemExit(
            "NablaMimicPolicy needs a checkpoint: pass --checkpoint <run>/ckpt/it_001250.pkl"
        )
    return NablaMimicPolicy(load_actor_ema(checkpoint))
