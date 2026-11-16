"""SuperMimic's *native* inference, driven through the canonical evaluator.

This is not a re-encoding of SuperMimic into mjlab's 160-dim observation. It
rebuilds, from the public robot state and the reference motion, the exact
tensors the original implementation fed to its network, and runs the original
``pi.enc`` / ``pi.dec`` weights unchanged.

Source of truth
---------------
``zita-ch/supermimic_vibed`` @ ``8bfc1158bd968f729fcc65d5e897a05b05ea3d36``
(the commit the evaluated checkpoint was trained at; the checkpoint records it
as ``args``/``saved_at`` 2026-11-03 06:13:58, iteration 15000):

============================  ============================================
``supermimic/mdp.py``         ``OBS_DIM = 99``, ``MDP.obs``
``supermimic/motion.py``      ``MotionLib.goal`` (730 = 10 x 73), ``_gradient``
``supermimic/networks.py``    ``Policy`` = ``enc`` / ``dec``, ``RunningNorm``
``supermimic/evaluate.py``    the eval loop (history + goal + action)
``supermimic/env.py``         ``target = default_q + dq0 + alpha * a``
``supermimic/robot.py``       joint order, default pose, ``action_scale``
============================  ============================================

The 99-dim observation (``mdp.obs``), concatenated in this order::

    0:29   q - default_q - dq0        (dq0 == 0 whenever domain randomisation is off)
    29:58  qd
    58:61  R^T w                      pelvis angular velocity, body frame
    61:64  -R[2, :]                   projected gravity, body frame
    64:93  a_prev                     raw action of the previous step
    93:99  mat_to_6d(R^T R_ref[f])    reference pelvis orientation, robot frame

The 730-dim goal (``MotionLib.goal``) is 10 future reference frames
``f+1 .. f+10`` (clamped to the clip end), each 73 wide::

     0:3   H^T (p[k] - p[f])          displacement in the current yaw frame
     3:9   mat_to_6d(H^T R[k])
     9:12  reference pelvis linear velocity, root frame
    12:15  reference pelvis angular velocity, root frame
    15:44  reference joint angles q[k]        (absolute, not offsets)
    44:73  reference joint velocities qd[k] * 0.1

``mat_to_6d(M)`` is ``M[:, :2].reshape(-1)`` in both implementations, i.e.
``[M00, M01, M10, M11, M20, M21]``.

Inference needs no recurrent, world-model or critic state: ``evaluate.py``
calls ``pi(hist, goal)`` and nothing else, and ``supermimic/mdp.py``'s world
model is used only for imagined rollouts during training.

Why the observation has to be reconstructed
-------------------------------------------
The evaluator does not hand out ``R^T w`` or ``R``; it hands out physical
quantities. Three of them are exactly what SuperMimic reads (joint positions,
joint velocities, IMU angular velocity), one is exactly its previous action,
and the remaining two are rebuilt algebraically:

* the relative pelvis orientation ``R^T R_ref(f)`` from the exposed *torso*
  relative orientation ``M = R_torso^T R_ref_torso(f)`` (``anchor_ori_b_prev``)
  and the three waist angles, because ``R_torso = R_pelvis @ A(q_waist)`` with
  ``A = Rz(yaw) Rx(roll) Ry(pitch)`` (``assets/g1/g1.xml``: pelvis ->
  ``waist_yaw_link`` -> ``waist_roll_link`` -> ``torso_link``, all body offsets
  rotation-free), so ``R^T R_ref = A(q) M A(q_ref)^T``;
* projected gravity ``A(q) M A(q_ref)^T g_ref(f)``, with ``g_ref(f)`` the
  reference pelvis projected gravity from ``ReferenceClip.root_quat_w[f]``.

Frame indexing
--------------
mjlab's ``MotionCommand`` writes the reference state of frame 0 at reset and
then advances ``time_steps`` to 1 (``commands.py::_update_command``), so the
first ``StepInput`` carries ``reference_index == 1`` while the robot stands at
frame 0. SuperMimic's own ``frame`` (``env.f`` in ``supermimic/evaluate.py``)
is therefore ``f = reference_index - 1``: ``MDP.obs`` reads ``root_R[f]`` and
``MotionLib.goal`` computes future frames ``f + 1 .. f + 10`` relative to
frame ``f``. Earlier revisions read ``reference_index`` and ``anchor_ori_b``
(frame ``f + 1``), shifting the entire observation and goal window ahead by one
frame; this module reads ``frame = reference_index - 1`` and
``anchor_ori_b_prev``.

``tests/test_supermimic_policy.py`` checks both against the simulator's own
``MjData`` every step of a real rollout; the residual is float32 reference
precision (~1e-7), not an approximation.

Control semantics
-----------------
The action SuperMimic emits is used *unchanged*: both stacks command
``target = default_joint_pos + action_scale * action`` with the same joint
order, the same default pose and the same ``0.25 * effort_limit / stiffness``
scale (verified at ``reset``), so no conversion belongs here. The differences
that do remain are physical (simulator, contact model, friction) and are
recorded in ``docs/SUPERMIMIC_POLICY.md`` rather than patched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.evaluator import mathx
from benchmark.evaluator.torchzip import load_checkpoint

OBS_DIM = 99
NUM_J = 29
Z_DIM = 32
GOAL_FRAMES = 10
HIST = 10
GOAL_PER_FRAME = 3 + 6 + 3 + 3 + NUM_J + NUM_J  # 73
CONTROL_DT = 0.02

#: ``supermimic/robot.py`` ``G1_JOINT_NAMES`` (== the LAFAN1 CSV column order).
JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: ``supermimic/robot.py`` ``DEFAULT_JOINT_POS`` (regex -> radians), unlisted 0.
_DEFAULT_POS = {
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

#: ``supermimic/robot.py`` joint motors: (effort_limit Nm, armature kg m^2).
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_MOTORS = {
    "hip_pitch": (88.0, _ARMATURE_7520_14),
    "hip_roll": (139.0, _ARMATURE_7520_22),
    "hip_yaw": (88.0, _ARMATURE_7520_14),
    "knee": (139.0, _ARMATURE_7520_22),
    "ankle_pitch": (50.0, 2.0 * _ARMATURE_5020),
    "ankle_roll": (50.0, 2.0 * _ARMATURE_5020),
    "waist_yaw": (88.0, _ARMATURE_7520_14),
    "waist_roll": (50.0, 2.0 * _ARMATURE_5020),
    "waist_pitch": (50.0, 2.0 * _ARMATURE_5020),
    "shoulder_pitch": (25.0, _ARMATURE_5020),
    "shoulder_roll": (25.0, _ARMATURE_5020),
    "shoulder_yaw": (25.0, _ARMATURE_5020),
    "elbow": (25.0, _ARMATURE_5020),
    "wrist_roll": (25.0, _ARMATURE_5020),
    "wrist_pitch": (5.0, _ARMATURE_4010),
    "wrist_yaw": (5.0, _ARMATURE_4010),
}
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535

_WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_WAIST_IDX = tuple(JOINT_ORDER.index(n) for n in _WAIST)
#: ``g1.xml`` joint axes, in chain order pelvis -> torso.
_WAIST_AXES = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


class SuperMimicContractError(RuntimeError):
    """The evaluator's control contract differs from what SuperMimic was trained on."""


def _anchor_ori_at_state_frame(step) -> np.ndarray:
    """The 6-D anchor orientation at the frame the robot's state belongs to.

    ``StepInput.anchor_ori_b`` is the anchor at ``reference_index`` (``f + 1``),
    whereas SuperMimic's ``env.f`` is ``f = reference_index - 1``. Fails closed
    if ``anchor_ori_b_prev`` is missing rather than falling back to the
    shifted-frame field.
    """
    value = getattr(step, "anchor_ori_b_prev", None)
    if value is None:
        raise SuperMimicContractError(
            "StepInput.anchor_ori_b_prev is missing: cannot supply the anchor at "
            "frame reference_index - 1 without falling back to anchor_ori_b (f + 1)."
        )
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# SuperMimic's own math, reimplemented (numpy) to keep the weights untouched.
# ---------------------------------------------------------------------------


def mat_to_6d(R: np.ndarray) -> np.ndarray:
    """``supermimic.rotations.mat_to_6d``: ``R[..., :, :2].reshape(-1)``."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def six_to_mat(v6: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mat_to_6d`; the third column is the cross product."""
    v6 = np.asarray(v6, dtype=np.float64)
    c0, c1 = v6[0::2], v6[1::2]
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues rotation about a fixed unit axis (``supermimic.rotations.axis_angle_to_mat``)."""
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
    """``A(q)`` with ``R_torso = R_pelvis @ A(q)``.

    ``g1.xml`` puts ``waist_yaw`` (axis z), ``waist_roll`` (axis x) and
    ``waist_pitch`` (axis y) between the pelvis and ``torso_link`` with no
    rotational body offsets, so ``A = Rz(yaw) Rx(roll) Ry(pitch)``.
    """
    yaw, roll, pitch = (float(q[i]) for i in _WAIST_IDX)
    return axis_rotation(_WAIST_AXES[0], yaw) @ axis_rotation(_WAIST_AXES[1], roll) @ (
        axis_rotation(_WAIST_AXES[2], pitch)
    )


def yaw_matrix(R: np.ndarray) -> np.ndarray:
    """``supermimic.rotations.yaw_mat``: heading-only rotation from the x axis."""
    fx, fy = R[0, 0], R[1, 0]
    n = np.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R`` (``supermimic.rotations.so3_log``, quaternion route)."""
    q = _quat_from_matrix(R)
    return _quat_to_rotvec(q)


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """``supermimic.rotations.mat_to_quat``: wxyz, w >= 0, branch by largest term."""
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    qw = np.sqrt(max(1.0 + m00 + m11 + m22, 1e-10)) / 2.0
    qx = np.sqrt(max(1.0 + m00 - m11 - m22, 1e-10)) / 2.0
    qy = np.sqrt(max(1.0 - m00 + m11 - m22, 1e-10)) / 2.0
    qz = np.sqrt(max(1.0 - m00 - m11 + m22, 1e-10)) / 2.0
    cands = np.array(
        [
            [qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)],
            [(m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)],
            [(m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)],
            [(m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz],
        ]
    )
    q = cands[int(np.argmax([qw, qx, qy, qz]))]
    q = q / max(np.linalg.norm(q), 1e-9)
    return q if q[0] >= 0 else -q


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """``supermimic.rotations.quat_to_rotvec``."""
    q = q if q[0] >= 0 else -q
    v = q[1:]
    sin_half = float(np.linalg.norm(v))
    half = float(np.arctan2(sin_half, q[0]))
    if sin_half < 1e-6:
        scale = 2.0 / max(q[0], 1e-6)
    else:
        scale = 2.0 * half / (sin_half + 1e-12)
    return v * scale


def gradient(x: np.ndarray, dt: float) -> np.ndarray:
    """``supermimic.motion._gradient``: central differences, one-sided at both ends."""
    g = np.empty_like(x, dtype=np.float64)
    g[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def reference_derivatives(
    joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, dt: float = CONTROL_DT
) -> dict[str, np.ndarray]:
    """The per-frame reference quantities ``MotionLib`` derives at construction time.

    Same formulas as ``supermimic/motion.py``: joint and root linear velocity by
    ``_gradient``, root angular velocity from the SO(3) derivative of every
    second orientation, then both rotated into the reference root frame.
    Returned in float64; the original ran this in float32 on the CSV values.
    """
    p = np.asarray(root_pos, dtype=np.float64)
    q = np.asarray(joint_pos, dtype=np.float64)
    R = np.stack([mathx.matrix_from_quat(x) for x in np.asarray(root_quat, dtype=np.float64)])

    root_v = gradient(p, dt)
    joint_v = gradient(q, dt)
    root_w = np.empty_like(p)
    root_w[1:-1] = np.stack(
        [so3_log(R[i + 1] @ R[i - 1].T) for i in range(1, len(R) - 1)]
    ) / (2.0 * dt)
    root_w[0] = so3_log(R[1] @ R[0].T) / dt
    root_w[-1] = so3_log(R[-1] @ R[-2].T) / dt

    Rt = np.transpose(R, (0, 2, 1))
    return {
        "R": R,
        "p": p,
        "q": q,
        "qd": joint_v,
        "root_v_local": np.einsum("tij,tj->ti", Rt, root_v),
        "root_w_local": np.einsum("tij,tj->ti", Rt, root_w),
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def action_scale_reference() -> np.ndarray:
    """``supermimic/robot.py`` ``action_scale`` for every joint, in joint order."""
    out = np.zeros(NUM_J)
    for i, name in enumerate(JOINT_ORDER):
        key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
        effort, armature = _MOTORS[key]
        out[i] = 0.25 * effort / (armature * _NATURAL_FREQ**2)
    return out


def default_joint_pos_reference() -> np.ndarray:
    """``supermimic/robot.py`` ``default_joint_pos_list``."""
    import re

    out = np.zeros(NUM_J)
    for i, name in enumerate(JOINT_ORDER):
        for pattern, value in _DEFAULT_POS.items():
            if re.fullmatch(pattern, name):
                out[i] = value
    return out


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class SuperMimicPolicy:
    """``reset(context)`` / ``act(step)`` around the original ``pi.enc`` / ``pi.dec``.

    The network runs in numpy float32 (the weights' own dtype); the observation
    and goal are built in float64 and cast once before the forward pass. The
    original evaluation ran the same matmuls under ``torch.autocast("cuda",
    bf16)``, which a CPU cannot reproduce: against an fp32 torch reference the
    action agrees to 5e-6, against a bf16-weight emulation of that GPU path to
    4e-2 max / 5e-3 mean. Numbers and closed-loop checks are in
    ``docs/SUPERMIMIC_POLICY.md`` §3.3.
    """

    kind = "supermimic-native-pi"

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors
        self.hist, self.goal_frames = self._check_shapes(tensors)

        self.dec_w = [tensors[f"pi.dec.{i}.weight"] for i in (0, 2, 4, 6, 8)]
        self.dec_b = [tensors[f"pi.dec.{i}.bias"] for i in (0, 2, 4, 6, 8)]
        self.enc_w = [tensors[f"pi.enc.{i}.weight"] for i in (0, 2, 4, 6, 8)]
        self.enc_b = [tensors[f"pi.enc.{i}.bias"] for i in (0, 2, 4, 6, 8)]
        self.obs_mean = tensors["pi.obs_norm.mean"]
        self.obs_std = np.sqrt(tensors["pi.obs_norm.var"] + 1e-2**2)
        self.goal_mean = tensors["pi.goal_norm.mean"]
        self.goal_std = np.sqrt(tensors["pi.goal_norm.var"] + 1e-2**2)

        self._context = None
        self._derived: dict[str, np.ndarray] | None = None
        self._hist: np.ndarray | None = None

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _check_shapes(t: dict[str, np.ndarray]) -> tuple[int, int]:
        """Derive (hist, goal_frames) from the weights and pin them to the paper's values."""
        enc_in = t["pi.enc.0.weight"].shape[1]
        dec_in = t["pi.dec.0.weight"].shape[1]
        z_dim, act_dim = t["pi.enc.8.weight"].shape[0], t["pi.dec.8.weight"].shape[0]
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
            raise SuperMimicContractError(f"decoder width {dec_in} is not 32 + k * {OBS_DIM}")
        if t["pi.obs_norm.mean"].shape != (OBS_DIM,) or t["pi.goal_norm.mean"].shape != (
            enc_in,
        ):
            raise SuperMimicContractError("normaliser shapes do not match the network widths")
        if (goal_frames, hist) != (GOAL_FRAMES, HIST):
            raise SuperMimicContractError(
                f"checkpoint encodes goal_frames={goal_frames} hist={hist}; "
                f"this implementation is the {GOAL_FRAMES}/{HIST} configuration"
            )
        return hist, goal_frames

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SuperMimicPolicy":
        """Load ``pi.*`` out of a ``torch.save`` file without importing torch."""
        tensors = load_checkpoint(Path(path))
        missing = [
            f"pi.{head}.{i}.{kind}"
            for head in ("enc", "dec")
            for i in (0, 2, 4, 6, 8)
            for kind in ("weight", "bias")
            if f"pi.{head}.{i}.{kind}" not in tensors
        ]
        missing += [
            f"pi.{name}.{stat}"
            for name in ("obs_norm", "goal_norm")
            for stat in ("mean", "var")
            if f"pi.{name}.{stat}" not in tensors
        ]
        if missing:
            raise SuperMimicContractError(f"checkpoint is missing {len(missing)} tensors: {missing}")
        return cls(tensors)

    # ---------------------------------------------------------------- reset
    def reset(self, context) -> None:
        """Clear history and precompute the reference derivatives for this clip."""
        if tuple(context.joint_order) != JOINT_ORDER:
            raise SuperMimicContractError(
                "evaluator joint order differs from SuperMimic's G1_JOINT_NAMES"
            )
        scale_err = np.abs(np.asarray(context.action_scale) - action_scale_reference()).max()
        if scale_err > 1e-8:
            raise SuperMimicContractError(
                f"action_scale differs from SuperMimic's 0.25*effort/stiffness by {scale_err:.3e}"
            )
        pose_err = np.abs(
            np.asarray(context.default_joint_pos) - default_joint_pos_reference()
        ).max()
        if pose_err > 1e-12:
            raise SuperMimicContractError(
                f"default_joint_pos differs from SuperMimic's keyframe by {pose_err:.3e}"
            )

        ref = context.reference
        self._context = context
        self._derived = reference_derivatives(
            ref.joint_pos, ref.root_pos_w, ref.root_quat_w, dt=context.control_dt
        )
        self._derived["yaw"] = np.stack([yaw_matrix(R) for R in self._derived["R"]])
        self._hist = None
        self._step = 0

    # ------------------------------------------------------------------ act
    def observation(self, step) -> np.ndarray:
        """``MDP.obs`` for one control step, from the evaluator's ``StepInput``."""
        k = int(step.reference_index)
        frame = k - 1
        if frame < 0:
            raise SuperMimicContractError(
                f"reference_index {k} does not leave a current state frame"
            )
        d = self._derived
        q = np.asarray(step.joint_pos, dtype=np.float64)
        q_ref = d["q"][frame]

        A = waist_rotation(q)
        A_ref = waist_rotation(q_ref)
        torso_rel = six_to_mat(_anchor_ori_at_state_frame(step))  # R_robot_torso^T R_ref_torso(f)
        g_ref = mathx.projected_gravity(np.asarray(step.reference.root_quat_w[frame]))

        pelvis_rel = A @ torso_rel @ A_ref.T
        gravity_b = A @ torso_rel @ A_ref.T @ g_ref

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
        """``MotionLib.goal(frame, 10)`` for a single clip."""
        d = self._derived
        last = len(d["p"]) - 1
        offsets = np.arange(1, self.goal_frames + 1)
        requested = frame + offsets
        idx = np.minimum(requested, last)
        recorded = requested <= last

        Ht = d["yaw"][frame].T
        dp = (d["p"][idx] - d["p"][frame]) @ Ht.T
        rot = np.einsum("ij,kjl->kil", Ht, d["R"][idx])
        v = np.where(recorded[:, None], d["root_v_local"][idx], 0.0)
        w = np.where(recorded[:, None], d["root_w_local"][idx], 0.0)
        qd = np.where(recorded[:, None], d["qd"][idx], 0.0)

        feats = np.concatenate(
            [dp, mat_to_6d(rot), v, w, d["q"][idx], qd * 0.1], axis=-1
        )
        return feats.reshape(-1)

    def forward(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """``Policy.forward``: ``dec(cat(enc(goal), obs_hist))``, deterministic."""
        goal_n = ((goal - self.goal_mean) / self.goal_std).astype(np.float32)
        x = goal_n
        for i, (w, b) in enumerate(zip(self.enc_w, self.enc_b)):
            x = _linear(x, w, b)
            if i < len(self.enc_w) - 1:
                x = _elu(x)
        z = np.tanh(x)

        obs_n = ((obs_hist - self.obs_mean) / self.obs_std).reshape(-1).astype(np.float32)
        x = np.concatenate([z, obs_n])
        for i, (w, b) in enumerate(zip(self.dec_w, self.dec_b)):
            x = _linear(x, w, b)
            if i < len(self.dec_w) - 1:
                x = _elu(x)
        return x

    def act(self, step) -> np.ndarray:
        k = int(step.reference_index)
        frame = k - 1
        if frame < 0:
            raise SuperMimicContractError(
                f"reference_index {k} does not leave a current state frame"
            )
        obs = self.observation(step)
        if self._hist is None:
            # ``evaluate.py``: the first observation is repeated ``hist`` times.
            self._hist = np.repeat(obs[None, :], self.hist, axis=0)
        else:
            self._hist = np.concatenate([self._hist[1:], obs[None, :]], axis=0)

        action = self.forward(self._hist, self.goal(frame))
        if not np.all(np.isfinite(action)):
            raise SuperMimicContractError(
                f"non-finite action at step {k} (frame {frame}) of {self._context.clip}"
            )
        return action.astype(np.float64)

    # ------------------------------------------------------------ reporting
    def parameter_count(self) -> int:
        """Weights + biases + normaliser statistics of ``pi`` (all inference reads)."""
        keep = ("weight", "bias", "mean", "var")
        return int(
            sum(t.size for key, t in self.tensors.items() if key.startswith("pi.") and key.rsplit(".", 1)[-1] in keep)
        )

    def architecture(self) -> dict[str, str]:
        """Layer widths actually executed by :meth:`forward`."""

        def widths(w):
            return [int(w[0].shape[1])] + [int(x.shape[0]) for x in w]

        return {
            "encoder": "Linear+ELU chain " + "->".join(map(str, widths(self.enc_w))) + ", tanh",
            "decoder": "Linear+ELU chain " + "->".join(map(str, widths(self.dec_w))),
            "activation": "ELU (alpha=1)",
            "observation": f"({self.hist}, {OBS_DIM}) flattened, RunningNorm(eps=1e-2)",
            "goal": f"({self.goal_frames * GOAL_PER_FRAME},) RunningNorm(eps=1e-2)",
        }


def make_policy(checkpoint: str | Path | None = None) -> SuperMimicPolicy:
    """Factory for ``--policy-file benchmark/policies/supermimic.py:make_policy``."""
    if checkpoint is None:
        raise SystemExit(
            "SuperMimicPolicy needs a checkpoint: pass --checkpoint <ckpt_XXXXXX.pt>"
        )
    return SuperMimicPolicy.from_checkpoint(checkpoint)
