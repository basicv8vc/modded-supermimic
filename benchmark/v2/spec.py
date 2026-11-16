"""Executable specification of ``modded-supermimic-g1-lafan1-v2``.

This module is the *authoritative, GPU-free* statement of V2's Reward,
Termination and Normalization.  It deliberately has no mjlab / torch / jax
import: it is the reference the parity tests compare the official mjlab terms
and the two original implementations against, and it is the definition the
small glue in :mod:`benchmark.v2.config` has to satisfy.

Provenance of every constant is in ``protocol_v2.json`` and
``docs/V2_SOURCE_COMPARISON.md``.  The mathematical definitions are:

  SuperMimic  supermimic/mdp.py:108-152, supermimic/evaluate.py:53-96 @ 8bfc1158
  NablaMimic  src/nablamimic/env.py:404-433, evaluate.py:35-116   @ bfbb2a67
  mjlab 1.6.0 tasks/tracking/mdp/rewards.py, tracking_env_cfg.py:209-251

V1 is untouched: nothing in ``benchmark/evaluator/`` is imported or modified
here except read-only access to the frozen protocol for identity checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

V2_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = V2_ROOT.parent
PROTOCOL_PATH = V2_ROOT / "protocol_v2.json"
V1_PROTOCOL_PATH = BENCHMARK_ROOT / "protocol.json"

# SHA256 of benchmark/protocol.json at PR #13 HEAD (66a6e526...).  Used by the
# test suite to prove V1's protocol was not overwritten while adding V2.
V1_PROTOCOL_SHA256 = "ddec7124dbee4c999986f995a8c2aa88097a494e1a0f4df00f2967366c15935b"

# SHA256 of benchmark/v2/protocol_v2.json, frozen by PR #14.  The V2 protocol
# (reward/termination/normalization/clips) must NOT change when GPU support is
# added; runtime tests assert this hash.
V2_PROTOCOL_SHA256 = "e5a5c0335fcc239edae4a085eb66b9ac5b93d4c44917c7168d21761c12675ba0"

PROTOCOL_ID = "modded-supermimic-g1-lafan1-v2"
EVALUATOR_IMPLEMENTATION = "canonical-mjlab-g1-v2-original-aligned"

TRACKED_BODIES: tuple[str, ...] = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_INDEX = TRACKED_BODIES.index("torso_link")  # 7 -- mjlab / NablaMimic
TERMINATION_ROOT = "pelvis"                        # both original implementations

CONTROL_DT = 0.02
# 6 tracking weights: 1,1,1,1,0.5,0.5 -> 5.0 ; x dt -> 0.1 (both originals).
TRACKING_WEIGHTS: dict[str, float] = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
}
TRACKING_STD: dict[str, float] = {
    "motion_global_root_pos": 0.3,
    "motion_global_root_ori": 0.4,
    "motion_body_pos": 0.3,
    "motion_body_ori": 0.4,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 3.14,
}
W_ACTION_RATE = -0.1
W_JOINT_LIMIT = -2.0
SOFT_LIMIT_FACTOR = 0.9
MAX_STEP_REWARD = sum(TRACKING_WEIGHTS.values()) * CONTROL_DT  # 0.1

TERM_HEIGHT = 0.25
TERM_TILT = 0.8

REWARD_TERM_NAMES: tuple[str, ...] = (*TRACKING_WEIGHTS, "action_rate_l2", "joint_limit")
TERMINATION_TERM_NAMES: tuple[str, ...] = ("root_height", "root_tilt", "nonfinite")


# --------------------------------------------------------------------------
# Protocol + identity
# --------------------------------------------------------------------------

def load_protocol_v2() -> dict:
    cfg = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == PROTOCOL_ID
    assert len(cfg["clip_ids"]) == 40 and len(set(cfg["clip_ids"])) == 40
    return cfg


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def load_v1_protocol() -> dict:
    return json.loads(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))


def v1_protocol_sha256() -> str:
    return hashlib.sha256(V1_PROTOCOL_PATH.read_bytes()).hexdigest()


def source_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=V2_ROOT.parents[2],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        return sha if len(sha) == 40 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def evaluator_identity() -> dict:
    """Independent V2 identity, never derived from V1's constants.py."""
    cfg = load_protocol_v2()
    return {
        "protocol_id": cfg["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "evaluator_implementation": cfg["evaluator_implementation"],
        "evaluator_source_tree": cfg["evaluator_source_tree"],
        "evaluator_source_git_sha": source_git_sha(),
        "mjlab": cfg["mjlab"],
        "reference_implementations": cfg["reference_implementations"],
        "depends_on_prs": cfg["depends_on_prs"],
        "v1_protocol_sha256": v1_protocol_sha256(),
    }


# --------------------------------------------------------------------------
# Quaternion / rotation helpers (wxyz, column-vector convention)
# --------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product; supports (...,4) broadcasting."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) -> rotation matrix (...,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w,x,y,z), robust near degeneracy."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    if m00 > m11 and m00 > m22:
        s = math.sqrt(max(1.0 + m00 - m11 - m22, 1e-12)) * 2
        return np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    if m11 > m22:
        s = math.sqrt(max(1.0 + m11 - m00 - m22, 1e-12)) * 2
        return np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    s = math.sqrt(max(1.0 + m22 - m00 - m11, 1e-12)) * 2
    out = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return out / np.linalg.norm(out)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by R(q).  q: (...,4), v: (...,3) -> (...,3)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q0 = np.broadcast_to(q, v.shape[:-1] + (4,))
    return np.einsum("...ij,...j->...i", quat_to_mat(q0), v)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only part of a quaternion.

    Matches supermimic/rotations.py:101-107 (``yaw_mat``),
    nablamimic/env.py:95-98 and mjlab math.py:566-586.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)
    return out


def quat_angle_sq(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Squared geodesic angle between unit quaternions.

    ``2*atan2(||v||, |w|)`` on ``q1 * conj(q2)`` -- the NablaMimic form
    (env.py:101-105), mathematically identical to SuperMimic's
    ``acos(trace)^2`` (rotations.py:94-98) and mjlab's
    ``quat_error_magnitude^2`` (math.py:590-604, 688-699).
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    e = quat_mul(q1, quat_inv(q2))
    vn = np.sqrt(np.sum(e[..., 1:] ** 2, axis=-1) + 1e-12)
    return (2.0 * np.arctan2(vn, np.abs(e[..., 0]))) ** 2


def gravity_b_z(q: np.ndarray) -> float:
    """Third component of ``R(q)^T (0,0,-1)`` == ``-R[2,2]``.

    SuperMimic termination uses ``|rR[2,2] - R[2,2]|`` (mdp.py:151);
    NablaMimic uses ``|gravity_b(q)[2] - gravity_b(rq)[2]|`` (env.py:431).
    """
    R = quat_to_mat(np.asarray(q, dtype=np.float64).reshape(4))
    return float(-R[2, 2])


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class V2State:
    """One control step, in world frame.  B = 14 tracked bodies, J = 29 joints.

    ``robot_*`` / ``ref_*`` are the quantities mjlab's MotionCommand exposes
    (``commands.py:139-197``); ``*_root_z`` / ``*_root_quat`` are pelvis
    (free joint) quantities used only by the termination, as in both originals.
    """

    robot_body_pos: np.ndarray       # (B,3)
    robot_body_quat: np.ndarray      # (B,4) wxyz
    robot_body_lin_vel: np.ndarray   # (B,3) world, body-origin
    robot_body_ang_vel: np.ndarray   # (B,3) world
    ref_body_pos: np.ndarray         # (B,3)
    ref_body_quat: np.ndarray        # (B,4) wxyz
    ref_body_lin_vel: np.ndarray     # (B,3)
    ref_body_ang_vel: np.ndarray     # (B,3)
    robot_root_z: float              # pelvis height
    robot_root_quat: np.ndarray      # (4,) pelvis wxyz
    ref_root_z: float                # pelvis height
    ref_root_quat: np.ndarray        # (4,) pelvis wxyz
    joint_pos: np.ndarray            # (J,)
    action: np.ndarray               # (J,)
    prev_action: np.ndarray          # (J,)
    soft_lo: np.ndarray              # (J,)
    soft_hi: np.ndarray              # (J,)
    qpos: np.ndarray | None = None   # full MuJoCo qpos, for the nonfinite guard
    qvel: np.ndarray | None = None   # full MuJoCo qvel, for the nonfinite guard


def reference_reanchor(state: V2State) -> tuple[np.ndarray, np.ndarray]:
    """BeyondMimic xy+yaw re-anchoring around the torso_link anchor.

    Verbatim semantics of:
      mjlab tasks/tracking/mdp/commands.py:385-407
      nablamimic/env.py:389-402
      supermimic/mdp.py:113-118  (same form, but around the *pelvis* root)

    Returns ``(rel_pos (B,3), rel_quat (B,4))``.
    """
    k = ANCHOR_INDEX
    ra_p = np.asarray(state.robot_body_pos[k], dtype=np.float64)
    ra_q = np.asarray(state.robot_body_quat[k], dtype=np.float64)
    ma_p = np.asarray(state.ref_body_pos[k], dtype=np.float64)
    ma_q = np.asarray(state.ref_body_quat[k], dtype=np.float64)

    d_pos = np.array([ra_p[0], ra_p[1], ma_p[2]])
    d_q = yaw_quat(quat_mul(ra_q, quat_inv(ma_q)))[0]
    rel_pos = d_pos + quat_rotate(d_q, np.asarray(state.ref_body_pos) - ma_p)
    rel_quat = quat_mul(d_q, np.asarray(state.ref_body_quat))
    return rel_pos, rel_quat


def reward_errors(state: V2State, rel_pos=None, rel_quat=None) -> dict[str, float]:
    """The six tracking errors (before exp / weight / dt).

    ``rel_pos`` / ``rel_quat`` optionally supply an already re-anchored
    reference (the pair :func:`reference_reanchor` returns).  This is used by
    the **GPU runtime parity harness**: the official mjlab body-position /
    body-orientation reward terms do not recompute re-anchoring, they read the
    ``MotionCommand`` CUDA buffers ``body_pos_relative_w`` /
    ``body_quat_relative_w`` (``commands.py:379-407``, ``rewards.py:39-70``).
    Those buffers are *reference state*, not RewardManager outputs.  Feeding
    them to the oracle makes A and B evaluate the **identical state and
    reference frame** (task §6) while the oracle still computes every error,
    ``exp``, weight and dt scaling itself.

    When omitted, the re-anchoring is recomputed from ``state`` -- the
    GPU-free V2 definition used by the record / normalization / source-parity
    tests.  The mathematical definition is unchanged in both paths.
    """
    if rel_pos is None or rel_quat is None:
        rel_pos, rel_quat = reference_reanchor(state)
    rel_pos = np.asarray(rel_pos, dtype=np.float64)
    rel_quat = np.asarray(rel_quat, dtype=np.float64)
    rp = np.asarray(state.robot_body_pos, dtype=np.float64)
    rq = np.asarray(state.robot_body_quat, dtype=np.float64)
    k = ANCHOR_INDEX
    return {
        "motion_global_root_pos": float(np.sum((state.ref_body_pos[k] - rp[k]) ** 2)),
        "motion_global_root_ori": float(quat_angle_sq(state.ref_body_quat[k], rq[k])),
        "motion_body_pos": float(np.mean(np.sum((rel_pos - rp) ** 2, axis=-1))),
        "motion_body_ori": float(np.mean(quat_angle_sq(rel_quat, rq))),
        "motion_body_lin_vel": float(np.mean(np.sum(
            (np.asarray(state.ref_body_lin_vel) - np.asarray(state.robot_body_lin_vel)) ** 2, axis=-1))),
        "motion_body_ang_vel": float(np.mean(np.sum(
            (np.asarray(state.ref_body_ang_vel) - np.asarray(state.robot_body_ang_vel)) ** 2, axis=-1))),
    }


def reward_terms(state: V2State, rel_pos=None, rel_quat=None) -> dict[str, float]:
    """V2 per-step reward, split into the 8 terms, already scaled by dt.

    ``sum(reward_terms(state).values())`` is the ``r_t`` of the normalization.
    ``rel_pos`` / ``rel_quat`` are the optional runtime-supplied re-anchored
    reference (see :func:`reward_errors`); they never come from a reward.
    """
    errors = reward_errors(state, rel_pos=rel_pos, rel_quat=rel_quat)
    terms = {
        name: CONTROL_DT * TRACKING_WEIGHTS[name] * math.exp(-e / TRACKING_STD[name] ** 2)
        for name, e in errors.items()
    }
    a = np.asarray(state.action, dtype=np.float64)
    a_prev = np.asarray(state.prev_action, dtype=np.float64)
    q = np.asarray(state.joint_pos, dtype=np.float64)
    terms["action_rate_l2"] = CONTROL_DT * W_ACTION_RATE * float(np.sum((a - a_prev) ** 2))
    terms["joint_limit"] = CONTROL_DT * W_JOINT_LIMIT * float(np.sum(
        np.maximum(q - state.soft_hi, 0.0) + np.maximum(state.soft_lo - q, 0.0)))
    return terms


def reward_terms_from_relative_reference(state: V2State, rel_pos, rel_quat) -> dict[str, float]:
    """Runtime-parity entry point: the *same* V2 math, GPU reference buffers.

    ``rel_pos`` / ``rel_quat`` must be ``MotionCommand.body_pos_relative_w`` /
    ``body_quat_relative_w`` read from the CUDA runtime at the instant the
    RewardManager evaluated its terms.  No RewardManager output is read here;
    this function is exactly :func:`reward_terms` with an explicit reference.
    """
    return reward_terms(
        state,
        rel_pos=np.asarray(rel_pos, dtype=np.float64),
        rel_quat=np.asarray(rel_quat, dtype=np.float64),
    )


def reward(state: V2State) -> float:
    return float(sum(reward_terms(state).values()))


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    reasons: tuple[str, ...]


def nonfinite(state: V2State) -> bool:
    arrays: list[np.ndarray] = [
        state.robot_body_pos, state.robot_body_quat, state.robot_body_lin_vel,
        state.robot_body_ang_vel, state.ref_body_pos, state.ref_body_quat,
        np.asarray([state.robot_root_z, state.ref_root_z]),
        np.asarray(state.robot_root_quat), np.asarray(state.ref_root_quat),
        state.joint_pos, state.action, state.prev_action,
    ]
    if state.qpos is not None:
        arrays.append(np.asarray(state.qpos))
    if state.qvel is not None:
        arrays.append(np.asarray(state.qvel))
    return not all(bool(np.all(np.isfinite(np.asarray(x, dtype=np.float64)))) for x in arrays)


def termination(state: V2State) -> TerminationResult:
    """V2 termination: pelvis height / pelvis tilt / nonfinite.  No ee term.

    Strict ``>`` at the boundary, exactly like
    supermimic/mdp.py:152 and nablamimic/env.py:433.
    Reaching the clip end is *not* handled here -- it is a driver-level
    success, see :func:`clip_end_success`.
    """
    reasons: list[str] = []
    if abs(state.robot_root_z - state.ref_root_z) > TERM_HEIGHT:
        reasons.append("root_height")
    if abs(gravity_b_z(state.robot_root_quat) - gravity_b_z(state.ref_root_quat)) > TERM_TILT:
        reasons.append("root_tilt")
    if nonfinite(state):
        reasons.append("nonfinite")
    return TerminationResult(bool(reasons), tuple(reasons))


def clip_end_success(step: int, n_steps: int, term: TerminationResult) -> bool:
    """True iff the clip reached its last control step without a termination."""
    return (step >= n_steps) and not term.terminated


# --------------------------------------------------------------------------
# Normalization (full clip, no clipping)
# --------------------------------------------------------------------------

def rollout_normalized_return(
    per_step_rewards: Sequence[float] | np.ndarray,
    n_steps: int,
    terminated_at: int | None = None,
) -> float:
    """Full-clip normalized return for one clip.

    Reproduces ``supermimic/evaluate.py:81-84`` and
    ``nablamimic/evaluate.py:60, 84, 90``:

    * ``n_steps = T - 1`` (the full clip, regardless of early termination),
    * the terminating step's reward **is** counted (all three implementations
      mask with ``alive`` *before* updating it),
    * every step after termination contributes exactly zero,
    * the denominator is ``(T-1) * MAX_STEP_REWARD`` = full-clip ceiling,
    * **no** ``[0,1]`` clamp; negatives and values > 1 are preserved.

    ``terminated_at`` is the 0-based index of the terminating step (or None).
    """
    if n_steps < 1:
        raise ValueError("clip must have at least 2 reference frames")
    r = np.asarray(per_step_rewards, dtype=np.float64).reshape(-1)
    if r.size != n_steps:
        raise ValueError(f"expected {n_steps} per-step rewards, got {r.size}")
    if terminated_at is not None:
        if not 0 <= terminated_at < n_steps:
            raise ValueError("terminated_at out of range")
        r = np.where(np.arange(n_steps) <= terminated_at, r, 0.0)
    return float(np.sum(r) / (n_steps * MAX_STEP_REWARD))


def aggregate(per_clip_returns: Iterable[float]) -> float:
    """Unweighted mean over clips, as in both original evaluators."""
    vals = np.asarray(list(per_clip_returns), dtype=np.float64)
    if vals.size == 0:
        raise ValueError("no clips")
    return float(np.mean(vals))


def failed_clips(per_clip_failed: Iterable[bool]) -> int:
    return int(sum(bool(x) for x in per_clip_failed))


def v1_clamped_normalized_return(raw: float) -> float:
    """The V1-only clamp, kept here *for tests*, never used by V2.

    V1 ``benchmark/evaluator/runner.py:164`` does exactly this.  V2 must not.
    """
    return float(min(max(raw, 0.0), 1.0))


# --------------------------------------------------------------------------
# V2 record validation (does not touch benchmark/core.py)
# --------------------------------------------------------------------------

def audit_v2_record(record: Any) -> dict:
    """Format/provenance check for a V2 record.  Never trusts the aggregate.

    Unlike V1's ``core.audit_record``, a V2 ``normalized_return`` may be
    negative or greater than 1 -- no clamping is applied or required.
    """
    cfg = load_protocol_v2()
    errors: list[str] = []
    result: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "status": "INVALID",
        "physics_e2e_verified": False,
        "errors": errors,
    }
    if not isinstance(record, dict):
        errors.append("record must be a JSON object")
        return result
    if record.get("protocol_id") != cfg["protocol_id"]:
        errors.append("protocol_id mismatch (a V1 record is not a V2 record)")
    ev = record.get("evaluator") or {}
    if ev.get("implementation") != cfg["evaluator_implementation"]:
        errors.append("evaluator.implementation mismatch")
    if ev.get("self_collision") is not True:
        errors.append("full self-collision physics must be enabled")
    if ev.get("self_collision_reward_penalty") is not False:
        errors.append("V2 must declare self_collision_reward_penalty=false")
    for key, expected in cfg["required_versions"].items():
        if ev.get(key) != expected:
            errors.append(f"evaluator version {key} must equal {expected}")

    clip_ids = set(cfg["clip_ids"])
    cps = record.get("checkpoints")
    if not isinstance(cps, list) or not cps:
        errors.append("checkpoints must be a nonempty list")
        return result
    returns: list[float] = []
    failed_total = 0
    for i, cp in enumerate(cps):
        rows = cp.get("per_clip") if isinstance(cp, dict) else None
        if not isinstance(rows, list) or len(rows) != 40:
            errors.append(f"checkpoints[{i}].per_clip must have 40 entries")
            continue
        ids: list[str] = []
        for j, row in enumerate(rows):
            cid = row.get("clip")
            if isinstance(cid, str):
                ids.append(cid)
            ret = row.get("normalized_return")
            if (not isinstance(ret, (int, float)) or isinstance(ret, bool)
                    or not math.isfinite(float(ret))):
                errors.append(f"checkpoints[{i}].per_clip[{j}].normalized_return must be finite")
            else:
                # Deliberately NO [0,1] range check.
                returns.append(float(ret))
            fl = row.get("failed")
            if type(fl) is not bool:
                errors.append(f"checkpoints[{i}].per_clip[{j}].failed must be boolean")
            elif fl:
                failed_total += 1
        if len(set(ids)) != 40 or set(ids) != clip_ids:
            errors.append(f"checkpoints[{i}] clip id set mismatch")
    if not errors:
        result["status"] = "FORMAT_VALID_PENDING_PHYSICS_AUDIT"
        result["aggregate_normalized_return"] = aggregate(returns)
        result["failed_clips"] = failed_total
        result["clipping_applied"] = False
    return result


# --------------------------------------------------------------------------
# Smoke records (task prompt §4): 1-2 clips, never a leaderboard submission.
#
# This is a *separate* validator.  ``audit_v2_record`` above is intentionally
# left byte-for-byte strict: a formal benchmark record still needs the exact
# 40 protocol clips and every Integrity Gate unchanged.
# --------------------------------------------------------------------------

SMOKE_MIN_CLIPS = 1
SMOKE_MAX_CLIPS = 2


def audit_v2_smoke(record: Any) -> dict:
    """Validate a CUDA runtime smoke record.  Never relaxes the 40-clip gate.

    A smoke record must be explicitly ``mode="smoke"`` and
    ``submission_eligible=False``, carry 1-2 known clips, and include
    runtime-detected CUDA provenance (a CLI ``--gpu-name`` string is not
    accepted as evidence).
    """
    cfg = load_protocol_v2()
    errors: list[str] = []
    result: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "status": "INVALID",
        "mode": "smoke",
        "submission_eligible": False,
        "errors": errors,
    }
    if not isinstance(record, dict):
        errors.append("record must be a JSON object")
        return result
    if record.get("protocol_id") != cfg["protocol_id"]:
        errors.append("protocol_id mismatch")
    ev = record.get("evaluation") or {}
    if ev.get("mode") != "smoke":
        errors.append("evaluation.mode must be 'smoke'")
    if ev.get("submission_eligible") is not False:
        errors.append("smoke must declare submission_eligible=false")
    if record.get("submission_eligible") is not False:
        errors.append("top-level submission_eligible=false is required")
    prov = ev.get("provenance") or {}
    actual_dev = str(prov.get("actual_torch_device", ""))
    if not actual_dev.startswith("cuda"):
        errors.append("provenance.actual_torch_device must be a CUDA device (no CPU fallback)")
    if not prov.get("actual_simulation_backend"):
        errors.append("provenance.actual_simulation_backend is missing")
    if prov.get("cuda_graph_step_captured") is not True:
        errors.append("no captured CUDA step graph: GPU physics not verified")
    if bool(prov.get("version_drift")):
        # Recorded, but does not invalidate a runtime smoke.
        result["version_drift"] = prov["version_drift"]
    if record.get("evaluator", {}).get("implementation") != cfg["evaluator_implementation"]:
        errors.append("evaluator.implementation mismatch")
    if record.get("evaluator", {}).get("self_collision_reward_penalty") is not False:
        errors.append("V2 must declare self_collision_reward_penalty=false")

    cps = record.get("checkpoints")
    if not isinstance(cps, list) or len(cps) != 1:
        errors.append("smoke needs exactly one checkpoints entry")
        return result
    rows = cps[0].get("per_clip")
    if not isinstance(rows, list) or not SMOKE_MIN_CLIPS <= len(rows) <= SMOKE_MAX_CLIPS:
        errors.append(f"smoke per_clip must have {SMOKE_MIN_CLIPS}-{SMOKE_MAX_CLIPS} entries")
        return result
    known = set(cfg["clip_ids"])
    seen: set[str] = set()
    for j, row in enumerate(rows):
        cid = row.get("clip")
        if cid not in known:
            errors.append(f"smoke clip {cid!r} is not in the protocol")
        seen.add(cid)
        ret = row.get("normalized_return")
        if (not isinstance(ret, (int, float)) or isinstance(ret, bool)
                or not math.isfinite(float(ret))):
            errors.append(f"smoke per_clip[{j}].normalized_return must be finite")
        if type(row.get("failed")) is not bool:
            errors.append(f"smoke per_clip[{j}].failed must be boolean")
    if len(seen) != len(rows):
        errors.append("duplicate smoke clip ids")
    if not errors:
        result["status"] = "SMOKE_VALID"
        result["clips"] = sorted(seen)
        result["aggregate_normalized_return"] = aggregate(
            float(r["normalized_return"]) for r in rows)
        result["failed_clips"] = sum(bool(r["failed"]) for r in rows)
    return result


def is_submission_eligible(record: Any) -> bool:
    """True only for a full 40-clip benchmark record that passes all gates.

    Smoke records are unconditionally ineligible, even if someone pads them
    with 40 rows: ``mode`` and ``submission_eligible`` are checked as well.
    """
    if not isinstance(record, dict):
        return False
    ev = record.get("evaluation") or {}
    if ev.get("mode") != "benchmark" or ev.get("submission_eligible") is not True:
        return False
    if record.get("submission_eligible") is not True:
        return False
    audit = audit_v2_record(record)
    return audit["status"].startswith("FORMAT_VALID") and not audit["errors"]


def is_comparison_valid(record: Any) -> bool:
    """True for a full 40-clip benchmark record valid for controlled comparison.

    Unlike formal leaderboard submission (which strictly requires target hardware
    and submission_eligible=True), a controlled baseline run on non-target hardware
    (e.g. A100 / L4) satisfies all 40-clip integrity gates, has finite returns,
    and valid protocol SHA, with submission_eligible=False, comparison_valid=True.
    """
    if not isinstance(record, dict):
        return False
    ev = record.get("evaluation") or {}
    if ev.get("mode") != "benchmark":
        return False
    if record.get("comparison_valid") is not True:
        return False
    audit = audit_v2_record(record)
    return audit["status"].startswith("FORMAT_VALID") and not audit["errors"]
