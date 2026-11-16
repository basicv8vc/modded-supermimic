"""V2 configuration over the official mjlab 1.6.0 tracking task.

Design rule (task prompt §4): compose official mjlab components; add glue only
where the official implementation cannot express the originals' mathematics.

Reused verbatim (no copy, no re-implementation):
  * the 8 official reward terms from ``tracking_env_cfg.py:209-251``;
  * physics / PD / actuators / contact set / MotionCommand / CSV->npz converter
    from PR #12 (``benchmark/evaluator/``).

Config-level changes vs V1 (all in the *scoring*, none in physics):
  1. drop reward term ``self_collisions``  (neither original has it);
  2. ``joint_limit`` weight -10.0 -> -2.0 (both originals use -2.0);
  3. drop ``ee_body_pos`` termination     (neither original has it);
  4. replace official ``anchor_pos`` / ``anchor_ori`` (mjlab uses torso_link)
     with the minimal glue below (both originals use the **pelvis**).

The self-collision *ContactSensor* stays configured and the scene/contact set
is untouched: full self-collision PHYSICS is preserved, only the reward penalty
is removed.
"""

from __future__ import annotations

import copy
from typing import Any

from . import spec as S

# --- pure-data description (usable without mjlab installed) -----------------

V2_REWARD_TERMS: dict[str, dict[str, Any]] = {
    name: {"weight": S.TRACKING_WEIGHTS[name], "std": S.TRACKING_STD[name]}
    for name in S.TRACKING_WEIGHTS
}
V2_REWARD_TERMS["action_rate_l2"] = {"weight": S.W_ACTION_RATE}
V2_REWARD_TERMS["joint_limit"] = {"weight": S.W_JOINT_LIMIT, "soft_limit_factor": S.SOFT_LIMIT_FACTOR}

V2_REMOVED_REWARDS: tuple[str, ...] = ("self_collisions",)
V2_REMOVED_TERMINATIONS: tuple[str, ...] = ("anchor_pos", "anchor_ori", "ee_body_pos")
V2_TERMINATIONS: dict[str, dict[str, Any]] = {
    "time_out": {"time_out": True},
    "root_height": {"body": "pelvis", "threshold": S.TERM_HEIGHT, "fires_when": ">"},
    "root_tilt": {"body": "pelvis", "threshold": S.TERM_TILT, "fires_when": ">"},
    "nonfinite": {"fields": ["qpos", "qvel"]},
}

# Official mjlab term callables, by reward name (for static verification).
OFFICIAL_REWARD_FUNCS: dict[str, str] = {
    "motion_global_root_pos": "motion_global_anchor_position_error_exp",
    "motion_global_root_ori": "motion_global_anchor_orientation_error_exp",
    "motion_body_pos": "motion_relative_body_position_error_exp",
    "motion_body_ori": "motion_relative_body_orientation_error_exp",
    "motion_body_lin_vel": "motion_global_body_linear_velocity_error_exp",
    "motion_body_ang_vel": "motion_global_body_angular_velocity_error_exp",
    "action_rate_l2": "action_rate_l2",
    "joint_limit": "joint_pos_limits",
}


# --- minimal glue: pelvis-based termination --------------------------------
# mjlab's bad_anchor_pos_z_only / bad_anchor_ori read the *torso_link* anchor
# (terminations.py:27-53).  Both original evaluations read the pelvis free
# joint (supermimic/mdp.py:149-152, nablamimic/env.py:428-433), so the official
# term cannot express the original mathematics.  These three functions are the
# smallest possible glue and follow mjlab's TerminationTerm func signature.

def _gravity_b_z(quat_wxyz):
    """``-R[2,2]`` of a wxyz quaternion == third component of ``R^T (0,0,-1)``.

    For a unit quaternion, ``R[2,2] = 1 - 2(x^2+y^2)``.  Matches
    ``spec.gravity_b_z`` exactly (tested in test_benchmark_v2.py).
    """
    return 2.0 * (quat_wxyz[..., 1] ** 2 + quat_wxyz[..., 2] ** 2) - 1.0


def bad_root_height(env, command_name: str, threshold: float = S.TERM_HEIGHT):
    """``|z_pelvis(robot) - z_pelvis(ref)| > 0.25``."""
    import torch  # noqa: PLC0415  (glue stays importable without torch/mjlab)

    command = env.command_manager.get_term(command_name)
    pelvis_idx = command.cfg.body_names.index("pelvis")
    ref_z = command.body_pos_w[:, pelvis_idx, 2]  # body_pos_w already adds env_origins
    robot_z = env.sim.data.qpos[:, 2]
    return torch.abs(ref_z - robot_z) > threshold


def bad_root_tilt(env, command_name: str, threshold: float = S.TERM_TILT):
    """``|g_b_z(robot_pelvis) - g_b_z(ref_pelvis)| > 0.8``."""
    import torch  # noqa: PLC0415

    command = env.command_manager.get_term(command_name)
    pelvis_idx = command.cfg.body_names.index("pelvis")
    ref_q = command.body_quat_w[:, pelvis_idx, :]
    robot_q = env.sim.data.qpos[:, 3:7]
    return torch.abs(_gravity_b_z(robot_q) - _gravity_b_z(ref_q)) > threshold


def bad_nonfinite(env):
    """Failure on any non-finite qpos/qvel (NablaMimic env.py:432-433)."""
    import torch  # noqa: PLC0415

    bad = ~torch.all(torch.isfinite(env.sim.data.qpos), dim=-1)
    bad = bad | ~torch.all(torch.isfinite(env.sim.data.qvel), dim=-1)
    return bad


# --- mjlab cfg assembly ----------------------------------------------------

def build_v2_env_cfg(play: bool = True):
    """Return an official mjlab 1.6.0 tracking cfg with the V2 overrides.

    Lazy imports so that the pure-numpy protocol/tests work on machines without
    mjlab / CUDA.  ``cfg.scene`` (entities + sensors + terrain) is never
    modified -- full self-collision physics is inherited from PR #12.
    """
    from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: PLC0415
    from mjlab.tasks.tracking.config.g1.env_cfgs import (  # noqa: PLC0415
        unitree_g1_flat_tracking_env_cfg,
    )

    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    # ---- rewards: 8 official terms ----
    rewards = copy.deepcopy(cfg.rewards)
    for name in V2_REMOVED_REWARDS:
        rewards.pop(name, None)
    rewards["joint_limit"].weight = float(S.W_JOINT_LIMIT)
    cfg.rewards = rewards

    # ---- terminations: official time_out + 3 glue terms ----
    time_out = copy.deepcopy(cfg.terminations["time_out"])
    cfg.terminations = {
        "time_out": time_out,
        "root_height": TerminationTermCfg(
            func=bad_root_height,
            params={"command_name": "motion", "threshold": S.TERM_HEIGHT},
        ),
        "root_tilt": TerminationTermCfg(
            func=bad_root_tilt,
            params={"command_name": "motion", "threshold": S.TERM_TILT},
        ),
        "nonfinite": TerminationTermCfg(func=bad_nonfinite, params={}),
    }

    # ---- determinism overrides (identical to PR #12 / V1 glue) ----
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.auto_reset = False
    motion_cfg = cfg.commands["motion"]
    motion_cfg.sampling_mode = "start"
    motion_cfg.joint_position_range = (0.0, 0.0)
    cfg.events = {}
    cfg.observations["actor"].enable_corruption = False
    return cfg


def verify_v2_cfg(cfg) -> list[str]:
    """Static conformance check of a built cfg; returns a list of problems."""
    problems: list[str] = []
    if tuple(cfg.rewards) != S.REWARD_TERM_NAMES:
        problems.append(f"reward terms {tuple(cfg.rewards)} != {S.REWARD_TERM_NAMES}")
    for name in S.REWARD_TERM_NAMES:
        term = cfg.rewards.get(name)
        if term is None:
            problems.append(f"missing reward {name}")
            continue
        want = V2_REWARD_TERMS[name]["weight"]
        if float(term.weight) != want:
            problems.append(f"{name} weight {term.weight} != {want}")
        if "std" in V2_REWARD_TERMS[name] and float(term.params.get("std", float("nan"))) != V2_REWARD_TERMS[name]["std"]:
            problems.append(f"{name} std mismatch")
        want_fn = OFFICIAL_REWARD_FUNCS[name]
        if getattr(term.func, "__name__", "") != want_fn:
            problems.append(f"{name} is not the official mjlab term {want_fn}")
    for name in V2_REMOVED_REWARDS:
        if name in cfg.rewards:
            problems.append(f"forbidden reward {name} is enabled")
    for name in V2_REMOVED_TERMINATIONS:
        if name in cfg.terminations:
            problems.append(f"forbidden termination {name} is enabled")
    if tuple(cfg.terminations) != ("time_out", "root_height", "root_tilt", "nonfinite"):
        problems.append(f"unexpected termination set {tuple(cfg.terminations)}")
    # Physics must be inherited untouched from the official config.
    if not cfg.scene.entities:
        problems.append("scene entities missing")
    if not cfg.scene.sensors:
        problems.append("self-collision contact sensor removed (physics changed)")
    return problems
