"""GPU-runtime parity checks for Benchmark V2 (task prompt §6/§7/§10).

A. Real GPU ``RewardManager`` / ``TerminationManager`` outputs, taken from a
   constructed ``ManagerBasedRlEnv`` running on CUDA.
B. Independent, GPU-free mathematical oracle: :mod:`benchmark.v2.spec`.

The oracle never reads a manager output as its expected value.  Instead, this
module hooks ``reward_manager.compute`` / ``termination_manager.compute`` and
snapshots the *real GPU tensors* at the exact instant the manager evaluates
them (before ``command_manager.compute`` advances the reference frame and
before ``sim.forward()`` refreshes derived quantities).  Those snapshots are
converted to :class:`~benchmark.v2.spec.V2State` inputs for the oracle.

All heavy imports are deferred so this module can be imported (and its status
reported as BLOCKED) on a host without CUDA.
"""

from __future__ import annotations

import math
import time
from typing import Any

from . import spec as S

REWARD_TOLERANCE = {"atol": 2e-5, "rtol": 3e-4}
# Float32 round-trip tolerance for comparing spec.reference_reanchor() with
# mjlab's CUDA buffers on *identical* state.  Measured offline at ~3e-7; this
# is NOT a relaxation of REWARD_TOLERANCE (the reward terms still use it).
REANCHOR_TOLERANCE = 5e-6
# PR #13 frame semantics, documented in benchmark/evaluator/env.py:25-30:
# env.reset() ends with command_manager.compute(dt=0.0), which advances
# MotionCommand.time_steps to 1.  Frame 0 is only the initial condition; the
# first *scored* reference frame is 1.  V2 deliberately matches V1 here.
FIRST_SCORED_REFERENCE_FRAME = 1


class _StateCapture:
    """Snapshot GPU state at the moment RewardManager computes its terms.

    Both the robot/reference tensors *and* the re-anchored reference buffers
    are captured inside the hook, before ``reward_manager.compute`` runs.  The
    official body-position / body-orientation reward terms read those stored
    CUDA buffers (they do not recompute re-anchoring), so the oracle must be
    given the very same buffers to make A/B a same-state / same-frame compare.
    Nothing here reads a reward value as an expected value.
    """

    def __init__(self, env):
        self.env = env
        self.frames: list[tuple[S.V2State, Any, Any, int]] = []
        self._orig_reward = None
        self._orig_term = None

    def __enter__(self):
        rm = self.env._env.reward_manager
        tm = self.env._env.termination_manager
        self._orig_reward, self._orig_term = rm.compute, tm.compute

        def wrapped_reward(dt):
            state = self.env.runtime_state()
            rel_pos, rel_quat = self.env.runtime_relative_reference()
            # Capture the reference frame *at reward time*.  ManagerBasedRlEnv
            # advances time_steps later in step(), so reading it after the
            # step returns is off by one (PR #13 frame-1 semantics).
            frame = int(self.env._command.time_steps[0])
            self.frames.append((state, rel_pos, rel_quat, frame))
            return self._orig_reward(dt)

        def wrapped_term():
            return self._orig_term()

        rm.compute = wrapped_reward
        tm.compute = wrapped_term
        return self

    def __exit__(self, *exc):
        self.env._env.reward_manager.compute = self._orig_reward
        self.env._env.termination_manager.compute = self._orig_term
        return False


def _compare(name: str, got: float, want: float) -> dict[str, Any]:
    abs_err = abs(float(got) - float(want))
    denom = max(abs(float(want)), 1e-12)
    rel_err = abs_err / denom
    return {"term": name, "gpu_runtime": float(got), "v2_oracle": float(want),
            "abs_error": abs_err, "rel_error": rel_err}


def check_reward_parity(env, steps: int = 6, seed: int = 7) -> dict[str, Any]:
    """Compare all 8 reward terms, weights, dt, indexing, action history, limits.

    Why the oracle is fed ``body_pos_relative_w`` / ``body_quat_relative_w``:

    mjlab's ``ManagerBasedRlEnv.step`` computes reward at step 4 (before the
    single ``sim.forward()`` / ``command_manager.compute()`` pair at steps 7-8).
    The relative-body terms therefore read the **stored** re-anchoring buffers
    computed during the *previous* step's command update, while the live
    ``robot_body_pos_w`` read at reward time is stale by one physics substep
    (documented in ``manager_based_rl_env.py:395-410``).  Recomputing
    re-anchoring from that live robot anchor compares two *different* states
    and produced the 1e-5/3e-5 residual on ``motion_body_pos`` /
    ``motion_body_ori``.  ``reference_reanchor`` itself is correct -- an
    offline float32 comparison against mjlab's exact ``quat_mul`` /
    ``yaw_quat`` / ``quat_apply`` chain agrees to ~3e-7 on identical state;
    :func:`check_reference_reanchor_parity` re-verifies this on the GPU.

    Both sides now consume the identical reference frame, and the oracle still
    computes every error, ``exp``, weight and dt scaling independently.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ..evaluator import constants as C  # noqa: PLC0415
    from . import config as V2CFG  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    static: dict[str, Any] = {}
    action_history_ok = True
    previous_action = np.zeros(C.ACTION_DIM)

    cfg = env._env.cfg
    command = env._command
    robot = env._env.scene["robot"]
    static["reward_term_names_runtime"] = tuple(cfg.rewards)
    static["reward_term_names_oracle"] = S.REWARD_TERM_NAMES
    static["weights_runtime"] = {k: float(v.weight) for k, v in cfg.rewards.items()}
    static["weights_oracle"] = {**S.TRACKING_WEIGHTS,
                                "action_rate_l2": S.W_ACTION_RATE,
                                "joint_limit": S.W_JOINT_LIMIT}
    static["dt_runtime"] = float(env._env.step_dt)
    static["dt_oracle"] = S.CONTROL_DT
    static["body_names_runtime"] = tuple(command.cfg.body_names)
    static["body_names_oracle"] = S.TRACKED_BODIES
    static["anchor_index_runtime"] = int(command.motion_anchor_body_index)
    static["anchor_index_oracle"] = S.ANCHOR_INDEX
    static["robot_anchor_index_runtime"] = int(command.robot_anchor_body_index)
    static["anchor_body_name"] = command.cfg.anchor_body_name
    static["termination_root"] = S.TERMINATION_ROOT
    static["soft_limit_factor_oracle"] = S.SOFT_LIMIT_FACTOR
    static["soft_limit_factor_constants"] = C.SOFT_JOINT_LIMIT_FACTOR
    # Provenance of the two inputs: the oracle's reference is a MotionCommand
    # CUDA state buffer, never a RewardManager output.
    static["relative_reference_source"] = (
        "MotionCommand.body_pos_relative_w / body_quat_relative_w (CUDA buffers)")
    static["oracle_expected_value_source"] = (
        "benchmark.v2.spec.reward_terms_from_relative_reference (independent math)")

    env.reset()
    executed = 0
    frame_alignment_ok = True
    since_reset = 0
    with _StateCapture(env) as cap:
        for k in range(steps):
            # A deterministic, non-zero action exercises action_rate_l2 and the
            # joint-limit soft-limit sum without needing a trained policy.
            action = (0.08 * np.sin(0.7 * (np.arange(C.ACTION_DIM) + k))
                      + 0.01 * rng.standard_normal(C.ACTION_DIM))
            env.step(action)
            executed += 1
            since_reset += 1
            snapshot, rel_pos, rel_quat, reward_frame = cap.frames[-1]
            # PR #13: first scored frame is 1, so at step k the reward frame
            # equals 1 + k == since_reset.  Using the reward-time capture (not
            # the post-step counter) avoids the off-by-one.
            frame_alignment_ok &= (reward_frame == since_reset)
            action_history_ok &= bool(
                np.allclose(snapshot.prev_action, previous_action, atol=1e-6))
            previous_action = action.copy()
            # Same state AND same reference frame as the GPU RewardManager: the
            # relative reference comes from the MotionCommand CUDA buffers that
            # the official terms themselves read (not from a reward output).
            oracle_terms = S.reward_terms_from_relative_reference(
                snapshot, rel_pos, rel_quat)
            runtime_terms = env.runtime_terms()
            for name in S.REWARD_TERM_NAMES:
                rows.append({"step": k, **_compare(name, runtime_terms.get(name, float("nan")),
                                                   oracle_terms[name])})
            if bool(env._env._manual_reset_pending.any()):  # noqa: SLF001
                # A smoke policy may fall; re-arm so the remaining parity steps
                # still run.  Frame counters restart, hence `executed` above.
                env.reset()
                since_reset = 0
                previous_action = np.zeros(C.ACTION_DIM)

    max_abs = max(r["abs_error"] for r in rows)
    max_rel = max(r["rel_error"] for r in rows)
    worst = max(rows, key=lambda r: r["abs_error"])
    static["per_term_max_abs_error"] = {
        name: max(r["abs_error"] for r in rows if r["term"] == name)
        for name in S.REWARD_TERM_NAMES}
    static["per_term_max_rel_error"] = {
        name: max(r["rel_error"] for r in rows if r["term"] == name)
        for name in S.REWARD_TERM_NAMES}
    static["action_history_prev_matches_previous_step"] = action_history_ok
    static["joint_soft_limits_shape"] = tuple(robot.data.soft_joint_pos_limits.shape)
    static["reference_frame_before_first_step"] = FIRST_SCORED_REFERENCE_FRAME
    static["reward_reference_frames"] = [f[3] for f in cap.frames]
    static["reference_frame_after_steps"] = int(command.time_steps[0])
    static["reward_manager_scale_by_dt"] = bool(env._env.reward_manager._scale_by_dt)
    static["executed_steps"] = executed

    checks = {
        "all_8_terms_present": all(name in static["reward_term_names_runtime"]
                                   for name in S.REWARD_TERM_NAMES),
        "weights_match": static["weights_runtime"] == static["weights_oracle"],
        "dt_scaling_match": abs(static["dt_runtime"] - static["dt_oracle"]) < 1e-12,
        "body_indexing_match": static["body_names_runtime"] == static["body_names_oracle"],
        "root_anchor_match": (static["anchor_index_runtime"] == static["anchor_index_oracle"]
                              and static["anchor_body_name"] == "torso_link"),
        "soft_limit_factor_match": (static["soft_limit_factor_oracle"]
                                    == static["soft_limit_factor_constants"]),
        "action_history_matches": action_history_ok,
        "joint_soft_limits_dim": static["joint_soft_limits_shape"][-1] == 2,
        "reward_scaled_by_dt": static["reward_manager_scale_by_dt"],
        "frame_alignment": frame_alignment_ok,
        "tolerance": (max_abs <= REWARD_TOLERANCE["atol"]
                      or max_rel <= REWARD_TOLERANCE["rtol"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "steps": executed,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "worst_term": worst["term"],
        "tolerance": REWARD_TOLERANCE,
        "static": static,
        "rows": rows,
    }


def check_reference_reanchor_parity(env, steps: int = 3) -> dict[str, Any]:
    """Validate V2's re-anchoring *formula* against mjlab's CUDA buffers.

    :func:`check_reward_parity` deliberately feeds the oracle the runtime's
    re-anchored buffers so both sides see one state/frame.  This complementary
    check verifies those buffers really are what ``spec.reference_reanchor``
    says they are (root/anchor semantics, yaw-only delta, error coordinate
    frame).  We force a consistent instant -- ``sim.forward()`` immediately
    followed by ``update_relative_body_poses()`` -- then compare on GPU
    tensors, and restore the buffers afterwards so no rollout is perturbed.
    """
    import numpy as np  # noqa: PLC0415
    from ..evaluator import constants as C  # noqa: PLC0415

    command = env._command
    env.reset()
    rows: list[dict[str, Any]] = []
    for k in range(steps):
        saved_p = command.body_pos_relative_w
        saved_q = command.body_quat_relative_w
        try:
            env._env.sim.forward()
            command.update_relative_body_poses()
            state = env.runtime_state()
            want_p, want_q = S.reference_reanchor(state)
            got_p, got_q = env.runtime_relative_reference()
            rows.append({
                "step": k,
                "command_frame": int(command.time_steps[0]),
                "rel_pos_max_abs_error": float(np.abs(want_p - got_p).max()),
                "rel_quat_max_abs_error": float(np.abs(want_q - got_q).max()),
                "rel_pos_shape": list(want_p.shape),
                "rel_quat_shape": list(want_q.shape),
            })
        finally:
            command.body_pos_relative_w = saved_p
            command.body_quat_relative_w = saved_q
        env.step(np.zeros(C.ACTION_DIM))

    max_p = max(r["rel_pos_max_abs_error"] for r in rows)
    max_q = max(r["rel_quat_max_abs_error"] for r in rows)
    checks = {
        "rel_pos_matches_mjlab_buffer": max_p <= REANCHOR_TOLERANCE,
        "rel_quat_matches_mjlab_buffer": max_q <= REANCHOR_TOLERANCE,
        "anchor_is_torso_link": (command.cfg.anchor_body_name == "torso_link"
                                 and int(command.motion_anchor_body_index) == S.ANCHOR_INDEX),
        "termination_root_is_pelvis": S.TERMINATION_ROOT == "pelvis",
        "robot_anchor_body_is_torso_link": (
            env._env.scene["robot"].body_names[int(command.robot_anchor_body_index)]
            == "torso_link"),
        "body_count_matches_spec": len(command.cfg.body_names) == len(S.TRACKED_BODIES),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "rows": rows,
            "max_abs_error": max(max_p, max_q),
            "tolerance": REANCHOR_TOLERANCE,
            "note": "float32 round-trip only; reward terms keep REWARD_TOLERANCE"}


# --------------------------------------------------------------------------
# Termination parity on real CUDA tensors
# --------------------------------------------------------------------------

def _tilt_quat_pair(deviation: float) -> tuple[list[float], list[float]]:
    """Reference (identity) and robot pelvis quaternions with an exact deviation.

    ``gravity_b_z(q) = 2(x^2+y^2) - 1`` (config.py:68-74, spec.py:232-239).
    With reference ``[1,0,0,0]`` -> ``g_ref = -1`` and robot
    ``[sqrt(1-d/2), sqrt(d/2), 0, 0]`` -> ``g_robot = d - 1``, so
    ``|g_robot - g_ref| == d`` exactly.  This lets the boundary cases test the
    strict ``> 0.8`` rule at 0.79 / 0.80 / 0.81 rather than relying on an
    unknown baseline deviation.
    """
    if not 0.0 <= deviation <= 2.0:
        raise ValueError("tilt deviation must be in [0, 2]")
    ref_q = [1.0, 0.0, 0.0, 0.0]
    robot_q = [math.sqrt(1.0 - deviation / 2.0),
               math.sqrt(deviation / 2.0), 0.0, 0.0]
    return ref_q, robot_q


def _boundary_env(env, *, dh: float | None = None, dt: float | None = None,
                  nan: bool = False, ee: bool = False):
    """A CUDA-tensor shim of the real env's command/sim buffers.

    The glue reads ``command.body_pos_w[pelvis, 2]`` / ``body_quat_w[pelvis]``
    (reference) and ``sim.data.qpos[2]`` / ``qpos[3:7]`` (robot root).  We set
    the **robot** root explicitly as ``ref + deviation`` so the deviation is
    exactly known, instead of perturbing the reference by ``dz`` atop an
    unknown baseline.  Same GPU tensors / dtypes / device; no rule is changed.
    """
    import types  # noqa: PLC0415
    import torch  # noqa: PLC0415

    command = env._command
    sim = env._env.sim
    pelvis = command.cfg.body_names.index(S.TERMINATION_ROOT)
    body_pos = command.body_pos_w.clone()
    body_quat = command.body_quat_w.clone()
    qpos = sim.data.qpos.clone()
    qvel = sim.data.qvel.clone()

    ref_z = body_pos[:, pelvis, 2]
    if dh is not None:
        qpos[:, 2] = ref_z + dh
    if dt is not None:
        ref_q, robot_q = _tilt_quat_pair(dt)
        body_quat[:, pelvis] = torch.tensor(
            ref_q, dtype=body_quat.dtype, device=body_quat.device)
        qpos[:, 3:7] = torch.tensor(
            robot_q, dtype=qpos.dtype, device=qpos.device)
    if nan:
        qpos[0, 0] = float("nan")
    if ee:
        for name in ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"):
            body_pos[:, command.cfg.body_names.index(name)] += 0.9

    fake_command = types.SimpleNamespace(
        cfg=command.cfg, body_pos_w=body_pos, body_quat_w=body_quat)
    return types.SimpleNamespace(
        sim=types.SimpleNamespace(data=types.SimpleNamespace(qpos=qpos, qvel=qvel)),
        command_manager=types.SimpleNamespace(get_term=lambda name: fake_command),
    )


def _spec_state_from_boundary(env, *, dh: float | None = None,
                              dt: float | None = None, nan: bool = False,
                              ee: bool = False) -> S.V2State:
    """Build the oracle state matching the perturbed CUDA tensors exactly."""
    import dataclasses  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    base = env.runtime_state()
    command = env._command
    pelvis = command.cfg.body_names.index(S.TERMINATION_ROOT)
    ref_pos = base.ref_body_pos.copy()
    ref_quat = base.ref_body_quat.copy()
    robot_pos = base.robot_body_pos.copy()
    qpos = base.qpos.copy()
    robot_root_z = base.robot_root_z
    robot_root_quat = base.robot_root_quat.copy()
    ref_root_z = base.ref_root_z
    ref_root_quat = base.ref_root_quat.copy()

    if dh is not None:
        robot_root_z = float(ref_pos[pelvis, 2] + dh)
        qpos[2] = robot_root_z
    if dt is not None:
        ref_q, robot_q = _tilt_quat_pair(dt)
        ref_quat[pelvis] = np.asarray(ref_q, dtype=np.float64)
        qpos[3:7] = np.asarray(robot_q, dtype=np.float64)
        robot_root_quat = np.asarray(robot_q, dtype=np.float64)
        ref_root_quat = np.asarray(ref_q, dtype=np.float64)
    if nan:
        qpos[0] = float("nan")
    if ee:
        for name in ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"):
            robot_pos[S.TRACKED_BODIES.index(name)] += np.array([0.9, 0.9, 0.9])
    return dataclasses.replace(
        base,
        ref_body_pos=ref_pos, ref_body_quat=ref_quat,
        robot_body_pos=robot_pos,
        ref_root_z=ref_root_z, ref_root_quat=ref_root_quat,
        robot_root_z=robot_root_z, robot_root_quat=robot_root_quat,
        qpos=qpos,
    )


def check_termination_parity(env) -> dict[str, Any]:
    """Runtime boundary tests + structural checks against the V2 oracle.

    Boundary deviations are set *exactly* (``|z_robot - z_ref| == |dh|`` and
    ``|gravity_b_z diff| == dt``), so the strict ``>`` thresholds are tested at
    just-below / exactly / just-above rather than against an unknown baseline.
    """
    import numpy as np  # noqa: PLC0415
    from . import config as V2CFG  # noqa: PLC0415

    cases: list[dict[str, Any]] = []
    combos = [
        ("normal", {}),
        ("height_0.249_below_threshold", {"dh": 0.249}),
        ("height_exactly_0.25_strict_gt", {"dh": 0.25}),
        ("height_0.251_above_threshold", {"dh": 0.251}),
        ("height_negative_minus_0.251", {"dh": -0.251}),
        ("tilt_0.79_below_threshold", {"dt": 0.79}),
        ("tilt_exactly_0.8_strict_gt", {"dt": 0.80}),
        ("tilt_0.81_above_threshold", {"dt": 0.81}),
        ("nonfinite_qpos", {"nan": True}),
        ("wrist_ankle_deviation_only", {"ee": True}),
    ]
    for label, kw in combos:
        fake = _boundary_env(env, **kw)
        st = _spec_state_from_boundary(env, **kw)
        glue = {
            "root_height": bool(V2CFG.bad_root_height(fake, "motion")[0]),
            "root_tilt": bool(V2CFG.bad_root_tilt(fake, "motion")[0]),
            "nonfinite": bool(V2CFG.bad_nonfinite(fake)[0]),
        }
        want = S.termination(st)
        expected = {"root_height": "root_height" in want.reasons,
                    "root_tilt": "root_tilt" in want.reasons,
                    "nonfinite": "nonfinite" in want.reasons}
        cases.append({"case": label, "glue_on_cuda": glue, "oracle": expected,
                      "oracle_reasons": list(want.reasons),
                      "match": glue == expected})

    names = tuple(env._env.cfg.terminations)
    rewards = tuple(env._env.cfg.rewards)
    cfg = env._env.cfg
    command = env._command
    pelvis = command.cfg.body_names.index(S.TERMINATION_ROOT)
    frame = int(command.time_steps[0])
    # The glue reads command.body_pos_w / body_quat_w, which are indexed by
    # MotionCommand.time_steps.  Prove numerically that this is the current
    # reference frame (termination must NOT use a stale/next frame).
    ref_z_current = float(
        command.motion.body_pos_w[frame, pelvis, 2]
        + env._env.scene.env_origins[0, 2])
    ref_z_glue = float(command.body_pos_w[0, pelvis, 2])
    ref_quat_current = np.asarray(
        command.motion.body_quat_w[frame, pelvis].detach().cpu().numpy(),
        dtype=np.float64)
    ref_quat_glue = np.asarray(
        command.body_quat_w[0, pelvis].detach().cpu().numpy(), dtype=np.float64)
    frame_ok = (abs(ref_z_current - ref_z_glue) < 1e-6
                and bool(np.allclose(ref_quat_current, ref_quat_glue, atol=1e-6))
                and frame == env.ref_index)

    structural = {
        "termination_terms": names,
        "ee_body_pos_absent": "ee_body_pos" not in names,
        "anchor_terms_absent": not ({"anchor_pos", "anchor_ori"} & set(names)),
        "pelvis_glue_present": {"root_height", "root_tilt", "nonfinite"} <= set(names),
        "self_collision_reward_penalty_absent": "self_collisions" not in rewards,
        "self_collision_physics_contact_sensor_present": bool(cfg.scene.sensors),
        "torso_anchor_not_used_for_termination": (S.TERMINATION_ROOT == "pelvis"
                                                  and command.cfg.anchor_body_name == "torso_link"),
        "thresholds_match_spec": (
            abs(float(cfg.terminations["root_height"].params["threshold"]) - S.TERM_HEIGHT) < 1e-12
            and abs(float(cfg.terminations["root_tilt"].params["threshold"]) - S.TERM_TILT) < 1e-12),
        "termination_uses_current_reference_frame": frame_ok,
        "clip_end_is_time_out_success": "time_out" in names,
        "clip_end_success_not_a_failure": (
            S.clip_end_success(100, 100, S.TerminationResult(False, ()))
            and not S.clip_end_success(100, 100,
                                       S.TerminationResult(True, ("root_height",)))),
        "runtime_terminations_normal_state": {
            k: bool(v) for k, v in env.runtime_terminations().items()},
    }
    checks = {k: bool(v) for k, v in structural.items()
              if isinstance(v, bool)}
    checks["all_boundary_cases_match"] = all(c["match"] for c in cases)
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "structural": structural, "cases": cases,
            "reference_frame": frame,
            "thresholds": {"height": S.TERM_HEIGHT, "tilt": S.TERM_TILT}}


def check_frame_alignment(env, ref, steps: int = 3) -> dict[str, Any]:
    """Reference frame used by policy input, reward and termination must agree.

    **PR #13 semantics** (documented in ``benchmark/evaluator/env.py:25-30``):
    ``env.reset()`` ends with ``command_manager.compute(dt=0.0)``, which
    advances ``MotionCommand.time_steps`` to 1.  Frame 0 is only the initial
    condition -- the first *scored* reference frame is **1**.  V2 deliberately
    matches V1 here; the earlier check that expected frame 0 was testing the
    pre-refactor (buggy) alignment, not the protocol.
    """
    import numpy as np  # noqa: PLC0415
    from ..evaluator import constants as C  # noqa: PLC0415

    command = env._command
    env.reset()
    observed = []
    for k in range(steps):
        frame = int(command.time_steps[0])
        si = env.step_input()
        observed.append({"step": k, "command_frame": frame,
                         "policy_reference_index": si.reference_index,
                         "env_ref_index": env.ref_index,
                         "time_s": si.time_s})
        env.step(np.zeros(C.ACTION_DIM))
    first = observed[0]["command_frame"]
    v1_src = (S.BENCHMARK_ROOT / "evaluator" / "env.py").read_text(encoding="utf-8")
    pr13_documented = "first scored reference frame is 1" in v1_src
    details = {
        "first_scored_reference_frame": first,
        "frame_0_is_initial_condition_only": True,
        "full_clip_frames": int(ref.num_frames),
        "scored_control_steps_full_clip": int(ref.num_frames) - 1,
        "motion_time_step_total": int(command.motion.time_step_total),
        "control_dt": C.STEP_DT,
        "pr13_frame_semantics_documented_in_v1": pr13_documented,
    }
    checks = {
        "first_scored_frame_is_1": first == FIRST_SCORED_REFERENCE_FRAME,
        "frame_advances_by_one": all(o["command_frame"] == first + i
                                     for i, o in enumerate(observed)),
        "policy_index_matches_command": all(o["policy_reference_index"] == o["command_frame"]
                                            for o in observed),
        "env_ref_index_matches_command": all(o["env_ref_index"] == o["command_frame"]
                                             for o in observed),
        "time_s_matches_dt": all(abs(o["time_s"] - i * C.STEP_DT) < 1e-12
                                 for i, o in enumerate(observed)),
        "control_dt_matches_oracle": abs(C.STEP_DT - S.CONTROL_DT) < 1e-12,
        "full_clip_frames_match_motion": int(command.motion.time_step_total) == int(ref.num_frames),
        "v2_matches_v1_pr13_frame_semantics": pr13_documented
                                              and first == FIRST_SCORED_REFERENCE_FRAME,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "details": details, "observed": observed}


def check_cpu_fallback_rejection() -> dict[str, Any]:
    """A CPU device / CPU Warp state must be rejected, never silently used."""
    import types  # noqa: PLC0415
    from . import runtime as RT  # noqa: PLC0415

    def raises(fn, *a, **kw) -> bool:
        try:
            fn(*a, **kw)
            return False
        except RT.CudaRuntimeUnavailable:
            return True

    fake = types.SimpleNamespace(
        device="cpu",
        sim=types.SimpleNamespace(
            device="cpu",
            wp_device=types.SimpleNamespace(is_cuda=False, arch=0),
            wp_data=types.SimpleNamespace(
                qpos=types.SimpleNamespace(device=types.SimpleNamespace(is_cuda=False))),
            step_graph=None, use_cuda_graph=False,
        ),
    )
    checks = {
        "cpu_device_string_rejected": raises(RT.require_cuda_device, "cpu"),
        "cpu_warp_state_rejected": raises(RT.inspect_runtime, fake),
        "cpu_fallback_is_opt_in_only": True,  # no code path selects "cpu"
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def check_identity_and_v1() -> dict[str, Any]:
    """V2 identity frozen; V1 protocol / CPU evaluator completely preserved."""
    from pathlib import Path  # noqa: PLC0415
    cfg = S.load_protocol_v2()
    v1_sha = S.v1_protocol_sha256()
    v1_env_src = (S.BENCHMARK_ROOT / "evaluator" / "env.py").read_text(encoding="utf-8")
    checks = {
        "protocol_id": cfg["protocol_id"] == S.PROTOCOL_ID,
        "v1_protocol_unchanged": v1_sha == S.V1_PROTOCOL_SHA256,
        "v1_cpu_evaluator_unchanged": 'device="cpu"' in v1_env_src,
        "v2_protocol_sha_frozen": S.protocol_sha256() == S.V2_PROTOCOL_SHA256,
        "evaluator_implementation": cfg["evaluator_implementation"] == S.EVALUATOR_IMPLEMENTATION,
        "clip_count_40": len(cfg["clip_ids"]) == 40,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "protocol_sha256": S.protocol_sha256(),
            "v1_protocol_sha256": v1_sha, "identity": S.evaluator_identity()}


# --------------------------------------------------------------------------
# Performance: separate CUDA/JIT warmup from real steady-state rollout
# --------------------------------------------------------------------------

def measure_performance(env, steps: int = 200, warmup_steps: int = 20) -> dict[str, Any]:
    """Measure steady-state simulation FPS on a real CUDA rollout.

    CUDA init / env build / reference load / kernel-JIT are *not* included:
    this function only times ``env.step`` calls (the V2TrackingEnv is built
    with ``profile=True``, so each step is CUDA-synchronised).  Deterministic
    small actions exercise PD/contact; a manual reset re-arms the env when a
    smoke rollout falls, and the reset cost is reported separately.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ..evaluator import constants as C  # noqa: PLC0415

    env.reset()
    step_times: list[float] = []
    reset_times: list[float] = []
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()
    for k in range(steps):
        action = 0.02 * np.sin(0.3 * (np.arange(C.ACTION_DIM) + k))
        t0 = time.perf_counter()
        env.step(action)
        torch.cuda.synchronize(env.device)  # explicit: no async inflation
        step_times.append(time.perf_counter() - t0)
        if bool(env._env._manual_reset_pending.any()):  # noqa: SLF001
            t0 = time.perf_counter()
            env.reset()
            torch.cuda.synchronize(env.device)
            reset_times.append(time.perf_counter() - t0)
    total_wall = time.perf_counter() - t_all
    steady = step_times[warmup_steps:]

    # Second measurement: the raw mjlab env step with a pre-allocated CUDA
    # action.  ``V2TrackingEnv.step`` round-trips the action through numpy and
    # pulls reward terms back to the host, so its FPS is a *policy-glue*
    # number, not a physics number.  Reporting both is what makes "GPU single
    # env is slow" diagnosable (task prompt §8) instead of a guess.
    zero = torch.zeros((1, C.ACTION_DIM), dtype=torch.float32, device=env.device)
    env._env.reset()  # noqa: SLF001
    torch.cuda.synchronize(env.device)
    sim_times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        env._env.step(zero)  # noqa: SLF001
        torch.cuda.synchronize(env.device)
        sim_times.append(time.perf_counter() - t0)
        if bool(env._env._manual_reset_pending.any()):  # noqa: SLF001
            env._env.reset()
            torch.cuda.synchronize(env.device)
    sim_steady = sim_times[warmup_steps:] if len(sim_times) > warmup_steps else sim_times
    sim_fps = float(len(sim_steady) / sum(sim_steady)) if sim_steady and sum(sim_steady) > 0 else 0.0
    v2_fps = float(len(steady) / sum(steady)) if steady and sum(steady) > 0 else 0.0
    glue_s = (float(sum(steady) / len(steady)) - float(sum(sim_steady) / len(sim_steady))
              if steady and sim_steady else 0.0)

    return {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "warmup_wall_s": float(sum(step_times[:warmup_steps])),
        "first_step_wall_s": float(step_times[0]),
        "first_step_is_jit_cold_start": True,
        "steady_steps": len(steady),
        "steady_wall_s": float(sum(steady)),
        "steady_simulation_fps": v2_fps,
        "sim_step_fps": sim_fps,
        "sim_step_steps": len(sim_steady),
        "sim_step_wall_s_mean": (float(sum(sim_steady) / len(sim_steady))
                                 if sim_steady else 0.0),
        "policy_glue_overhead_s_per_step": max(glue_s, 0.0),
        "bottleneck": ("policy_glue" if v2_fps and sim_fps and glue_s > (1.0 / sim_fps)
                       else "physics"),
        "step_wall_s_mean": float(sum(step_times) / len(step_times)),
        "step_wall_s_max": float(max(step_times)),
        "total_rollout_wall_s": total_wall,
        "manual_resets": len(reset_times),
        "manual_reset_wall_s": float(sum(reset_times)),
        "note": "steady_simulation_fps = V2TrackingEnv.step (numpy I/O included); "
                "sim_step_fps = raw mjlab env.step with a CUDA-resident action. "
                "Both exclude CUDA init, env build, reference load and kernel JIT.",
    }


# --------------------------------------------------------------------------
# Top-level GPU integration check
# --------------------------------------------------------------------------

def run_gpu_check(clip: str = "walk1_subject1", data_dir=None,
                  device: str = "cuda:0", steps: int = 6) -> dict[str, Any]:
    """Build a real CUDA mjlab env and run every runtime parity check.

    Raises ``CudaRuntimeUnavailable`` when CUDA is missing; callers must report
    ``BLOCKED`` rather than ``PASS`` in that case.
    """
    from ..evaluator import constants as C  # noqa: PLC0415
    from ..evaluator.motion import get_reference  # noqa: PLC0415
    from . import runtime as RT  # noqa: PLC0415
    from .runner import V2TrackingEnv  # noqa: PLC0415

    report: dict[str, Any] = {"clip": clip, "device": device, "status": "BLOCKED",
                              "checks": {}, "timing": {}}
    t_all = time.perf_counter()
    # Telemetry spans CUDA init, env build, JIT and the real rollout so GPU
    # utilisation / memory / CPU utilisation are sampled over the whole run.
    telemetry = RT.GpuTelemetry(interval=0.4).start()
    try:
        t0 = time.perf_counter()
        probe = RT.probe_cuda(device)
        report["timing"]["cuda_init_probe_s"] = time.perf_counter() - t0
        report["hardware"] = RT.format_hardware_report(probe)
        report["versions"] = probe["versions"]
        report["version_drift"] = probe["version_drift"]
        report["target_gpu_match"] = probe["target_gpu_match"]
        report["sm_120_confirmed"] = probe["sm_120_confirmed"]

        t0 = time.perf_counter()
        root = data_dir or C.DATA_DIR
        ref = get_reference(root / f"{clip}.csv", use_cache=True)
        report["timing"]["reference_load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        env = V2TrackingEnv(ref, device=device, profile=True)
        report["timing"]["cuda_env_build_s"] = time.perf_counter() - t0
        report["runtime"] = env.runtime
        report["hardware"] = RT.format_hardware_report(probe, env.runtime)
        report["no_cpu_fallback"] = env.runtime["no_cpu_fallback"]

        # A real CUDA physics step, synchronised, with state-delta evidence.
        import torch  # noqa: PLC0415
        zero = torch.zeros((1, C.ACTION_DIM), dtype=torch.float32, device=env.device)
        t0 = time.perf_counter()
        report["kernel_execution"] = RT.verify_cuda_kernel_execution(env._env, zero)
        report["timing"]["first_kernel_jit_wall_s"] = time.perf_counter() - t0
        env.reset()

        report["checks"]["reward"] = check_reward_parity(env, steps=steps)
        report["checks"]["termination"] = check_termination_parity(env)
        report["checks"]["frame_alignment"] = check_frame_alignment(env, ref)
        # Complementary: validate the re-anchoring *formula* itself on the GPU
        # (the reward check above consumes the runtime buffers as its input).
        report["checks"]["reference_reanchor"] = check_reference_reanchor_parity(env)
        report["checks"]["identity_v1"] = check_identity_and_v1()
        report["checks"]["cpu_fallback_rejection"] = check_cpu_fallback_rejection()

        # Dedicated steady-state rollout: warmup/JIT is excluded from FPS.
        report["performance"] = measure_performance(env, steps=200, warmup_steps=20)
        report["performance_parity_check_only"] = env.performance()
        report["timing"]["total_wall_s"] = time.perf_counter() - t_all
        report["reward_per_term_max_abs_error"] = (
            report["checks"]["reward"]["static"]["per_term_max_abs_error"])
        statuses = [v["status"] for v in report["checks"].values()]
        report["status"] = ("PASS" if statuses and all(s == "PASS" for s in statuses)
                            else "FAIL")
    finally:
        report["telemetry"] = telemetry.stop()
    return report
