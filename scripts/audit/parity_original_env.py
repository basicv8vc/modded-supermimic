#!/usr/bin/env python3
"""Observation parity against the original NablaMimic environment, on real clips.

Ground truth is the upstream training-time environment itself
(``sailab-ai/NablaMimic``, ``src/nablamimic/env.py``): its own ``proprio``,
``goal``, ``_anchor_rel`` and ``_goal_features``, driven by MuJoCo forward
kinematics over the real LAFAN1 G1 reference trajectories. The canonical
adapter is fed a ``StepInput`` built from the same robot state and the same
reference clip, and the two observation vectors are compared channel by
channel.

This is the test the synthetic unit tests cannot be: the reference features come
from the original ``motion.py`` resampler on real CSVs, the torso poses come from
the real kinematic tree, and the robot state is a *drifted* state -- the case a
mid-rollout evaluator actually sees. The bug this exists to catch is invisible
when the robot state equals the reference state, because then the anchor
rotation is the identity.

The actor weights are irrelevant here (only observation construction is under
test), so the policy is instantiated with correctly *shaped* stand-in weights.

Usage:
    NABLAMIMIC_ROOT=~/Downloads/NablaMimic LAFAN1_G1_DIR=/tmp/lafan1_g1/g1 \
    /tmp/venv-nabla/bin/python -m scripts.audit.parity_original_env \
        --clips walk3_subject4 dance1_subject1 run1_subject2 fallAndGetUp2_subject3 \
        --out reports/audit/parity_original_env.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("NABLAMIMIC_ROOT", Path.home() / "Downloads/NablaMimic"))


def _menagerie_dir() -> Path:
    """The original repo expects its own ``third_party`` checkout; fall back to audit/upstream."""
    candidates = (
        UPSTREAM / "third_party/mujoco_menagerie",
        UPSTREAM / "audit/upstream/menagerie",
    )
    for cand in candidates:
        if (cand / "unitree_g1/scene_mjx.xml").exists():
            return cand
    raise SystemExit(f"no Menagerie checkout under {UPSTREAM}")


def _lafan1_dir() -> Path:
    return Path(os.environ.get("LAFAN1_G1_DIR", UPSTREAM / "data/lafan1_g1"))


os.environ["MENAGERIE_DIR"] = str(_menagerie_dir())
os.environ["LAFAN1_G1_DIR"] = str(_lafan1_dir())
os.environ.setdefault("MOTION_CACHE_DIR", "/tmp/nablamimic_cache")
sys.path.insert(0, str(UPSTREAM / "src"))
sys.path.insert(0, str(REPO))

import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402

from benchmark.evaluator.policy_api import (  # noqa: E402
    ClipContext,
    ReferenceClip,
    StepInput,
)
from benchmark.policies import nablamimic as adapter  # noqa: E402
from nablamimic import env as nenv  # noqa: E402
from nablamimic import motion as nmotion  # noqa: E402

# Observation channel map, matching env.proprio's concatenation order.
CHANNELS = (
    ("joint_pos", slice(0, 29)),
    ("joint_vel", slice(29, 58)),
    ("base_ang_vel", slice(58, 61)),
    ("gravity_b", slice(61, 64)),
    ("previous_action", slice(64, 93)),
)


# --------------------------------------------------------------------------- rig
class UpstreamRig:
    """One clip: the original env, its model, and the adapter's view of the same data."""

    def __init__(self, clip: str, seed: int = 0):
        self.clip = clip
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(nmotion.SCENE))
        # build_model applies the PD gains / solver settings the env uses; the
        # kinematics we need are unaffected, but use it so q0 / action_scale match.
        self.cfg = nenv.EnvConfig(
            dr=False, obs_noise=False, reset_noise=False,
            residual_action=True, action_rate_space="setpoint",
        )
        self.lib = nmotion.load_library([clip], model=self.model, bodies=nenv.BM_BODIES)
        self.env = nenv.TrackingEnv(self.lib, self.cfg)
        self.data = mujoco.MjData(self.model)

        self.nu = int(self.model.nu)
        self.joint_names = tuple(
            self.model.joint(int(self.model.actuator_trnid[i, 0])).name for i in range(self.nu)
        )
        self.q0 = np.asarray(self.env._q0_np, dtype=np.float64)
        # float64, as the protocol publishes it; env.action_scale is the same
        # quantity held as float32 by JAX, which the adapter's 1e-8 check rejects.
        self.action_scale = self._action_scale()

        self.policy = adapter.NablaMimicPolicy(_stand_in_params())
        self.policy.reset(self._context())

    # -- adapter side -------------------------------------------------------
    def _action_scale(self) -> np.ndarray:
        """``0.25 * effort / kp`` in float64, the form the protocol publishes."""
        m = self.model
        kp = np.empty(self.nu, dtype=np.float64)
        effort = np.empty(self.nu, dtype=np.float64)
        for i in range(self.nu):
            jnt = int(m.actuator_trnid[i, 0])
            kp[i] = float(m.dof_armature[m.jnt_dofadr[jnt]]) * float(nenv.OMEGA) ** 2
            effort[i] = float(m.jnt_actfrcrange[jnt, 1])
        return 0.25 * effort / kp

    def _context(self) -> ClipContext:
        qpos = np.asarray(self.lib.qpos, dtype=np.float64)
        qvel = np.asarray(self.lib.qvel, dtype=np.float64)
        ref = ReferenceClip(
            num_frames=int(qpos.shape[0]),
            control_dt=float(nenv.TIMESTEP) * 4,
            joint_pos=qpos[:, 7:],
            joint_vel=qvel[:, 6:],
            root_pos_w=qpos[:, :3],
            root_quat_w=qpos[:, 3:7],
        )
        return ClipContext(
            clip=self.clip,
            joint_order=self.joint_names,
            default_joint_pos=self.q0,
            action_scale=self.action_scale,
            control_dt=float(nenv.TIMESTEP) * 4,
            reference=ref,
        )

    # -- states -------------------------------------------------------------
    def ref_state(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.lib.qpos[frame], dtype=np.float64),
            np.asarray(self.lib.qvel[frame], dtype=np.float64),
        )

    def drifted_state(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """The reference state plus a mid-rollout drift: pose, joints and velocity."""
        qpos, qvel = self.ref_state(frame)
        qpos = qpos.copy()
        qvel = qvel.copy()
        qpos[:3] += self.rng.normal(0.0, 0.05, 3)
        axis = self.rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)
        ang = self.rng.normal(0.0, 0.06)
        dq = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
        qpos[3:7] = _quat_mul(dq, qpos[3:7])
        qpos[7:] += self.rng.normal(0.0, 0.03, self.nu)
        qvel += self.rng.normal(0.0, 0.04, qvel.shape[0])
        return qpos, qvel

    def forward(self, qpos: np.ndarray, qvel: np.ndarray) -> SimpleNamespace:
        d = self.data
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(self.model, d)
        return SimpleNamespace(
            qpos=jnp.asarray(d.qpos), qvel=jnp.asarray(d.qvel),
            xpos=jnp.asarray(d.xpos), xquat=jnp.asarray(d.xquat),
        )

    # -- the two observations ----------------------------------------------
    def truth(self, dx, frame: int, prev_a: np.ndarray):
        xi = nenv.Xi(jnp.float32(1.0), jnp.zeros(self.nu), jnp.zeros(3))
        proprio = np.asarray(self.env.proprio(dx, xi, jnp.asarray(prev_a)), dtype=np.float64)
        state = SimpleNamespace(frame=jnp.int32(frame), dx=dx, prev_a=jnp.asarray(prev_a))
        goal = np.asarray(self.env.goal(state, None), dtype=np.float64)
        anchor = np.asarray(self.env._anchor_rel(dx, frame), dtype=np.float64)
        return proprio, goal, anchor

    def adapter_step(self, qpos, qvel, frame: int, prev_a: np.ndarray, anchor):
        """A StepInput as the evaluator would build it at this step.

        ``anchor_*_b`` carry mjlab's meaning (frame ``reference_index = f + 1``);
        ``anchor_*_prev`` are the state's own frame ``f``. The original emits the
        relative rotation's two *columns* stacked; mjlab flattens the first two
        columns of the matrix row-major, so the payload is interleaved -- the
        same re-ordering ``six_to_mat`` documents.
        """
        anchor_next = self._anchor_at(self.forward(qpos, qvel), min(frame + 1, int(self.lib.last[frame])))
        nxt = _interleave(anchor_next[3:])
        cur = _interleave(anchor[3:])
        step = StepInput(
            step=frame,
            reference_index=frame + 1,
            time_s=frame * 0.02,
            joint_pos=qpos[7:].copy(),
            joint_vel=qvel[6:].copy(),
            base_lin_vel=qvel[:3].copy(),
            base_ang_vel=qvel[3:6].copy(),
            anchor_pos_b=np.asarray(anchor_next[:3], dtype=np.float64),
            anchor_ori_b=nxt,
            anchor_pos_b_prev=np.asarray(anchor[:3], dtype=np.float64),
            anchor_ori_b_prev=cur,
            previous_action=np.asarray(prev_a, dtype=np.float64),
            reference=self.policy._context.reference,
        )
        # ``act`` stores the raw action it issued; ``proprioception`` reads it back
        # from the policy, not from the step.
        self.policy._prev_action_raw = np.asarray(prev_a, dtype=np.float64)
        return step

    def _anchor_at(self, dx, frame: int) -> np.ndarray:
        return np.asarray(self.env._anchor_rel(dx, frame), dtype=np.float64)

    def adapter_obs(self, step, frame: int, prefix: bool = False):
        if prefix:
            # The pre-fix adapter: same field the goal and gravity used, i.e. f + 1.
            saved = adapter._anchor_ori_at_state_frame
            adapter._anchor_ori_at_state_frame = lambda s: np.asarray(s.anchor_ori_b, dtype=np.float64)
            try:
                return (
                    self.policy.proprioception(step, frame),
                    self.policy.goal(frame, step),
                )
            finally:
                adapter._anchor_ori_at_state_frame = saved
        return self.policy.proprioception(step, frame), self.policy.goal(frame, step)


def _interleave(col6: np.ndarray) -> np.ndarray:
    """Columns-stacked 6-D rotation -> mjlab's row-major interleaved payload."""
    return np.asarray(col6, dtype=np.float64)[[0, 3, 1, 4, 2, 5]]


def _stand_in_params() -> dict:
    """Correctly shaped stand-in weights: only observation construction is under test."""
    rng = np.random.default_rng(0)
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.01  # noqa: E731
    return {
        "anchor_dec": True,
        "enc": [(f32(adapter.GOAL_DIM - adapter.ANCHOR_DIM, 32), np.zeros(32, np.float32)),
                (f32(32, 32), np.zeros(32, np.float32))],
        "dec": [(f32(adapter.HIST * adapter.PROPRIO_DIM + 32 + adapter.ANCHOR_DIM, adapter.NUM_J),
                 np.zeros(adapter.NUM_J, np.float32))],
    }


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# --------------------------------------------------------------------------- frames
def pick_frames(rig: UpstreamRig) -> list[tuple[str, int]]:
    """Initial, early, middle, fastest torso rotation, and near-final frames."""
    n = int(rig.lib.qpos.shape[0])
    clip_end = int(rig.lib.last[0])
    q = np.asarray(rig.lib.qpos[:, 3:7])
    ang = np.abs(np.asarray(rig.lib.qvel[:, 3:6])).sum(1)
    interior = max(clip_end // 10, 1)
    fast = int(np.argmax(np.where(np.arange(len(ang)) < clip_end, ang, -1.0)))
    return [
        ("initial", 1),
        ("early", min(interior, clip_end)),
        ("middle", clip_end // 2),
        ("high_ang_vel", fast),
        ("near_final", max(clip_end - 2, 1)),
    ]


def _errors(pred: np.ndarray, truth: np.ndarray, labels) -> dict:
    out = {}
    for name, sl in labels:
        out[name] = float(np.abs(pred[sl] - truth[sl]).max())
    return out


def run_clip(clip: str, seed: int = 0) -> dict:
    rig = UpstreamRig(clip, seed=seed)
    frames = pick_frames(rig)

    # Reference features: the original numpy pipeline vs the adapter's, all frames.
    feat_truth = np.asarray(rig.env.goal_feat, dtype=np.float64)
    feat_adapter = np.asarray(rig.policy._feat, dtype=np.float64)
    feat_err = np.abs(feat_adapter - feat_truth)
    ref = {
        "frames": int(feat_truth.shape[0]),
        "goal_features_max_abs": float(feat_err.max()),
        "goal_features_worst_frame": int(np.unravel_index(feat_err.argmax(), feat_err.shape)[0]),
        "goal_features_worst_frame_max": float(feat_err.max(axis=1).max()),
        "goal_features_median": float(np.median(feat_err)),
    }

    rows = []
    for tag, frame in frames:
        frame = int(min(max(frame, 1), int(rig.lib.last[0]) - 1))
        for mode in ("drifted", "matched"):
            qpos, qvel = (
                rig.drifted_state(frame) if mode == "drifted" else rig.ref_state(frame)
            )
            dx = rig.forward(qpos, qvel)
            prev_a = rig.rng.normal(0.0, 0.4, rig.nu)
            p_true, g_true, anchor = rig.truth(dx, frame, prev_a)
            step = rig.adapter_step(qpos, qvel, frame, prev_a, anchor)
            p_fix, g_fix = rig.adapter_obs(step, frame)
            p_old, g_old = rig.adapter_obs(step, frame, prefix=True)

            rows.append({
                "frame": frame,
                "tag": tag,
                "mode": mode,
                "proprio_fixed": _errors(p_fix, p_true, CHANNELS),
                "proprio_prefix": _errors(p_old, p_true, CHANNELS),
                "goal_features_fixed": float(np.abs(g_fix[:-6] - g_true[:-6]).max()),
                "goal_features_prefix": float(np.abs(g_old[:-6] - g_true[:-6]).max()),
                "goal_anchor_fixed": float(np.abs(g_fix[-6:] - g_true[-6:]).max()),
                "goal_anchor_prefix": float(np.abs(g_old[-6:] - g_true[-6:]).max()),
                "goal_686_fixed": float(np.abs(g_fix - g_true).max()),
                "goal_686_prefix": float(np.abs(g_old - g_true).max()),
                "anchor_truth_vs_prev_field": float(np.abs(anchor[3:] - step.anchor_ori_b_prev).max()),
                "anchor_truth_vs_next_field": float(np.abs(anchor[3:] - step.anchor_ori_b).max()),
            })
    return {"clip": clip, "reference": ref, "samples": rows}


# --------------------------------------------------------------------------- gyro
def gyro_check(clip: str) -> dict:
    """Is the evaluator's ``imu_ang_vel`` the same quantity as the original's ``qvel[3:6]``?"""
    path = REPO / "assets/g1/g1.xml"
    if not path.exists():
        return {"available": False}
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    if sid < 0:
        return {"available": False, "reason": "no imu_ang_vel sensor"}
    adr = int(model.sensor_adr[sid])

    rig = UpstreamRig(clip)
    qpos, qvel = rig.drifted_state(int(rig.lib.last[0]) // 2)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    gyro = np.array(data.sensordata[adr: adr + 3], dtype=np.float64)
    qvel_ang = np.asarray(qvel[3:6], dtype=np.float64)
    return {
        "available": True,
        "gyro": gyro.tolist(),
        "qvel_ang": qvel_ang.tolist(),
        "max_abs_diff": float(np.abs(gyro - qvel_ang).max()),
        "norms": [float(np.linalg.norm(gyro)), float(np.linalg.norm(qvel_ang))],
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "reports/audit/parity_original_env.json")
    args = ap.parse_args(argv)

    available = {p.stem for p in _lafan1_dir().glob("*.csv")}
    clips = args.clips or sorted(available)
    missing = [c for c in clips if c not in available]
    if missing:
        raise SystemExit(f"missing clips: {missing} under {_lafan1_dir()}")

    results = [run_clip(c) for c in clips]
    gyro = gyro_check(clips[0]) if clips else {"available": False}

    def _agg(key):
        vals = [r[key] for res in results for r in res["samples"]]
        return float(np.max(vals)) if vals else float("nan")

    def _agg_channel(which, channel):
        vals = [r[which][channel] for res in results for r in res["samples"]]
        return float(np.max(vals)) if vals else float("nan")

    summary = {
        "clips": len(results),
        "samples": sum(len(r["samples"]) for r in results),
        "reference_goal_features_max_abs": float(
            max(r["reference"]["goal_features_max_abs"] for r in results)
        ),
        "proprio_fixed_max_abs": {c: _agg_channel("proprio_fixed", c) for c, _ in CHANNELS},
        "proprio_prefix_max_abs": {c: _agg_channel("proprio_prefix", c) for c, _ in CHANNELS},
        "goal_686_fixed_max_abs": _agg("goal_686_fixed"),
        "goal_686_prefix_max_abs": _agg("goal_686_prefix"),
        "goal_anchor_fixed_max_abs": _agg("goal_anchor_fixed"),
        "goal_anchor_prefix_max_abs": _agg("goal_anchor_prefix"),
        "gyro": gyro,
    }

    payload = {"summary": summary, "clips": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'channel':22s} {'fixed':>12s} {'pre-fix':>12s}")
    for c, _ in CHANNELS:
        print(f"{c:22s} {summary['proprio_fixed_max_abs'][c]:12.3e} "
              f"{summary['proprio_prefix_max_abs'][c]:12.3e}")
    print(f"{'goal 680 features':22s} {'':>12s} {'':>12s} "
          f"max|err| fixed={_agg("goal_features_fixed"):.3e} "
          f"prefix={_agg("goal_features_prefix"):.3e}")
    print(f"{'goal 686 total':22s} {summary['goal_686_fixed_max_abs']:12.3e} "
          f"{summary['goal_686_prefix_max_abs']:12.3e}")
    print(f"{'reference features':22s} {summary['reference_goal_features_max_abs']:12.3e}")
    if gyro.get("available"):
        print(f"gyro vs qvel[3:6]: max|diff| = {gyro['max_abs_diff']:.3e} "
              f"(norms {gyro['norms'][0]:.4f} vs {gyro['norms'][1]:.4f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
