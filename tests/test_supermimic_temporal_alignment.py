"""Temporal alignment of the SuperMimic adapter's observations and goals.

What this file is for
---------------------
``mjlab``'s ``MotionCommand`` initializes the robot at reference frame 0 on
``reset()`` and then calls ``_update_command(dt=0)`` which advances
``time_steps`` to 1. Consequently, at every control step:

* robot state (``q``, ``qd``, ``base_ang_vel``, ``previous_action``) and
  ``anchor_ori_b_prev`` belong to frame ``f = step.reference_index - 1``;
* ``step.reference_index`` and ``step.anchor_ori_b`` belong to ``f + 1``.

Earlier revisions of ``SuperMimicPolicy`` (and ``BatchedSuperMimicPolicy``)
passed ``step.reference_index`` and ``step.anchor_ori_b`` directly to
``observation()`` and ``goal()``, shifting ``q_ref(f)``, ``g_ref(f)``,
``anchor_ori_b``, and the entire 10-frame goal horizon ``f+1 .. f+10`` ahead by
one frame at every control step.

These tests pin both algebraically without requiring MuJoCo or CUDA, and
include a PyTorch + upstream ``supermimic_vibed`` cross-check that runs whenever
``torch`` and ``.tmp/v2_upstream/supermimic_vibed`` (or ``SUPERMIMIC_UPSTREAM``)
are available.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import mathx  # noqa: E402
from benchmark.evaluator.policy_api import (  # noqa: E402
    ClipContext,
    ReferenceClip,
    StepInput,
)
from benchmark.policies import supermimic as sm  # noqa: E402

DT = C.STEP_DT
T = 40
G_WORLD = np.array([0.0, 0.0, -1.0])
TOL = 1e-12


def _rot(axis, angle: float) -> np.ndarray:
    return sm.axis_rotation(tuple(float(a) for a in axis), float(angle))


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    return sm._quat_from_matrix(m)  # noqa: SLF001


class SyntheticClip:
    """A T-frame clip whose root + waist rotate cleanly between consecutive frames."""

    def __init__(self, torso_rate: float = 2.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.torso_rate = float(torso_rate)
        t = np.arange(T) * DT
        self.root_rot = np.stack(
            [
                _rot((0, 0, 1), 0.4 * t[i])
                @ _rot((1, 0, 0), self.torso_rate * t[i])
                @ _rot((0, 1, 0), 0.05 * np.sin(6.0 * t[i]))
                for i in range(T)
            ]
        )
        self.root_pos = np.stack(
            [0.3 * t, 0.02 * np.sin(3 * t), 0.75 + 0.01 * np.sin(9 * t)], axis=1
        )
        self.joint = sm.default_joint_pos_reference()[None, :] + 0.25 * np.sin(
            np.outer(t, np.linspace(1.0, 3.0, sm.NUM_J)) + rng.uniform(0, 1, sm.NUM_J)
        )
        self.joint_vel = np.gradient(self.joint, DT, axis=0)
        self.ref = ReferenceClip(
            num_frames=T,
            control_dt=DT,
            joint_pos=self.joint,
            joint_vel=self.joint_vel,
            root_pos_w=self.root_pos,
            root_quat_w=np.stack([_quat_from_mat(m) for m in self.root_rot]),
        )

    def A(self, frame: int) -> np.ndarray:
        return sm.waist_rotation(self.joint[max(frame, 0)])

    def ref_torso_rot(self, frame: int) -> np.ndarray:
        return self.root_rot[frame] @ self.A(frame)

    def torso_step_rad(self) -> float:
        return float(
            mathx.quat_error_magnitude(
                _quat_from_mat(self.ref_torso_rot(5)),
                _quat_from_mat(self.ref_torso_rot(6)),
            )
        )


def _anchor6(clip: SyntheticClip, robot_rot: np.ndarray, q: np.ndarray, frame: int) -> np.ndarray:
    R_robot_torso = robot_rot @ sm.waist_rotation(q)
    m = R_robot_torso.T @ clip.ref_torso_rot(frame)
    return sm.mat_to_6d(m)


def _tiny_policy() -> sm.SuperMimicPolicy:
    rng = np.random.default_rng(7)
    t: dict[str, np.ndarray] = {}
    enc_dims = [(730, 32), (32, 32), (32, 32), (32, 32), (32, 32)]
    dec_dims = [(32 + 10 * 99, 32), (32, 32), (32, 32), (32, 32), (32, 29)]
    for idx, (nin, nout) in zip((0, 2, 4, 6, 8), enc_dims):
        t[f"pi.enc.{idx}.weight"] = rng.normal(0, 0.05, (nout, nin)).astype(np.float32)
        t[f"pi.enc.{idx}.bias"] = np.zeros(nout, dtype=np.float32)
    for idx, (nin, nout) in zip((0, 2, 4, 6, 8), dec_dims):
        t[f"pi.dec.{idx}.weight"] = rng.normal(0, 0.05, (nout, nin)).astype(np.float32)
        t[f"pi.dec.{idx}.bias"] = np.zeros(nout, dtype=np.float32)
    t["pi.obs_norm.mean"] = np.zeros(99, dtype=np.float32)
    t["pi.obs_norm.var"] = np.ones(99, dtype=np.float32)
    t["pi.goal_norm.mean"] = np.zeros(730, dtype=np.float32)
    t["pi.goal_norm.var"] = np.ones(730, dtype=np.float32)
    return sm.SuperMimicPolicy(t)


class Rig:
    def __init__(self, clip: SyntheticClip, robot_offset_rad: float = 0.25):
        self.clip = clip
        self.policy = _tiny_policy()
        self.q0 = sm.default_joint_pos_reference()
        self.scale = sm.action_scale_reference()
        self.robot_rot = np.stack(
            [_rot((0, 1, 0), robot_offset_rad) @ R for R in clip.root_rot]
        )
        self.policy.reset(
            ClipContext(
                clip="synthetic",
                joint_order=C.JOINT_ORDER,
                default_joint_pos=self.q0,
                action_scale=self.scale,
                control_dt=DT,
                reference=clip.ref,
            )
        )

    def true_gravity(self, frame: int) -> np.ndarray:
        return self.robot_rot[frame].T @ G_WORLD

    def true_pelvis_rel_6d(self, frame: int) -> np.ndarray:
        return sm.mat_to_6d(self.robot_rot[frame].T @ self.clip.root_rot[frame])

    def step(self, frame: int, *, q: np.ndarray | None = None, previous=None) -> StepInput:
        clip = self.clip
        q = clip.joint[frame] if q is None else q
        return StepInput(
            step=frame,
            reference_index=frame + 1,
            time_s=frame * DT,
            joint_pos=q,
            joint_vel=clip.joint_vel[frame],
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.array([0.1, -0.2, 0.3]),
            anchor_pos_b=np.zeros(3),
            anchor_ori_b=_anchor6(clip, self.robot_rot[frame], q, min(frame + 1, T - 1)),
            previous_action=np.zeros(sm.NUM_J) if previous is None else previous,
            reference=clip.ref,
            anchor_pos_b_prev=np.zeros(3),
            anchor_ori_b_prev=_anchor6(clip, self.robot_rot[frame], q, frame),
        )


class ObservationAndGoalFrameTest(unittest.TestCase):
    """SuperMimic must read frame ``f = reference_index - 1`` and ``anchor_ori_b_prev``."""

    def setUp(self):
        self.clip = SyntheticClip(torso_rate=6.0)
        self.rig = Rig(self.clip)

    def test_observation_uses_frame_f_not_f_plus_1(self):
        for frame in (0, 1, 7, 20, 35):
            step = self.rig.step(frame)
            obs = self.rig.policy.observation(step)
            np.testing.assert_allclose(
                obs[61:64], self.rig.true_gravity(frame), atol=TOL, err_msg=f"frame {frame}"
            )
            np.testing.assert_allclose(
                obs[93:99], self.rig.true_pelvis_rel_6d(frame), atol=TOL, err_msg=f"frame {frame}"
            )

    def test_swapping_anchor_fields_breaks_gravity_and_pelvis_rel(self):
        frame = 10
        step = self.rig.step(frame)
        swapped = StepInput(
            **{
                **step.__dict__,
                "anchor_ori_b_prev": step.anchor_ori_b,
                "anchor_ori_b": step.anchor_ori_b_prev,
            }
        )
        obs = self.rig.policy.observation(swapped)
        grav_err = float(np.abs(obs[61:64] - self.rig.true_gravity(frame)).max())
        rel_err = float(np.abs(obs[93:99] - self.rig.true_pelvis_rel_6d(frame)).max())
        self.assertGreater(grav_err, 0.05)
        self.assertGreater(rel_err, 0.05)

    def test_error_scales_monotonically_with_torso_angular_velocity(self):
        errors = []
        for rate in (0.5, 2.0, 6.0, 12.0):
            clip = SyntheticClip(torso_rate=rate)
            rig = Rig(clip)
            frame = 10
            step = rig.step(frame)
            buggy = StepInput(**{**step.__dict__, "anchor_ori_b_prev": step.anchor_ori_b})
            obs = rig.policy.observation(buggy)
            err = float(np.abs(obs[61:64] - rig.true_gravity(frame)).max())
            theta = clip.torso_step_rad()
            self.assertLessEqual(err, 2.0 * np.sin(theta / 2.0) + 1e-9)
            self.assertGreater(err, 0.0)
            errors.append(err)
        self.assertEqual(errors, sorted(errors))
        self.assertGreater(errors[-1], 0.05)

    def test_act_forwards_goal_at_frame_f_not_f_plus_1(self):
        recorded: list[tuple[np.ndarray, np.ndarray]] = []
        orig_forward = self.rig.policy.forward

        def spy_forward(obs_hist, goal):
            recorded.append((obs_hist.copy(), goal.copy()))
            return orig_forward(obs_hist, goal)

        self.rig.policy.forward = spy_forward
        for frame in (0, 1, 10, T - 3):
            step = self.rig.step(frame)
            self.rig.policy.act(step)
            _, sent_goal = recorded[-1]
            expected_goal = self.rig.policy.goal(frame)
            shifted_goal = self.rig.policy.goal(frame + 1)
            np.testing.assert_allclose(sent_goal, expected_goal, atol=TOL)
            self.assertGreater(float(np.abs(sent_goal - shifted_goal).max()), 1e-2)

    def test_missing_prev_anchor_fails_closed(self):
        step = self.rig.step(3)
        stripped = types.SimpleNamespace(
            **{k: v for k, v in step.__dict__.items() if k != "anchor_ori_b_prev"}
        )
        with self.assertRaises(sm.SuperMimicContractError):
            self.rig.policy.act(stripped)

    def test_nonpositive_reference_index_fails_closed(self):
        step = self.rig.step(0)
        bad = StepInput(**{**step.__dict__, "reference_index": 0})
        with self.assertRaises(sm.SuperMimicContractError):
            self.rig.policy.act(bad)


def _upstream_root() -> Path | None:
    env = os.environ.get("SUPERMIMIC_UPSTREAM")
    candidates = [
        Path(env) if env else None,
        REPO_ROOT / ".tmp/v2_upstream/supermimic_vibed",
        Path("/tmp/supermimic_vibed"),
    ]
    for cand in candidates:
        if cand is not None and (cand / "supermimic/mdp.py").exists():
            return cand
    return None


def _torch_and_upstream_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return _upstream_root() is not None


@unittest.skipUnless(_torch_and_upstream_available(), "torch or upstream supermimic_vibed not present")
class UpstreamSourceParityTest(unittest.TestCase):
    """Direct cross-check against upstream ``supermimic/mdp.py`` & ``networks.py``."""

    def test_mdp_obs_and_policy_forward_match_upstream(self):
        import torch  # noqa: PLC0415

        upstream = _upstream_root()
        assert upstream is not None
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))

        # Stub G1Kinematics cleanly (no IsaacLab needed for MDP.obs / Policy)
        if "supermimic.kinematics" not in sys.modules:
            kin_mod = types.ModuleType("supermimic.kinematics")

            class _StubKin:
                q_lo = torch.zeros(sm.NUM_J)
                q_hi = torch.zeros(sm.NUM_J)

                def forward(self, p, R, v, w, q, qd):
                    z = torch.zeros_like(p)
                    return z, R, z, z

            kin_mod.G1Kinematics = _StubKin
            sys.modules["supermimic.kinematics"] = kin_mod

        import supermimic.mdp as up_mdp  # noqa: E402, PLC0415
        import supermimic.networks as up_net  # noqa: E402, PLC0415
        import supermimic.rotations as up_rot  # noqa: E402, PLC0415

        clip = SyntheticClip(torso_rate=4.0, seed=1)
        rig = Rig(clip, robot_offset_rad=0.35)

        motion_stub = types.SimpleNamespace(
            root_p=torch.from_numpy(clip.root_pos).to(torch.float32),
            root_R=torch.from_numpy(clip.root_rot).to(torch.float32),
            q=torch.from_numpy(clip.joint).to(torch.float32),
        )
        kin = sys.modules["supermimic.kinematics"].G1Kinematics()
        mdp = up_mdp.MDP(kin, motion_stub, torch.from_numpy(rig.q0).to(torch.float32))

        pi = up_net.Policy(sm.OBS_DIM, sm.GOAL_FRAMES * sm.GOAL_PER_FRAME, hist=sm.HIST)
        with torch.no_grad():
            pi.obs_norm.mean.normal_(0.0, 0.1)
            pi.obs_norm.var.uniform_(0.5, 2.0)
            pi.goal_norm.mean.normal_(0.0, 0.1)
            pi.goal_norm.var.uniform_(0.5, 2.0)
        pi.eval()
        up_net.AMP_DTYPE = None

        tensors = {f"pi.{k}": v.detach().cpu().numpy() for k, v in pi.state_dict().items()}
        policy_from_up = sm.SuperMimicPolicy(tensors)
        policy_from_up.reset(
            ClipContext(
                clip="synthetic",
                joint_order=C.JOINT_ORDER,
                default_joint_pos=rig.q0,
                action_scale=rig.scale,
                control_dt=DT,
                reference=clip.ref,
            )
        )

        prev_a = np.zeros(sm.NUM_J, dtype=np.float64)
        hist_t = None
        for frame in range(0, 12):
            # Drifted state at frame f: robot orientation + disturbed waist joints
            rng = np.random.default_rng(100 + frame)
            q_robot = clip.joint[frame] + rng.normal(0, 0.04, sm.NUM_J)
            qd_robot = clip.joint_vel[frame] + rng.normal(0, 0.1, sm.NUM_J)
            w_body = np.array([0.1, -0.2, 0.3])
            R_robot = rig.robot_rot[frame]
            w_world = R_robot @ w_body

            step = rig.step(frame, q=q_robot, previous=prev_a)
            step = StepInput(**{**step.__dict__, "joint_vel": qd_robot})

            # Upstream MDP.obs at env.f == frame
            S = (
                torch.from_numpy(clip.root_pos[frame]).unsqueeze(0).to(torch.float32),
                torch.from_numpy(R_robot).unsqueeze(0).to(torch.float32),
                torch.zeros((1, 3), dtype=torch.float32),
                torch.from_numpy(w_world).unsqueeze(0).to(torch.float32),
                torch.from_numpy(q_robot).unsqueeze(0).to(torch.float32),
                torch.from_numpy(qd_robot).unsqueeze(0).to(torch.float32),
            )
            xi = torch.zeros((1, 33), dtype=torch.float32)
            o_up = mdp.obs(
                S,
                torch.from_numpy(prev_a).unsqueeze(0).to(torch.float32),
                xi,
                torch.tensor([frame], dtype=torch.long),
            )
            obs_adapter = policy_from_up.observation(step)
            np.testing.assert_allclose(obs_adapter, o_up[0].numpy(), atol=1e-5)

            # Upstream MotionLib.goal formula at env.f == frame
            hist_t = (
                o_up.unsqueeze(1).repeat(1, sm.HIST, 1)
                if hist_t is None
                else torch.cat((hist_t[:, 1:], o_up.unsqueeze(1)), 1)
            )
            goal_adapter = policy_from_up.goal(frame)
            with torch.no_grad():
                a_up = pi(hist_t, torch.from_numpy(goal_adapter).unsqueeze(0).to(torch.float32))[0].numpy()
            a_adapter = policy_from_up.act(step)
            np.testing.assert_allclose(a_adapter, a_up, atol=1e-5)
            prev_a = a_adapter
            # Ensure reference yaw_mat parity too
            np.testing.assert_allclose(
                sm.yaw_matrix(clip.root_rot[frame]),
                up_rot.yaw_mat(motion_stub.root_R[frame]).numpy(),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
