"""The SuperMimic policy must rebuild its native inputs, not approximate them.

These tests do not need torch: the policy loads ``pi.*`` with the torch-free
reader and runs the network in numpy. The checkpoint lives outside the repo, so
the tests that need weights are skipped when it is absent (set
``SUPERMIMIC_CKPT`` to point somewhere else).

The interesting one is ``ObservationReconstructionTest``: it drives the real
simulator and compares the 99-dim vector the policy builds from ``StepInput``
with the vector the original code would have built from ``MjData``.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import mathx, motion, spec  # noqa: E402
from benchmark.evaluator.env import TrackingEnv  # noqa: E402
from benchmark.policies import supermimic as sm  # noqa: E402

SHORT_CLIP = "fallAndGetUp3_subject1"
SHORT_CSV = C.DATA_DIR / f"{SHORT_CLIP}.csv"

DEFAULT_CHECKPOINT = Path.home() / "Downloads/gptfiles/research/supermimic/colab_baseline/ckpt_015000.pt"
CHECKPOINT = Path(os.environ.get("SUPERMIMIC_CKPT", DEFAULT_CHECKPOINT))


def _have_dataset() -> bool:
    return SHORT_CSV.exists()


def _have_checkpoint() -> bool:
    return CHECKPOINT.exists()


_MODEL: spec.ModelInfo | None = None
_REF = None


def _fixture():
    """Build the frozen model and the short reference once for the whole module."""
    global _MODEL, _REF
    if _MODEL is None:
        _MODEL = spec.build_model()
        _REF = motion.get_reference(SHORT_CSV)
    return _MODEL, _REF


class _Ctx:
    """Minimal stand-in for ``ClipContext`` (the evaluator builds the real one)."""

    def __init__(self, info, reference):
        self.clip = SHORT_CLIP
        self.joint_order = C.JOINT_ORDER
        self.default_joint_pos = np.array(info.default_joint_pos, dtype=np.float64)
        self.action_scale = np.array(info.action_scale, dtype=np.float64)
        self.control_dt = C.STEP_DT
        self.reference = reference


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class ObservationReconstructionTest(unittest.TestCase):
    """``MDP.obs`` must be reconstructible from the policy interface alone."""

    @classmethod
    def setUpClass(cls):
        cls.info, cls.ref = _fixture()
        cls.env = TrackingEnv(cls.ref)
        cls.ref_view = cls.env.reference_view
        cls.policy = sm.SuperMimicPolicy.__new__(sm.SuperMimicPolicy)
        # Only the pieces ``observation``/``goal`` need, so the test runs without
        # a checkpoint: the network is irrelevant to this check.
        cls.policy._context = _Ctx(cls.info, cls.ref_view)
        cls.policy._derived = sm.reference_derivatives(
            cls.ref_view.joint_pos, cls.ref_view.root_pos_w, cls.ref_view.root_quat_w
        )
        cls.policy._derived["yaw"] = np.stack(
            [sm.yaw_matrix(R) for R in cls.policy._derived["R"]]
        )
        cls.pelvis = 1
        cls.root_R = mathx.matrix_from_quat(cls.ref.body_quat_w[:, 0])

    def _ground_truth_obs(self, k: int) -> np.ndarray:
        """What ``supermimic.mdp.MDP.obs`` would build from the simulator state."""
        env, info = self.env, self.info
        # raw simulator state, independent from the policy-interface plumbing
        qpos = env._env.sim.data.qpos[0].cpu().numpy()
        qvel = env._env.sim.data.qvel[0].cpu().numpy()
        sensordata = env._env.sim.data.sensordata[0].cpu().numpy()
        R = mathx.matrix_from_quat(
            env._env.scene["robot"].data.body_link_quat_w[0, 0].cpu().numpy()
        )
        return np.concatenate(
            [
                qpos[info.qpos_adr] - info.default_joint_pos,
                qvel[info.dof_adr],
                sensordata[info.imu_ang_vel_adr : info.imu_ang_vel_adr + 3],
                -R[2, :],
                env._action,
                sm.mat_to_6d(R.T @ self.root_R[k]),
            ]
        )

    def test_matches_simulator_state_over_a_rollout(self):
        worst = 0.0
        checked = 0
        for _ in range(40):
            step = self.env.step_input()
            rebuilt = self.policy.observation(step)
            self.assertEqual(rebuilt.shape, (sm.OBS_DIM,))
            # The robot state at step s belongs to reference frame reference_index - 1.
            worst = max(worst, float(np.abs(rebuilt - self._ground_truth_obs(step.reference_index - 1)).max()))
            target = self.ref.joint_pos[min(step.reference_index, self.ref.num_frames - 1)]
            result = self.env.step((target - self.info.default_joint_pos) / self.info.action_scale)
            checked += 1
            # ``auto_reset=False``: the official env refuses to step past a
            # termination, so the rollout window ends there.
            if result.terminated:
                break
        self.assertGreaterEqual(checked, 15, "rollout ended before providing coverage")
        # float32 reference arrays bound the residual; an approximation would be O(1).
        self.assertLess(worst, 1e-5, f"observation reconstruction error {worst:.2e}")

    def test_pelvis_and_torso_frames_are_consistent(self):
        """``R_torso = R_pelvis @ A(q)`` holds for the frozen model."""
        torso = self.info.body_names.index("torso_link")
        step = self.env.step_input()
        A = sm.waist_rotation(step.joint_pos)
        quat = self.env._env.scene["robot"].data.body_link_quat_w[0].cpu().numpy()
        R_pelvis = mathx.matrix_from_quat(quat[0])
        R_torso = mathx.matrix_from_quat(quat[torso])
        # float32 FK bounds the residual
        self.assertLess(float(np.abs(R_torso - R_pelvis @ A).max()), 1e-5)

    def test_waist_axes_match_the_model(self):
        model = self.info.model
        expected = {"waist_yaw_joint": (0, 0, 1), "waist_roll_joint": (1, 0, 0), "waist_pitch_joint": (0, 1, 0)}
        for name, axis in expected.items():
            jid = model.joint(name).id
            np.testing.assert_allclose(model.jnt_axis[jid], axis, atol=0.0)


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class GoalTest(unittest.TestCase):
    """``MotionLib.goal`` layout, clamping and normalisation."""

    @classmethod
    def setUpClass(cls):
        cls.info, cls.ref = _fixture()
        cls.ref_view = TrackingEnv(cls.ref).reference_view
        cls.policy = sm.SuperMimicPolicy.__new__(sm.SuperMimicPolicy)
        cls.policy.goal_frames = sm.GOAL_FRAMES
        cls.policy._derived = sm.reference_derivatives(
            cls.ref_view.joint_pos, cls.ref_view.root_pos_w, cls.ref_view.root_quat_w
        )
        cls.policy._derived["yaw"] = np.stack(
            [sm.yaw_matrix(R) for R in cls.policy._derived["R"]]
        )

    def test_shape_and_layout(self):
        goal = self.policy.goal(0)
        self.assertEqual(goal.shape, (sm.GOAL_FRAMES * sm.GOAL_PER_FRAME,))
        frame0 = goal[: sm.GOAL_PER_FRAME].reshape(sm.GOAL_PER_FRAME)
        # frame 1 of the clip, expressed in the yaw frame of frame 0
        d = self.policy._derived
        expected_dp = d["yaw"][0].T @ (d["p"][1] - d["p"][0])
        np.testing.assert_allclose(frame0[:3], expected_dp, atol=1e-12)
        np.testing.assert_allclose(frame0[3:9], sm.mat_to_6d(d["yaw"][0].T @ d["R"][1]), atol=1e-12)
        np.testing.assert_allclose(frame0[15:44], d["q"][1], atol=0.0)
        np.testing.assert_allclose(frame0[44:73], d["qd"][1] * 0.1, atol=0.0)

    def test_clamps_to_the_last_frame_and_zeroes_velocity(self):
        last = self.ref_view.num_frames - 1
        goal = self.policy.goal(last - 2)
        frames = goal.reshape(sm.GOAL_FRAMES, sm.GOAL_PER_FRAME)
        d = self.policy._derived
        # requested frames last-1, last, then 8 past the end
        np.testing.assert_allclose(frames[0, :3], d["yaw"][last - 2].T @ (d["p"][last - 1] - d["p"][last - 2]))
        # requested frames last-1 and last are real; everything past is held
        for k in range(2, sm.GOAL_FRAMES):
            np.testing.assert_allclose(frames[k, 9:15], 0.0, atol=0.0)
        np.testing.assert_allclose(frames[1, 15:44], d["q"][last], atol=0.0)

    def test_mat_to_6d_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            A = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            np.testing.assert_allclose(sm.six_to_mat(sm.mat_to_6d(Q)), Q, atol=1e-12)


class ContractTest(unittest.TestCase):
    """The action semantics and joint order must match before anything is run."""

    @classmethod
    def setUpClass(cls):
        cls.info = spec.build_model()

    def test_action_scale_matches_the_frozen_model(self):
        np.testing.assert_allclose(
            sm.action_scale_reference(), self.info.action_scale, atol=1e-8, rtol=0
        )

    def test_default_pose_matches(self):
        np.testing.assert_allclose(
            sm.default_joint_pos_reference(), self.info.default_joint_pos, atol=1e-12
        )

    def test_joint_order_matches(self):
        self.assertEqual(sm.JOINT_ORDER, C.JOINT_ORDER)

    def test_rejects_a_foreign_control_contract(self):
        if not _have_checkpoint():
            self.skipTest("checkpoint not present")
        policy = sm.SuperMimicPolicy.from_checkpoint(CHECKPOINT)
        info, ref = _fixture() if _have_dataset() else (self.info, None)
        if ref is None:
            self.skipTest("dataset not present")
        ctx = _Ctx(info, TrackingEnv(ref).reference_view)
        ctx.action_scale = ctx.action_scale * 1.01
        with self.assertRaises(sm.SuperMimicContractError):
            policy.reset(ctx)


@unittest.skipUnless(_have_checkpoint(), "SuperMimic checkpoint not present")
class CheckpointTest(unittest.TestCase):
    """Weight loading, determinism and the shape of what the network emits."""

    @classmethod
    def setUpClass(cls):
        cls.info, cls.ref = _fixture() if _have_dataset() else (spec.build_model(), None)
        cls.policy = sm.SuperMimicPolicy.from_checkpoint(CHECKPOINT)

    def test_dims_come_from_the_weights(self):
        self.assertEqual(self.policy.hist, sm.HIST)
        self.assertEqual(self.policy.goal_frames, sm.GOAL_FRAMES)
        self.assertEqual(self.policy.tensors["pi.enc.0.weight"].shape, (2048, 730))
        self.assertEqual(self.policy.tensors["pi.dec.0.weight"].shape, (2048, 32 + 99 * 10))
        self.assertEqual(self.policy.parameter_count(), 9_118_135)

    def test_forward_is_deterministic_and_bounded(self):
        rng = np.random.default_rng(1)
        obs = rng.normal(scale=0.1, size=(sm.HIST, sm.OBS_DIM)).astype(np.float32)
        goal = rng.normal(scale=0.1, size=(730,)).astype(np.float32)
        a = self.policy.forward(obs, goal)
        b = self.policy.forward(obs, goal)
        self.assertEqual(a.shape, (29,))
        self.assertTrue(np.all(np.isfinite(a)))
        np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
    def test_reset_clears_history_and_repeats_the_first_observation(self):
        ctx = _Ctx(self.info, TrackingEnv(self.ref).reference_view)
        env = TrackingEnv(self.ref)
        self.policy.reset(ctx)
        self.assertIsNone(self.policy._hist)
        step = env.step_input()
        first = self.policy.act(step)
        self.assertEqual(first.shape, (29,))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertEqual(self.policy._hist.shape, (sm.HIST, sm.OBS_DIM))
        np.testing.assert_array_equal(
            self.policy._hist, np.repeat(self.policy.observation(step)[None, :], sm.HIST, axis=0)
        )

        self.policy.reset(ctx)  # a second clip must not see the first one's history
        self.assertIsNone(self.policy._hist)
        again = self.policy.act(env.step_input())
        np.testing.assert_array_equal(first, again)


if __name__ == "__main__":
    unittest.main()
