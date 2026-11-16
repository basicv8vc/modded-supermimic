"""The NablaMimic adapter must speak the original's conventions, not its own.

Two things are easy to get silently wrong and impossible to notice from a
shape check:

* the 6-D orientation. ``motion_anchor_ori_b`` slices the first two columns of
  the rotation and then flattens row-major, i.e. ``[m00, m01, m10, m11, m20,
  m21]``. NablaMimic's ``_anchor_rel_pq`` emits ``[m00, m10, m20, m01, m11,
  m21]``. Feeding the evaluator's layout into the network unchanged transposes
  the rotation the goal is conditioned on; the first version of this adapter
  did exactly that and fell over after 36 control steps on ``walk3_subject4``.
* the reference root velocities, which ``ReferenceClip`` does not carry and
  have to be rebuilt from the pose arrays.

The convention tests are pure numpy and always run. The simulator-driven test
needs the LAFAN1 cache and the checkpoint, and skips without them
(``NABLAMIMIC_CKPT`` overrides the path).
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
from benchmark.policies import nablamimic as nm  # noqa: E402

SHORT_CLIP = "walk3_subject4"
SHORT_CSV = C.DATA_DIR / f"{SHORT_CLIP}.csv"
CHECKPOINT = Path(
    os.environ.get(
        "NABLAMIMIC_CKPT",
        str(Path.home() / "Downloads/gptfiles/research/nablamimic/colab_workflow/out"
            / "ea_s0_20261111T065918Z/checkpoints/it_001250.pkl"),
    )
)


def _rotation(seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=4)
    return mathx.matrix_from_quat(q / np.linalg.norm(q))


class SixDimensionalConventionTest(unittest.TestCase):
    """The two 6-D layouts differ by a permutation, and it has to be applied."""

    def test_six_to_mat_inverts_the_evaluators_layout(self):
        for seed in range(8):
            m = _rotation(seed)
            v6 = m[:, :2].reshape(-1)  # mjlab's motion_anchor_ori_b
            recovered = nm.six_to_mat(v6)
            np.testing.assert_allclose(recovered, m, atol=1e-12)

    def test_stacked_reorder_recovers_nablamimics_layout(self):
        for seed in range(8):
            m = _rotation(seed)
            v6 = m[:, :2].reshape(-1)
            np.testing.assert_allclose(
                nm.anchor_ori_stacked(v6), np.concatenate([m[:, 0], m[:, 1]]), atol=1e-12
            )

    def test_identity_layouts_differ_where_the_axes_sit(self):
        eye = np.eye(3)
        v6 = eye[:, :2].reshape(-1)
        # interleaved rows: [m00, m01, m10, m11, m20, m21]
        self.assertEqual(v6.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        # stacked columns: [m00, m10, m20, m01, m11, m21]
        self.assertEqual(
            nm.anchor_ori_stacked(v6).tolist(), np.concatenate([eye[:, 0], eye[:, 1]]).tolist()
        )

    def test_permutation_is_reversible(self):
        m = _rotation(3)
        v6 = m[:, :2].reshape(-1)
        back = nm.anchor_ori_stacked(nm.anchor_ori_stacked(v6))
        # The permutation is a 3-cycle, not an involution: two applications are
        # not the identity, which is exactly why the direction has to be pinned.
        self.assertGreater(float(np.abs(back - v6).max()), 1e-3)
        inverse = np.argsort([0, 2, 4, 1, 3, 5])
        np.testing.assert_allclose(nm.anchor_ori_stacked(v6)[inverse], v6, atol=1e-15)

    def test_layouts_are_not_equal_for_a_nontrivial_rotation(self):
        """A regression guard: the permutation must actually do something."""
        m = _rotation(0)
        v6 = m[:, :2].reshape(-1)
        self.assertGreater(float(np.abs(nm.anchor_ori_stacked(v6) - v6).max()), 1e-3)


class ReferenceVelocityTest(unittest.TestCase):
    """``_root_velocities`` reproduces ``mj_differentiatePos`` (frame included)."""

    @unittest.skipUnless(SHORT_CSV.exists(), "LAFAN1 cache not present")
    def test_matches_mujoco_central_difference(self):
        import mujoco

        info = spec.build_model()
        model = info.model
        ref = motion.get_reference(SHORT_CSV)
        root_pos = np.asarray(ref.body_pos_w[:, 0], np.float64)
        root_quat = np.asarray(ref.body_quat_w[:, 0], np.float64)
        qpos = np.concatenate([root_pos, root_quat, np.asarray(ref.joint_pos, np.float64)], 1)

        h = C.STEP_DT
        qvel = np.zeros((ref.num_frames, model.nv))
        for i in range(ref.num_frames):
            a, b = max(i - 1, 0), min(i + 1, ref.num_frames - 1)
            mujoco.mj_differentiatePos(model, qvel[i], (b - a) * h, qpos[a], qpos[b])

        lin, ang = nm._root_velocities(root_pos, root_quat, h)  # noqa: SLF001
        self.assertLess(float(np.abs(lin - qvel[:, :3]).max()), 1e-9)
        # MuJoCo's free-joint angular velocity is body-frame; the world-frame
        # reading must disagree, otherwise the test proves nothing.
        self.assertLess(float(np.abs(ang - qvel[:, 3:6]).max()), 1e-9)
        nm.ROOT_ANG_VEL_FRAME = "world"
        try:
            _, ang_world = nm._root_velocities(root_pos, root_quat, h)  # noqa: SLF001
        finally:
            nm.ROOT_ANG_VEL_FRAME = "body"
        self.assertGreater(float(np.abs(ang_world - qvel[:, 3:6]).max()), 0.1)


class CheckpointContractTest(unittest.TestCase):
    """The 8-minute recipe's actor, checked against the observation layout."""

    @unittest.skipUnless(CHECKPOINT.exists(), "checkpoint not present")
    def test_actor_shapes_match_the_recipe(self):
        params = nm.load_actor_ema(CHECKPOINT)
        self.assertIn("anchor_dec", params)
        enc_in = params["enc"][0][0].shape[0]
        dec_in = params["dec"][0][0].shape[0]
        out = params["dec"][-1][0].shape[1]
        latent = params["enc"][-1][0].shape[1]
        self.assertEqual(enc_in, nm.GOAL_DIM - nm.ANCHOR_DIM)
        self.assertEqual(enc_in, nm.FUTURE * nm.GOAL_FRAME_DIM)
        self.assertEqual(dec_in, latent + nm.HIST * nm.PROPRIO_DIM + nm.ANCHOR_DIM)
        self.assertEqual(out, nm.NUM_J)


class ProprioceptionTest(unittest.TestCase):
    """The 93-dim vector must agree with the simulator, gravity included."""

    @unittest.skipUnless(
        SHORT_CSV.exists() and CHECKPOINT.exists(), "dataset or checkpoint not present"
    )
    def test_proprioception_matches_mjdata(self):
        from benchmark.evaluator.env import TrackingEnv
        from benchmark.evaluator.policy_api import ClipContext

        info = spec.build_model()
        ref = motion.get_reference(SHORT_CSV)
        env = TrackingEnv(ref)
        policy = nm.NablaMimicPolicy(nm.load_actor_ema(CHECKPOINT))
        policy.reset(
            ClipContext(
                clip=SHORT_CLIP,
                joint_order=C.JOINT_ORDER,
                default_joint_pos=np.asarray(info.default_joint_pos, np.float64),
                action_scale=np.asarray(info.action_scale, np.float64),
                control_dt=C.STEP_DT,
                reference=env.reference_view,
            )
        )

        worst = {"gravity": 0.0, "joint": 0.0, "joint_vel": 0.0, "ang_vel": 0.0}
        for _ in range(12):
            step = env.step_input()
            obs = policy.proprioception(step, step.reference_index - 1)
            qpos = env._env.sim.data.qpos[0].cpu().numpy().astype(np.float64)  # noqa: SLF001
            qvel = env._env.sim.data.qvel[0].cpu().numpy().astype(np.float64)  # noqa: SLF001
            worst["joint"] = max(worst["joint"], float(np.abs(obs[:29] - (qpos[7:] - info.default_joint_pos)).max()))
            worst["joint_vel"] = max(worst["joint_vel"], float(np.abs(obs[29:58] - qvel[6:] * nm.JOINT_VEL_SCALE).max()))
            worst["ang_vel"] = max(worst["ang_vel"], float(np.abs(obs[58:61] - qvel[3:6] * nm.ANG_VEL_SCALE).max()))
            worst["gravity"] = max(worst["gravity"], float(np.abs(obs[61:64] - mathx.projected_gravity(qpos[3:7])).max()))
            if env.step(policy.act(step)).terminated:
                break

        self.assertLess(worst["joint"], 1e-6)
        self.assertLess(worst["joint_vel"], 1e-5)
        self.assertLess(worst["ang_vel"], 1e-5)
        # The gravity vector is rebuilt through the waist chain from the anchor
        # orientation; measured at ~1e-3 on a walking clip.
        self.assertLess(worst["gravity"], 5e-3)


if __name__ == "__main__":
    unittest.main()
