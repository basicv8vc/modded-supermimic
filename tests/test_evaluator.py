"""Conformance + behaviour tests for the canonical evaluator.

These tests do not need a trained checkpoint: they check that the evaluator
reproduces the frozen mjlab v1.6.0 G1 tracking specification, that the rollout is
deterministic, that the reward matches the reference at t=0, and that the policy
adapter accepts real checkpoint files while rejecting mismatched ones.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark import core  # noqa: E402
from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import manifest, mathx, motion, policy, policy_api, spec  # noqa: E402
from benchmark.evaluator.env import TrackingEnv  # noqa: E402
from benchmark.evaluator.runner import ClipResult, build_record, evaluate_clip  # noqa: E402

SHORT_CLIP = "fallAndGetUp3_subject1"
SHORT_CSV = C.DATA_DIR / f"{SHORT_CLIP}.csv"


def _have_dataset() -> bool:
    return SHORT_CSV.exists()


def _scratch_file(suffix: str = ".npz") -> Path:
    """A unique file path in the workspace root (no directory creation needed)."""
    return REPO_ROOT / f"_test_actor_{uuid.uuid4().hex[:12]}{suffix}"


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass  # sandboxed deletes are best-effort; files are gitignored


# --- minimal torch-zip writer -------------------------------------------------
# A torch checkpoint is a pickle whose tensors point at raw storage buffers; the
# helpers below emit that format so the torch-free reader can be tested without
# torch, and without depending on any external checkpoint file.


def _rebuild_tensor_v2(*args, **kwargs):
    """Marker only: ``torchzip`` intercepts this global by name during unpickling."""
    raise AssertionError("marker must never be called")


class _FakeStorage:
    def __init__(self, key: str, dtype_name: str, numel: int):
        self.key = key
        self.dtype_name = dtype_name
        self.numel = numel


class _FakeTensor:
    def __init__(self, storage, shape, stride):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)

    def __reduce__(self):
        return (_rebuild_tensor_v2, (self.storage, 0, self.shape, self.stride))


class _ZipPickler(__import__("pickle").Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            return ("storage", obj.dtype_name, obj.key, "cpu", obj.numel)
        return None


def _geom_name(model, index: int) -> str:
    import mujoco  # noqa: PLC0415

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""


class ModelSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = spec.build_model()

    def test_structural_fingerprint(self):
        fp = spec.fingerprint(self.info)
        self.assertEqual(fp["nbody"], 31)  # 1 world + 30 robot bodies
        self.assertEqual(fp["njnt"], 30)  # 1 free + 29 hinge
        self.assertEqual(fp["nq"], 36)
        self.assertEqual(fp["nv"], 35)
        self.assertEqual(fp["nu"], 29)
        self.assertEqual(fp["ngeom"], 69)  # 68 robot geoms + ground plane
        self.assertEqual(C.ROBOT_DOF, 29)

    def test_joint_order_frozen(self):
        self.assertEqual(self.info.joint_names, C.JOINT_ORDER)
        self.assertEqual(len(set(C.JOINT_ORDER)), 29)

    def test_full_self_collision_enabled(self):
        model = self.info.model
        colliding = [
            g
            for g in range(model.ngeom)
            if model.geom_contype[g] == 1 and model.geom_conaffinity[g] == 1
        ]
        # 33 robot collision geoms + the ground plane.
        self.assertEqual(len(colliding), 34)
        visual = [
            g
            for g in range(model.ngeom)
            if not _geom_name(model, g).endswith("_collision")
            and _geom_name(model, g) != "terrain"
        ]
        self.assertGreater(len(visual), 0)
        for g in visual:
            self.assertEqual(model.geom_contype[g], 0)
            self.assertEqual(model.geom_conaffinity[g], 0)

    def test_foot_contact_parameters(self):
        model = self.info.model
        foot_ids = [
            g
            for g in range(model.ngeom)
            if _geom_name(model, g).startswith(("left_foot", "right_foot"))
            and _geom_name(model, g).endswith("_collision")
        ]
        self.assertEqual(len(foot_ids), 14)
        for g in foot_ids:
            self.assertEqual(model.geom_condim[g], 3)
            self.assertEqual(model.geom_priority[g], 1)
            self.assertAlmostEqual(model.geom_friction[g, 0], C.FOOT_FRICTION)

    def test_actuator_gains(self):
        model = self.info.model
        kp = model.actuator_gainprm[self.info.actuator_ids, 0]
        kv = -model.actuator_biasprm[self.info.actuator_ids, 2]
        expected_kp = np.zeros(29)
        expected_eff = np.zeros(29)
        for group in C.ACTUATOR_GROUPS:
            import re  # noqa: PLC0415

            patterns = [re.compile(p) for p in group.patterns]
            for i, name in enumerate(C.JOINT_ORDER):
                if any(p.fullmatch(name) for p in patterns):
                    expected_kp[i] = group.stiffness
                    expected_eff[i] = group.effort_limit
        np.testing.assert_allclose(kp, expected_kp, rtol=1e-6)
        np.testing.assert_allclose(kv / kp, 2.0 * C.DAMPING_RATIO / C.NATURAL_FREQ, rtol=1e-9)
        np.testing.assert_allclose(model.actuator_forcerange[self.info.actuator_ids, 1], expected_eff)
        self.assertFalse(bool(model.actuator_ctrllimited[self.info.actuator_ids].any()))

    def test_action_scale(self):
        expected = np.zeros(29)
        for group in C.ACTUATOR_GROUPS:
            import re  # noqa: PLC0415

            patterns = [re.compile(p) for p in group.patterns]
            for i, name in enumerate(C.JOINT_ORDER):
                if any(p.fullmatch(name) for p in patterns):
                    expected[i] = 0.25 * group.effort_limit / group.stiffness
        np.testing.assert_allclose(self.info.action_scale, expected)
        self.assertGreater(self.info.action_scale.min(), 0.0)

    def test_default_joint_pos(self):
        pos = self.info.default_joint_pos
        idx = {n: i for i, n in enumerate(C.JOINT_ORDER)}
        self.assertAlmostEqual(pos[idx["left_knee_joint"]], 0.669)
        self.assertAlmostEqual(pos[idx["right_hip_pitch_joint"]], -0.312)
        self.assertAlmostEqual(pos[idx["left_elbow_joint"]], 0.6)
        self.assertAlmostEqual(pos[idx["left_ankle_roll_joint"]], 0.0)

    def test_soft_limits(self):
        hard = self.info.joint_pos_limits
        soft = self.info.soft_joint_pos_limits
        np.testing.assert_allclose((soft[:, 0] + soft[:, 1]) / 2, (hard[:, 0] + hard[:, 1]) / 2)
        np.testing.assert_allclose(
            soft[:, 1] - soft[:, 0], (hard[:, 1] - hard[:, 0]) * C.SOFT_JOINT_LIMIT_FACTOR
        )

    def test_sim_options(self):
        opt = self.info.model.opt
        self.assertAlmostEqual(opt.timestep, 0.005)
        self.assertEqual(opt.iterations, 10)
        self.assertEqual(opt.ls_iterations, 20)
        self.assertEqual(opt.integrator, 3)  # implicitfast
        self.assertEqual(opt.cone, 0)  # pyramidal


class ManifestTest(unittest.TestCase):
    def test_manifest_hashes_are_current(self):
        frozen = manifest.load_manifest()
        current = manifest.build_manifest()
        self.assertEqual(frozen["robot_model"], current["robot_model"])
        self.assertEqual(frozen["dataset"], current["dataset"])
        self.assertEqual(len(frozen["robot_model"]["sha256"]), 64)
        self.assertEqual(len(frozen["dataset"]["manifest_sha256"]), 64)

    def test_dataset_has_40_clips(self):
        frozen = json.loads((C.DATA_DIR.parent / "lafan1_g1.sha256.json").read_text())
        self.assertEqual(len(frozen), 40)
        protocol_clips = set(core.load_protocol()["clip_ids"])
        self.assertEqual(set(frozen), protocol_clips)


class MathTest(unittest.TestCase):
    def test_quat_apply_matches_matrix(self):
        rng = np.random.default_rng(0)
        q = mathx.normalize(rng.normal(size=4))
        v = rng.normal(size=3)
        np.testing.assert_allclose(
            mathx.quat_apply(q, v), mathx.matrix_from_quat(q) @ v, atol=1e-12
        )

    def test_quat_error(self):
        q = mathx.quat_from_euler_xyz(0.0, 0.0, np.pi / 3)
        self.assertAlmostEqual(float(mathx.quat_error_magnitude(np.array([1.0, 0, 0, 0]), q)), np.pi / 3)

    def test_yaw_quat_removes_roll_pitch(self):
        q = mathx.quat_from_euler_xyz(0.3, -0.2, 0.7)
        yaw = mathx.yaw_quat(q)
        self.assertAlmostEqual(float(mathx.quat_error_magnitude(q, yaw)), mathx.quat_error_magnitude(
            q, yaw
        ))
        rot = mathx.matrix_from_quat(yaw)
        self.assertAlmostEqual(rot[2, 2], 1.0, places=12)

    def test_subtract_frame_transforms(self):
        q = mathx.quat_from_euler_xyz(0.1, 0.2, 0.3)
        t = np.array([1.0, 2.0, 3.0])
        pos_b, quat_b = mathx.subtract_frame_transforms(t, q, t, q)
        np.testing.assert_allclose(pos_b, 0.0, atol=1e-12)
        np.testing.assert_allclose(quat_b, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class MotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = spec.build_model()
        cls.ref = motion.get_reference(SHORT_CSV)

    def test_frame_count(self):
        rows = np.loadtxt(SHORT_CSV, delimiter=",").shape[0]
        duration = (rows - 1) / C.LAFAN1_INPUT_FPS
        expected = len(np.arange(0.0, duration, C.STEP_DT))
        self.assertEqual(self.ref.num_frames, expected)
        self.assertEqual(self.ref.joint_pos.shape[1], 29)
        self.assertEqual(self.ref.body_pos_w.shape[1], 30)

    def test_first_frame_matches_csv(self):
        raw = np.loadtxt(SHORT_CSV, delimiter=",")
        np.testing.assert_allclose(self.ref.joint_pos[0], raw[0, 7:], atol=1e-6)
        np.testing.assert_allclose(self.ref.body_pos_w[0, 0], raw[0, :3], atol=1e-6)

    def test_quaternions_normalized(self):
        norms = np.linalg.norm(self.ref.body_quat_w, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_velocity_shapes_finite(self):
        for array in (
            self.ref.joint_vel,
            self.ref.body_lin_vel_w,
            self.ref.body_ang_vel_w,
        ):
            self.assertTrue(np.all(np.isfinite(array)))
        self.assertEqual(self.ref.body_ang_vel_w.shape, self.ref.body_pos_w.shape)


class _StubConverter:
    """Stand-in for OfficialConverter: records calls, writes a tiny cache."""

    def __init__(self):
        self.calls: list[str] = []

    def convert(self, csv_path, npz_path):
        self.calls.append(csv_path.name)
        ref = motion.MotionReference(
            clip_id=csv_path.stem,
            joint_pos=np.zeros((2, 29), dtype=np.float32),
            joint_vel=np.zeros((2, 29), dtype=np.float32),
            body_pos_w=np.zeros((2, 30, 3), dtype=np.float32),
            body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2, 30, 1)),
            body_lin_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
            source_sha256=motion.sha256_file(csv_path),
        )
        motion.save_reference(ref, npz_path)
        return ref


class CacheVersionTest(unittest.TestCase):
    """A cache from another conversion pipeline must be rebuilt, never loaded.

    Regression guard for the stale-cache hazard: only the source CSV hash
    used to be checked, so a pre-refactor npz (different velocity convention)
    could silently survive an evaluator upgrade.
    """

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.csv = self.tmp / "clip_a.csv"
        self.csv.write_text("0,0,0,1,0,0,0,0\n", encoding="utf-8")
        self.cache_dir = self.tmp / "npz"
        self.stub = _StubConverter()
        self._saved_converter = motion._converter
        self._saved_factory = motion._get_converter
        motion._converter = self.stub
        motion._get_converter = lambda: self.stub

    def tearDown(self):
        motion._converter = self._saved_converter
        motion._get_converter = self._saved_factory
        self._tmp.cleanup()

    def _write_cache(self, converter_version: str | None):
        """Seed a cache; ``None`` writes the pre-tag (legacy) layout."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / "clip_a.npz"
        arrays = dict(
            clip_id=np.array("clip_a"),
            fps=np.array(50.0),
            source_sha256=np.array(motion.sha256_file(self.csv)),
            joint_pos=np.zeros((2, 29), dtype=np.float32),
            joint_vel=np.zeros((2, 29), dtype=np.float32),
            body_pos_w=np.zeros((2, 30, 3), dtype=np.float32),
            body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2, 30, 1)),
            body_lin_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
        )
        if converter_version is not None:
            arrays["converter_version"] = np.array(converter_version)
        np.savez_compressed(path, **arrays)
        return path

    def test_matching_version_is_loaded_not_rebuilt(self):
        self._write_cache(motion.CONVERTER_VERSION)
        ref = motion.get_reference(self.csv, cache_dir=self.cache_dir)
        self.assertEqual(self.stub.calls, [])
        self.assertEqual(ref.num_frames, 2)

    def test_legacy_cache_without_version_is_rebuilt(self):
        self._write_cache(None)  # pre-refactor layout: no converter_version key
        ref = motion.get_reference(self.csv, cache_dir=self.cache_dir)
        self.assertEqual(self.stub.calls, ["clip_a.csv"])  # invalidated + rebuilt
        # the rebuilt cache carries the current tag
        with np.load(self.cache_dir / "clip_a.npz") as data:
            self.assertEqual(str(data["converter_version"]), motion.CONVERTER_VERSION)

    def test_foreign_version_is_rebuilt(self):
        self._write_cache("some-other-pipeline-v9")
        motion.get_reference(self.csv, cache_dir=self.cache_dir)
        self.assertEqual(self.stub.calls, ["clip_a.csv"])

    def test_changed_csv_is_rebuilt_even_with_current_version(self):
        self._write_cache(motion.CONVERTER_VERSION)
        self.csv.write_text("0,0,0,1,0,0,0,0\n1,0,0,1,0,0,0,0\n", encoding="utf-8")
        motion.get_reference(self.csv, cache_dir=self.cache_dir)
        self.assertEqual(self.stub.calls, ["clip_a.csv"])


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class RolloutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = spec.build_model()
        cls.ref = motion.get_reference(SHORT_CSV)

    def test_reset_reproduces_frame_zero(self):
        env = TrackingEnv(self.ref)
        snap = env.state_snapshot()
        np.testing.assert_allclose(
            snap["root_pos"], self.ref.body_pos_w[0, 0], atol=1e-5
        )
        np.testing.assert_allclose(
            snap["root_quat"], self.ref.body_quat_w[0, 0], atol=1e-5
        )
        np.testing.assert_allclose(
            snap["joint_pos"], self.ref.joint_pos[0], atol=1e-5
        )
        # full-body FK agrees with the converter's npz (same mjlab Scene code)
        fk = env._env.scene["robot"].data.body_link_pos_w[0].cpu().numpy()
        np.testing.assert_allclose(fk, self.ref.body_pos_w[0], atol=1e-5)

    def test_reset_command_index_advances_to_one(self):
        """mjlab pairing: reset() ends with ``command_manager.compute(dt=0)``,
        so the first scored frame is 1 and frame 0 is only the initial
        condition. The pre-refactor evaluator scored frames 0..T-2; the
        constant +1 offset is documented in ``reports/alignment_v1_vs_mjlab``.
        """
        env = TrackingEnv(self.ref)
        self.assertEqual(env.ref_index, 1)
        self.assertEqual(env.step_input().reference_index, 1)

    def test_observation_layout(self):
        env = TrackingEnv(self.ref)
        obs = env.observation()
        self.assertEqual(obs.shape, (C.OBS_DIM,))
        self.assertEqual(C.OBS_DIM, 160)
        # the first 58 dims are the reference *command* -- under mjlab's
        # pairing the command frame after reset is 1, not 0
        k = env.ref_index
        self.assertEqual(k, 1)
        np.testing.assert_allclose(obs[0:29], self.ref.joint_pos[k], atol=1e-5)
        np.testing.assert_allclose(obs[29:58], self.ref.joint_vel[k], atol=1e-5)
        # the state segments carry the frame-0 initial condition (robot state)
        # last 29 dims are the previous (zero) action
        np.testing.assert_allclose(obs[-29:], 0.0)
        # joint_pos term is ``qpos - default_joint_pos`` (encoder bias frozen to 0)
        expected_joint = np.clip(
            self.ref.joint_pos[0],
            self.info.soft_joint_pos_limits[:, 0],
            self.info.soft_joint_pos_limits[:, 1],
        ) - self.info.default_joint_pos
        np.testing.assert_allclose(obs[73:102], expected_joint, atol=1e-5)
        np.testing.assert_allclose(obs[102:131], self.ref.joint_vel[0], atol=1e-5)
        # anchor channels are exactly the policy-interface anchor fields; they
        # compare the frame-0 state against frame 1 under mjlab's pairing, so
        # they are not zero here (zero error is checked on a frozen clip in
        # test_anchor_error_is_zero_when_state_matches_target).
        step = env.step_input()
        np.testing.assert_allclose(obs[58:61], step.anchor_pos_b, atol=1e-12)
        np.testing.assert_allclose(obs[61:67], step.anchor_ori_b, atol=1e-12)

    def test_zero_action_targets_default_pose(self):
        env = TrackingEnv(self.ref)
        env.step(np.zeros(29))
        # JointPositionAction: processed = raw * scale + offset; raw == 0.
        # ``sim.data.ctrl`` is in model actuator order; reindex into joint
        # order with the frozen actuator mapping (same as the v1 evaluator).
        ctrl = env._env.sim.data.ctrl[0].cpu().numpy()[self.info.actuator_ids]
        np.testing.assert_allclose(ctrl, env.default_joint_pos, atol=1e-9)

    def test_reward_denominator(self):
        self.assertAlmostEqual(C.MAX_STEP_REWARD, 0.1)

    def test_near_reference_step_is_near_max_reward(self):
        """Holding the reference pose the six tracking terms must be ~1 each.

        Under mjlab's pairing the first scored frame is 1 while the reset state
        is frame 0, so this allows for one frame of tracking error.
        """
        env = TrackingEnv(self.ref)
        obs = env.observation()
        action = (obs[:29] - self.info.default_joint_pos) / self.info.action_scale
        result = env.step(action)
        tracking = sum(
            v
            for k, v in result.terms.items()
            if k.startswith("motion_") and v > 0
        )
        # 5.0 * 0.02 = 0.1 is the maximum; allow for one step of tracking error.
        self.assertGreater(tracking, 0.08)
        self.assertLessEqual(tracking, 0.1 + 1e-9)

    def test_anchor_error_is_zero_when_state_matches_target(self):
        """On a frozen clip (every frame identical) the anchor error at reset
        must vanish regardless of the frame-index pairing: target frame 1 is
        the same pose as the frame-0 initial condition."""
        frames = 3
        body_shape = self.ref.body_pos_w.shape[1:]
        quat_shape = self.ref.body_quat_w.shape[1:]
        frozen = motion.MotionReference(
            clip_id="anchor_frozen",
            joint_pos=np.tile(self.ref.joint_pos[0], (frames, 1)).astype(np.float32),
            joint_vel=np.zeros((frames, 29), dtype=np.float32),
            body_pos_w=np.tile(self.ref.body_pos_w[0], (frames, 1, 1)).astype(np.float32),
            body_quat_w=np.tile(self.ref.body_quat_w[0], (frames, 1, 1)).astype(np.float32),
            body_lin_vel_w=np.zeros((frames,) + body_shape, dtype=np.float32),
            body_ang_vel_w=np.zeros((frames,) + quat_shape[:1] + (3,), dtype=np.float32),
        )
        npz = _scratch_file(".npz")
        self.addCleanup(_remove, npz)
        motion.save_reference(frozen, npz)
        frozen.npz_path = npz

        env = TrackingEnv(frozen)
        obs = env.observation()
        np.testing.assert_allclose(obs[58:61], 0.0, atol=1e-5)
        np.testing.assert_allclose(obs[61:67], [1, 0, 0, 1, 0, 0], atol=1e-5)

    def test_deterministic_rollout(self):
        def run():
            env = TrackingEnv(self.ref)
            obs = env.observation()
            trace = []
            for _ in range(12):
                action = (obs[:29] - self.info.default_joint_pos) / self.info.action_scale
                result = env.step(action)
                obs = env.observation()
                trace.append((result.reward, obs.copy()))
            return trace

        first, second = run(), run()
        for (r1, o1), (r2, o2) in zip(first, second):
            self.assertEqual(r1, r2)
            np.testing.assert_array_equal(o1, o2)

    def test_rollout_is_deterministic_across_process_state(self):
        env1 = TrackingEnv(self.ref)
        env2 = TrackingEnv(self.ref)
        np.testing.assert_array_equal(env1.observation(), env2.observation())

    def test_evaluate_clip_reports_failure(self):
        class Hold:
            kind = "smoke_scripted"

            def reset(self, context):
                self.context = context

            def act(self, step):
                return np.zeros(29)

        result = evaluate_clip(self.info, self.ref, Hold(), SHORT_CLIP)
        self.assertIsInstance(result, ClipResult)
        self.assertTrue(result.failed)
        self.assertEqual(result.steps, self.ref.num_frames - 1)
        self.assertGreaterEqual(result.normalized_return, 0.0)
        self.assertLessEqual(result.normalized_return, 1.0)
        self.assertIn("ee_body_pos", result.fall_terms)

    def test_failed_clip_forfeits_remaining_steps(self):
        class Hold:
            def reset(self, context):
                self.context = context

            def act(self, step):
                return np.zeros(29)

        result = evaluate_clip(self.info, self.ref, Hold(), SHORT_CLIP)
        self.assertLess(result.alive_steps, result.steps)
        self.assertEqual(result.alive_steps, result.fall_step)

    def test_long_horizon_contact_stability(self):
        """600 steps of an unactuated-balance hold must stay finite and land, not blow up.

        The frozen default pose is an inverted pendulum on compliant ankles, so it
        does tip over -- that is physics, and the termination terms must catch it.
        What must never happen is a NaN or the robot sinking through the floor.
        """
        frames = 600
        body_shape = self.ref.body_pos_w.shape[1:]
        quat_shape = self.ref.body_quat_w.shape[1:]
        frozen = motion.MotionReference(
            clip_id="frozen",
            joint_pos=np.tile(self.ref.joint_pos[0], (frames, 1)).astype(np.float32),
            joint_vel=np.zeros((frames, 29), dtype=np.float32),
            body_pos_w=np.tile(self.ref.body_pos_w[0], (frames, 1, 1)).astype(np.float32),
            body_quat_w=np.tile(self.ref.body_quat_w[0], (frames, 1, 1)).astype(np.float32),
            body_lin_vel_w=np.zeros((frames,) + body_shape, dtype=np.float32),
            body_ang_vel_w=np.zeros((frames,) + quat_shape[:1] + (3,), dtype=np.float32),
        )
        # the official MotionCommand loads the reference from npz, so even a
        # synthetic clip must go through the npz file
        npz = _scratch_file(".npz")
        self.addCleanup(_remove, npz)
        motion.save_reference(frozen, npz)
        frozen.npz_path = npz

        env = TrackingEnv(frozen)
        env.reset()
        action = (frozen.joint_pos[0] - self.info.default_joint_pos) / self.info.action_scale
        terminated_terms: tuple[str, ...] = ()
        for _ in range(frames - 2):
            result = env.step(action)
            if result.terminated:
                terminated_terms = result.terminated_terms
                break
        qpos = env._env.sim.data.qpos[0].cpu().numpy()
        qvel = env._env.sim.data.qvel[0].cpu().numpy()
        self.assertTrue(np.all(np.isfinite(qpos)))
        self.assertTrue(np.all(np.isfinite(qvel)))
        self.assertGreater(qpos[2], 0.05)  # never sinks through the floor
        self.assertLess(qpos[2], 1.2)
        # A fall gate must have fired. Which of the three fires first depends
        # on float32-vs-float64 trajectory drift (v1: anchor_ori; official
        # warp float32: ee_body_pos) -- see reports/alignment_v1_vs_mjlab.
        self.assertTrue(
            {"anchor_ori", "anchor_pos", "ee_body_pos"} & set(terminated_terms),
            f"no fall detector fired: {terminated_terms}",
        )

class PolicyAdapterTest(unittest.TestCase):
    def setUp(self):
        self.info = spec.build_model()

    def _write_checkpoint(self, path: Path, *, hidden=(8, 4), seed=0, override_meta=None):
        rng = np.random.default_rng(seed)
        dims = (C.OBS_DIM,) + hidden + (C.ACTION_DIM,)
        payload = {}
        for i in range(len(dims) - 1):
            payload[f"w{i}"] = rng.normal(scale=0.05, size=(dims[i + 1], dims[i])).astype(np.float32)
            payload[f"b{i}"] = np.zeros(dims[i + 1], dtype=np.float32)
        payload["obs_mean"] = rng.normal(size=C.OBS_DIM).astype(np.float32)
        payload["obs_var"] = np.full(C.OBS_DIM, 0.5, dtype=np.float32)
        meta = {
            "joint_names": list(C.JOINT_ORDER),
            "action_scale": [float(x) for x in self.info.action_scale],
            "obs_dim": C.OBS_DIM,
            "action_dim": C.ACTION_DIM,
            "hidden_dims": list(hidden),
            "activation": "elu",
            "source": {"run": "test"},
        }
        if override_meta:
            meta.update(override_meta)
        payload["meta_json"] = np.array(json.dumps(meta))
        np.savez(path, **payload)

    def test_load_and_act(self):
        path = _scratch_file()
        self.addCleanup(_remove, path)
        self._write_checkpoint(path)
        loaded = policy.load_policy(path, self.info)
        obs = np.zeros(C.OBS_DIM)
        # ``forward`` is the raw network call; ``act(step)`` is the evaluator
        # interface and needs a clip context first.
        first = loaded.forward(obs)
        self.assertEqual(first.shape, (C.ACTION_DIM,))
        np.testing.assert_array_equal(first, loaded.forward(obs))  # deterministic
        self.assertFalse(np.allclose(first, 0.0))

    def test_act_requires_reset(self):
        path = _scratch_file()
        self.addCleanup(_remove, path)
        self._write_checkpoint(path)
        loaded = policy.load_policy(path, self.info)
        with self.assertRaises(policy_api.PolicyInterfaceError):
            loaded.act(None)

    def test_rejects_wrong_joint_order(self):
        path = _scratch_file()
        self.addCleanup(_remove, path)
        order = list(C.JOINT_ORDER)
        order[0], order[1] = order[1], order[0]
        self._write_checkpoint(path, override_meta={"joint_names": order})
        with self.assertRaises(policy.PolicyFormatError):
            policy.load_policy(path, self.info)

    def test_rejects_wrong_action_scale(self):
        path = _scratch_file()
        self.addCleanup(_remove, path)
        scale = [float(x) * 2.0 for x in self.info.action_scale]
        self._write_checkpoint(path, override_meta={"action_scale": scale})
        with self.assertRaises(policy.PolicyFormatError):
            policy.load_policy(path, self.info)

    def test_rejects_wrong_observation_dim(self):
        path = _scratch_file()
        self.addCleanup(_remove, path)
        rng = np.random.default_rng(1)
        payload = {
            "w0": rng.normal(size=(8, 85)).astype(np.float32),
            "b0": np.zeros(8, dtype=np.float32),
            "w1": rng.normal(size=(29, 8)).astype(np.float32),
            "b1": np.zeros(29, dtype=np.float32),
            "meta_json": np.array(
                json.dumps(
                    {
                        "joint_names": list(C.JOINT_ORDER),
                        "action_scale": [float(x) for x in self.info.action_scale],
                        "obs_dim": 85,
                        "action_dim": 29,
                        "hidden_dims": [8],
                        "activation": "elu",
                    }
                )
            ),
        }
        np.savez(path, **payload)
        with self.assertRaises(policy.PolicyFormatError):
            policy.load_policy(path, self.info)

    def test_missing_checkpoint(self):
        with self.assertRaises(policy.PolicyFormatError):
            policy.load_policy(Path("/nonexistent/actor.txt"), self.info)


class TorchZipTest(unittest.TestCase):
    """The .pt path must work without torch installed."""

    def setUp(self):
        self.info = spec.build_model()

    def _write_pt(self, path: Path, tensors: dict) -> None:
        import collections
        import io
        import pickle
        import zipfile

        storages: dict[str, bytes] = {}
        refs = collections.OrderedDict()
        for i, (name, arr) in enumerate(tensors.items()):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            storages[str(i)] = arr.tobytes()
            stride = tuple(int(arr.strides[j] // arr.itemsize) for j in range(arr.ndim))
            refs[name] = _FakeTensor(_FakeStorage(str(i), "FloatStorage", arr.size), arr.shape, stride)

        buf = io.BytesIO()
        _ZipPickler(buf, protocol=2).dump(refs)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("archive/data.pkl", buf.getvalue())
            for key, raw in storages.items():
                zf.writestr(f"archive/data/{key}", raw)

    def test_loads_actor_without_torch(self):
        path = _scratch_file(".pt")
        self.addCleanup(_remove, path)
        rng = np.random.default_rng(0)
        dims = (C.OBS_DIM, 8, 4, C.ACTION_DIM)
        tensors = {}
        for i in range(len(dims) - 1):
            tensors[f"actor.{2 * i}.weight"] = rng.normal(scale=0.05, size=(dims[i + 1], dims[i]))
            tensors[f"actor.{2 * i}.bias"] = np.zeros(dims[i + 1])
        tensors["obs_normalizer._mean"] = np.zeros(C.OBS_DIM)
        tensors["obs_normalizer._var"] = np.ones(C.OBS_DIM)
        self._write_pt(path, tensors)

        loaded = policy.load_policy(path, self.info)
        self.assertEqual(loaded.meta.obs_dim, C.OBS_DIM)
        self.assertEqual(loaded.meta.hidden_dims, (8, 4))
        self.assertEqual(loaded.forward(np.zeros(C.OBS_DIM)).shape, (C.ACTION_DIM,))

    def test_rejects_world_model_checkpoint(self):
        """A goal-conditioned world-model checkpoint has no actor MLP."""
        path = _scratch_file(".pt")
        self.addCleanup(_remove, path)
        self._write_pt(
            path,
            {
                "pi.enc.0.weight": np.zeros((4, 730)),
                "pi.enc.0.bias": np.zeros(4),
                "pi.dec.0.weight": np.zeros((4, 32)),
                "pi.dec.0.bias": np.zeros(4),
                "wm.net.0.weight": np.zeros((4, 133)),
            },
        )
        with self.assertRaises(policy.PolicyFormatError) as ctx:
            policy.load_policy(path, self.info)
        self.assertIn("no actor MLP", str(ctx.exception))


@unittest.skipUnless(_have_dataset(), "LAFAN1 dataset not present")
class RecordTest(unittest.TestCase):
    def test_subset_record_is_rejected_by_validator(self):
        info = spec.build_model()
        ref = motion.get_reference(SHORT_CSV)

        class Replay:
            def reset(self, context):
                self.context = context

            def act(self, step):
                ctx = self.context
                target = ctx.reference.joint_pos[step.reference_index]
                return (target - ctx.default_joint_pos) / ctx.action_scale

        result = evaluate_clip(info, ref, Replay(), SHORT_CLIP)
        record = build_record(
            algorithm="unit-test",
            training_commit="0" * 40,
            seed=0,
            gpu_name="cpu-only",
            gpu_count=1,
            checkpoint_sha256="0" * 64,
            training_wall_s=1.0,
            setup_wall_s=1.0,
            results=[result],
            mode="smoke",
        )
        audit = core.audit_record(record)
        self.assertEqual(audit["status"], "INVALID")
        self.assertTrue(any("40 entries" in e for e in audit["errors"]))
        self.assertTrue(any("GPU model" in e for e in audit["errors"]))
        self.assertFalse(audit["physics_e2e_verified"])


if __name__ == "__main__":
    unittest.main()
