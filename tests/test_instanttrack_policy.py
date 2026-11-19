"""CPU unit tests for the InstantTrack / BlinkMimic V2 policy adapter."""

from __future__ import annotations

import collections
import importlib.util
import io
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import policy_api  # noqa: E402
from benchmark.policies import supermimic as sm  # noqa: E402


def _load_instanttrack_policy_module():
    path = REPO_ROOT / "baselines" / "instanttrack" / "policy.py"
    mspec = importlib.util.spec_from_file_location("_test_instanttrack_policy", path)
    assert mspec is not None and mspec.loader is not None
    module = importlib.util.module_from_spec(mspec)
    sys.modules[mspec.name] = module
    mspec.loader.exec_module(module)
    return module


def _load_patch_module():
    path = REPO_ROOT / "baselines" / "instanttrack" / "vm" / "patch_full_collision.py"
    mspec = importlib.util.spec_from_file_location("_test_patch_full_collision", path)
    assert mspec is not None and mspec.loader is not None
    module = importlib.util.module_from_spec(mspec)
    sys.modules[mspec.name] = module
    mspec.loader.exec_module(module)
    return module


it_pol = _load_instanttrack_policy_module()
patch_fc = _load_patch_module()


def _rebuild_tensor_v2(*args, **kwargs):
    raise AssertionError("marker must never be called")


class _FakeStorage:
    def __init__(self, key: str, dtype_name: str, numel: int):
        self.key = key
        self.dtype_name = dtype_name
        self.numel = numel


class _FakeTensor:
    def __init__(self, storage: _FakeStorage, shape: tuple[int, ...], stride: tuple[int, ...]):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)

    def __reduce__(self):
        return (_rebuild_tensor_v2, (self.storage, 0, self.shape, self.stride))


class _ZipPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            return ("storage", obj.dtype_name, obj.key, "cpu", obj.numel)
        return None


def _write_torch_zip(path: Path, nested_pi: dict[str, np.ndarray]) -> None:
    storages: dict[str, bytes] = {}
    pi_refs = collections.OrderedDict()
    for i, (name, arr) in enumerate(nested_pi.items()):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        storages[str(i)] = arr.tobytes()
        stride = tuple(int(arr.strides[j] // arr.itemsize) for j in range(arr.ndim))
        pi_refs[name] = _FakeTensor(_FakeStorage(str(i), "FloatStorage", arr.size), arr.shape, stride)
    payload = {"pi": pi_refs, "iteration": 1200, "wall_time": 110.5}
    buf = io.BytesIO()
    _ZipPickler(buf, protocol=2).dump(payload)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        for key, raw in storages.items():
            zf.writestr(f"archive/data/{key}", raw)


def _make_synthetic_tensors(
    rng: np.random.Generator,
    enc_hidden: tuple[int, ...],
    dec_hidden: tuple[int, ...],
    *,
    with_ref_skip: bool = True,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    d = sm.GOAL_FRAMES * sm.GOAL_PER_FRAME
    for i, h in enumerate(enc_hidden):
        idx = 2 * i
        tensors[f"pi.enc.{idx}.weight"] = (rng.standard_normal((h, d)) * 0.05).astype(np.float32)
        tensors[f"pi.enc.{idx}.bias"] = (rng.standard_normal(h) * 0.01).astype(np.float32)
        d = h
    out_idx = 2 * len(enc_hidden)
    tensors[f"pi.enc.{out_idx}.weight"] = (rng.standard_normal((sm.Z_DIM, d)) * 0.05).astype(np.float32)
    tensors[f"pi.enc.{out_idx}.bias"] = (rng.standard_normal(sm.Z_DIM) * 0.01).astype(np.float32)

    d = sm.Z_DIM + sm.HIST * sm.OBS_DIM
    for i, h in enumerate(dec_hidden):
        idx = 2 * i
        tensors[f"pi.dec.{idx}.weight"] = (rng.standard_normal((h, d)) * 0.05).astype(np.float32)
        tensors[f"pi.dec.{idx}.bias"] = (rng.standard_normal(h) * 0.01).astype(np.float32)
        d = h
    out_idx = 2 * len(dec_hidden)
    tensors[f"pi.dec.{out_idx}.weight"] = (rng.standard_normal((sm.NUM_J, d)) * 0.05).astype(np.float32)
    tensors[f"pi.dec.{out_idx}.bias"] = (rng.standard_normal(sm.NUM_J) * 0.01).astype(np.float32)

    tensors["pi.obs_norm.mean"] = np.zeros(sm.OBS_DIM, dtype=np.float32)
    tensors["pi.obs_norm.var"] = np.ones(sm.OBS_DIM, dtype=np.float32)
    tensors["pi.goal_norm.mean"] = np.zeros(sm.GOAL_FRAMES * sm.GOAL_PER_FRAME, dtype=np.float32)
    tensors["pi.goal_norm.var"] = np.ones(sm.GOAL_FRAMES * sm.GOAL_PER_FRAME, dtype=np.float32)
    if with_ref_skip:
        tensors["pi.ref_q0"] = sm.default_joint_pos_reference().astype(np.float32)
        tensors["pi.ref_alpha"] = sm.action_scale_reference().astype(np.float32)
    return tensors


def _make_reference(num_frames: int = 20) -> policy_api.ReferenceClip:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)[:, None]
    joint_pos = np.tile(sm.default_joint_pos_reference(), (num_frames, 1)) + 0.05 * np.sin(t)
    joint_vel = np.gradient(joint_pos, sm.CONTROL_DT, axis=0)
    root_pos_w = np.zeros((num_frames, 3), dtype=np.float64)
    root_pos_w[:, 0] = 0.2 * t[:, 0]
    root_pos_w[:, 2] = 0.78
    root_quat_w = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (num_frames, 1))
    return policy_api.ReferenceClip(
        num_frames=num_frames,
        control_dt=sm.CONTROL_DT,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
    )


def _make_context(ref: policy_api.ReferenceClip, clip: str = "walk1_subject1") -> policy_api.ClipContext:
    return policy_api.ClipContext(
        clip=clip,
        joint_order=C.JOINT_ORDER,
        default_joint_pos=sm.default_joint_pos_reference(),
        action_scale=sm.action_scale_reference(),
        control_dt=sm.CONTROL_DT,
        reference=ref,
    )


def _make_step(ref: policy_api.ReferenceClip, step_idx: int) -> policy_api.StepInput:
    frame = step_idx
    return policy_api.StepInput(
        step=step_idx,
        reference_index=frame + 1,
        time_s=step_idx * sm.CONTROL_DT,
        joint_pos=ref.joint_pos[frame].copy(),
        joint_vel=ref.joint_vel[frame].copy(),
        base_lin_vel=np.zeros(3, dtype=np.float64),
        base_ang_vel=np.zeros(3, dtype=np.float64),
        anchor_pos_b=np.zeros(3, dtype=np.float64),
        anchor_ori_b=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        previous_action=np.zeros(sm.NUM_J, dtype=np.float64),
        reference=ref,
        anchor_pos_b_prev=np.zeros(3, dtype=np.float64),
        anchor_ori_b_prev=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    )


class InstantTrackPolicyTest(unittest.TestCase):
    def test_factory_requires_checkpoint(self):
        with self.assertRaises(SystemExit):
            it_pol.make_policy(None)

    def test_supports_3_and_4_hidden_layer_architectures(self):
        rng = np.random.default_rng(7)
        for enc_h, dec_h in [
            ((1024, 512, 256), (2048, 1024, 512, 256)),  # InstantTrack v1.2
            ((2048, 1024, 512, 256), (2048, 1024, 512, 256)),  # InstantTrack v1.3/v1.4 & BlinkMimic
        ]:
            tensors = _make_synthetic_tensors(rng, enc_h, dec_h, with_ref_skip=True)
            pol = it_pol.InstantTrackPolicy(tensors)
            policy_api.require_interface(pol)
            ref = _make_reference(20)
            pol.reset(_make_context(ref))
            act = pol.act(_make_step(ref, 0))
            checked = policy_api.check_action(act, clip="c", step=0)
            self.assertEqual(checked.shape, (29,))
            self.assertEqual(checked.dtype, np.float64)

    def test_ref_skip_adds_next_reference_offset(self):
        rng = np.random.default_rng(11)
        t_skip = _make_synthetic_tensors(rng, (64, 32), (64, 32), with_ref_skip=True)
        t_plain = {k: v for k, v in t_skip.items() if k not in ("pi.ref_q0", "pi.ref_alpha")}

        p_skip = it_pol.InstantTrackPolicy(t_skip)
        p_plain = it_pol.InstantTrackPolicy(t_plain)
        ref = _make_reference(20)
        ctx = _make_context(ref)
        p_skip.reset(ctx)
        p_plain.reset(ctx)

        step = _make_step(ref, 2)
        a_skip = p_skip.act(step)
        a_plain = p_plain.act(step)

        expected_skip = (
            ref.joint_pos[step.reference_index].astype(np.float32)
            - ctx.default_joint_pos.astype(np.float32)
        ) / ctx.action_scale.astype(np.float32)
        np.testing.assert_allclose(a_skip - a_plain, expected_skip, rtol=1e-6, atol=1e-6)

    def test_goal_clamps_end_of_clip_matching_motion_lib(self):
        rng = np.random.default_rng(13)
        pol = it_pol.InstantTrackPolicy(_make_synthetic_tensors(rng, (32,), (32,), with_ref_skip=False))
        ref = _make_reference(15)
        pol.reset(_make_context(ref))
        # At frame=12, future frames 13, 14, 14, ... repeat frame 14 without zeroing velocities.
        g = pol.goal(12).reshape(sm.GOAL_FRAMES, sm.GOAL_PER_FRAME)
        np.testing.assert_allclose(g[1], g[2], atol=1e-12)
        np.testing.assert_allclose(g[1, 15 + 29 : 15 + 58], pol._derived["qd"][14] * 0.1, atol=1e-12)

    def test_loads_from_torch_zip_checkpoint_without_torch(self):
        rng = np.random.default_rng(19)
        tensors = _make_synthetic_tensors(rng, (32, 16), (32, 16), with_ref_skip=True)
        pi_only = {k.removeprefix("pi."): v for k, v in tensors.items()}
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt_001200.pt"
            _write_torch_zip(ckpt, pi_only)
            pol = it_pol.make_policy(ckpt)
            ref = _make_reference(20)
            pol.reset(_make_context(ref))
            out = pol.act(_make_step(ref, 0))
            self.assertEqual(out.shape, (29,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_reset_clears_history_and_validates_contract(self):
        rng = np.random.default_rng(23)
        pol = it_pol.InstantTrackPolicy(_make_synthetic_tensors(rng, (32,), (32,), with_ref_skip=True))
        ref = _make_reference(20)
        pol.reset(_make_context(ref, "clip_a"))
        pol.act(_make_step(ref, 0))
        self.assertIsNotNone(pol._hist)

        pol.reset(_make_context(ref, "clip_b"))
        self.assertIsNone(pol._hist)

        bad_ctx = policy_api.ClipContext(
            clip="bad",
            joint_order=tuple(reversed(C.JOINT_ORDER)),
            default_joint_pos=sm.default_joint_pos_reference(),
            action_scale=sm.action_scale_reference(),
            control_dt=sm.CONTROL_DT,
            reference=ref,
        )
        with self.assertRaises(sm.SuperMimicContractError):
            pol.reset(bad_ctx)

        step = _make_step(ref, 0)
        step_no_prev = policy_api.StepInput(
            step=step.step,
            reference_index=step.reference_index,
            time_s=step.time_s,
            joint_pos=step.joint_pos,
            joint_vel=step.joint_vel,
            base_lin_vel=step.base_lin_vel,
            base_ang_vel=step.base_ang_vel,
            anchor_pos_b=step.anchor_pos_b,
            anchor_ori_b=step.anchor_ori_b,
            previous_action=step.previous_action,
            reference=step.reference,
            anchor_pos_b_prev=None,  # type: ignore[arg-type]
            anchor_ori_b_prev=None,  # type: ignore[arg-type]
        )
        with self.assertRaises(sm.SuperMimicContractError):
            pol.act(step_no_prev)


class PatchFullCollisionTest(unittest.TestCase):
    _SAMPLE_G1_CONSTANTS = (
        'MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")\n'
        'MOTION_DIR = os.path.join(DATA_DIR, "lafan1_g1")\n\n'
        "def get_spec() -> mujoco.MjSpec:\n"
        "    spec = mujoco.MjSpec.from_file(MJCF_PATH)\n"
        "    for geom in spec.geoms:\n"
        "        if geom.contype or geom.conaffinity:\n"
        "            geom.contype, geom.conaffinity = 0, 1\n"
        "    return spec\n\n"
        "def _actuators():\n"
        "    return ()\n\n"
        "SPEC = RobotSpec(\n"
        "    name=\"g1\",\n"
        "    motion_glob=\"*.csv\",\n"
        "    load_motion=load_lafan1_csv,\n"
        ")\n"
    )

    def test_patch_is_idempotent_and_enables_full_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g1_constants.py"
            p.write_text(self._SAMPLE_G1_CONSTANTS, encoding="utf-8")
            mjcf = Path("/content/modded-supermimic/assets/g1/g1.xml")
            motions = Path("/content/modded-supermimic/data/lafan1_g1")

            patch_fc.patch_g1_constants(p, mjcf, motions, nconmax=256, njmax=1024)
            once = p.read_text(encoding="utf-8")
            self.assertIn(f"MJCF_PATH = {str(mjcf)!r}", once)
            self.assertIn(f"MOTION_DIR = {str(motions)!r}", once)
            self.assertIn("geom.contype = 1", once)
            self.assertIn("geom.conaffinity = 1", once)
            self.assertNotIn("0, 1", once)
            self.assertIn("njmax=1024", once)
            self.assertIn("nconmax=256", once)

            patch_fc.patch_g1_constants(p, mjcf, motions, nconmax=256, njmax=1024)
            self.assertEqual(p.read_text(encoding="utf-8"), once)


if __name__ == "__main__":
    unittest.main()
