"""Tests for Option B batched V2 evaluation (benchmark/v2/batched.py & fast_eval.py).

Tier 1 tests run anywhere with NumPy (plus PyTorch CPU if installed) — no CUDA,
mjlab or LAFAN1 data needed.  They verify:

1. ``PackedMotionBank`` frame packing, boundaries, and ``max_steps`` truncation.
2. Exact numerical parity between ``BatchedSuperMimicPolicy`` (both NumPy and
   PyTorch modes) and serial ``SuperMimicPolicy`` across multi-clip rollouts
   including slot reassignment mid-run and end-of-clip goal clamping.
3. Exact numerical parity between ``BatchedNablaMimicPolicy`` (both NumPy and
   PyTorch modes) and serial ``NablaMimicPolicy`` across multi-clip rollouts
   including ``anchor_ori_b_prev`` temporal alignment and residual actions.
4. ``SlotPolicyAdapter`` fallback for arbitrary custom ``reset/act`` policies.
5. ``evaluate_batched`` end-to-end scheduling (LPT queue, protocol clip-order
   output, idle-slot handling, per-clip ``parts_dir`` atomic write & resume,
   and exact match against serial evaluation) via a mock ``mjlab`` env.
6. ``scripts/v2/fast_eval.py`` backend selection (``--backend``, ``--workers``).
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import mathx, policy_api  # noqa: E402
from benchmark.evaluator.motion import MotionReference  # noqa: E402
from benchmark.policies import nablamimic as nm  # noqa: E402
from benchmark.policies import supermimic as sm  # noqa: E402
from benchmark.v2 import batched as B  # noqa: E402
from benchmark.v2 import spec as S  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _synthetic_ref(clip_id: str, num_frames: int, seed: int) -> MotionReference:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, (num_frames - 1) * C.STEP_DT, num_frames, dtype=np.float64)
    q0 = sm.default_joint_pos_reference()
    joint_pos = q0[None, :] + 0.15 * np.sin(t[:, None] + rng.uniform(0, 2, (1, 29)))
    joint_vel = np.gradient(joint_pos, C.STEP_DT, axis=0)

    nbody = 16
    body_pos_w = np.zeros((num_frames, nbody, 3), dtype=np.float32)
    body_pos_w[:, :, 0] = 0.3 * t[:, None] + rng.uniform(-0.1, 0.1, (1, nbody))
    body_pos_w[:, :, 1] = 0.05 * np.sin(t)[:, None]
    body_pos_w[:, :, 2] = 0.8 + 0.05 * np.cos(t)[:, None]

    # Smooth unit quaternions (wxyz)
    angles = 0.25 * np.sin(1.7 * t + seed)
    body_quat_w = np.zeros((num_frames, nbody, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = np.cos(angles / 2.0)[:, None]
    body_quat_w[:, :, 3] = np.sin(angles / 2.0)[:, None]
    # Add small roll/pitch on pelvis (body 0)
    angles_r = 0.1 * np.cos(1.3 * t + seed)
    body_quat_w[:, 0, 1] = np.sin(angles_r / 2.0) * 0.2
    norms = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
    body_quat_w /= norms

    body_lin_vel_w = np.gradient(body_pos_w, C.STEP_DT, axis=0).astype(np.float32)
    body_ang_vel_w = rng.normal(0.0, 0.1, (num_frames, nbody, 3)).astype(np.float32)

    return MotionReference(
        clip_id=clip_id,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        npz_path=Path(f"/tmp/{clip_id}.npz"),
    )


def _make_supermimic_tensors(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t: dict[str, np.ndarray] = {}
    enc_dims = [(730, 64), (64, 48), (48, 40), (40, 36), (36, 32)]
    dec_dims = [(32 + 10 * 99, 64), (64, 48), (48, 40), (40, 32), (32, 29)]
    for idx, (nin, nout) in zip((0, 2, 4, 6, 8), enc_dims):
        t[f"pi.enc.{idx}.weight"] = (rng.normal(0, 0.08, (nout, nin))).astype(np.float32)
        t[f"pi.enc.{idx}.bias"] = (rng.normal(0, 0.02, (nout,))).astype(np.float32)
    for idx, (nin, nout) in zip((0, 2, 4, 6, 8), dec_dims):
        t[f"pi.dec.{idx}.weight"] = (rng.normal(0, 0.08, (nout, nin))).astype(np.float32)
        t[f"pi.dec.{idx}.bias"] = (rng.normal(0, 0.02, (nout,))).astype(np.float32)
    t["pi.obs_norm.mean"] = rng.normal(0, 0.1, (99,)).astype(np.float32)
    t["pi.obs_norm.var"] = rng.uniform(0.5, 2.0, (99,)).astype(np.float32)
    t["pi.goal_norm.mean"] = rng.normal(0, 0.1, (730,)).astype(np.float32)
    t["pi.goal_norm.var"] = rng.uniform(0.5, 2.0, (730,)).astype(np.float32)
    return t


def _make_nablamimic_params(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    enc_dims = [(680, 64), (64, 48), (48, 32)]
    dec_dims = [(32 + 10 * 93 + 6, 64), (64, 48), (48, 29)]
    return {
        "enc": [
            (rng.normal(0, 0.08, (nin, nout)).astype(np.float32),
             rng.normal(0, 0.02, (nout,)).astype(np.float32))
            for nin, nout in enc_dims
        ],
        "dec": [
            (rng.normal(0, 0.08, (nin, nout)).astype(np.float32),
             rng.normal(0, 0.02, (nout,)).astype(np.float32))
            for nin, nout in dec_dims
        ],
        "anchor_dec": np.array(1, dtype=np.int32),
    }


def _make_step_inputs(rng, ref: MotionReference, step: int, prev_action: np.ndarray) -> policy_api.StepInput:
    k = step + 1
    return policy_api.StepInput(
        step=step,
        reference_index=k,
        time_s=step * C.STEP_DT,
        joint_pos=(ref.joint_pos[k].astype(np.float64) + rng.normal(0, 0.05, 29)),
        joint_vel=(ref.joint_vel[k].astype(np.float64) + rng.normal(0, 0.2, 29)),
        base_lin_vel=rng.normal(0, 0.2, 3),
        base_ang_vel=rng.normal(0, 0.2, 3),
        anchor_pos_b=rng.normal(0, 0.05, 3),
        anchor_ori_b=mathx.matrix_from_quat(
            mathx.normalize(np.array([1.0, *rng.normal(0, 0.1, 3)]))
        )[:, :2].reshape(6),
        previous_action=prev_action.copy(),
        reference=policy_api.ReferenceClip.from_motion(ref),
        anchor_pos_b_prev=rng.normal(0, 0.05, 3),
        anchor_ori_b_prev=mathx.matrix_from_quat(
            mathx.normalize(np.array([1.0, *rng.normal(0, 0.1, 3)]))
        )[:, :2].reshape(6),
    )


class BatchedUnitTests(unittest.TestCase):
    def test_packed_motion_bank_boundaries(self):
        r0 = _synthetic_ref("clip_a", 15, seed=0)
        r1 = _synthetic_ref("clip_b", 8, seed=1)
        bank = B.PackedMotionBank([("clip_a", r0), ("clip_b", r1)])
        self.assertEqual(bank.total_frames, 23)
        self.assertEqual(bank.metas[0].start, 0)
        self.assertEqual(bank.metas[0].end, 15)
        self.assertEqual(bank.metas[0].n_steps, 14)
        self.assertEqual(bank.metas[1].start, 15)
        self.assertEqual(bank.metas[1].end, 23)
        self.assertEqual(bank.metas[1].n_steps, 7)

        bank_cap = B.PackedMotionBank([("clip_a", r0), ("clip_b", r1)], max_steps=5)
        self.assertEqual(bank_cap.metas[0].n_steps, 5)
        self.assertEqual(bank_cap.metas[1].n_steps, 5)

        with self.assertRaises(ValueError):
            B.PackedMotionBank([("too_short", _synthetic_ref("too_short", 1, seed=2))])

    def _run_supermimic_parity(self, use_torch: bool):
        q0 = sm.default_joint_pos_reference()
        scale = sm.action_scale_reference()
        refs = [
            ("clip_long", _synthetic_ref("clip_long", 22, seed=10)),
            ("clip_short", _synthetic_ref("clip_short", 7, seed=11)),
            ("clip_mid", _synthetic_ref("clip_mid", 11, seed=12)),
        ]
        bank = B.PackedMotionBank(refs)
        base = sm.SuperMimicPolicy(_make_supermimic_tensors())
        batched = B.BatchedSuperMimicPolicy(
            base, bank, num_slots=2, default_joint_pos=q0, action_scale=scale,
            control_dt=C.STEP_DT, device="cpu", use_torch=use_torch,
        )

        # Compute serial ground truth for each clip independently
        rng = np.random.default_rng(123)
        serial_steps: dict[str, list[tuple[policy_api.StepInput, np.ndarray]]] = {}
        for cid, ref in refs:
            pol = sm.SuperMimicPolicy(base.tensors)
            pol.reset(policy_api.ClipContext(
                clip=cid, joint_order=C.JOINT_ORDER,
                default_joint_pos=q0.copy(), action_scale=scale.copy(),
                control_dt=C.STEP_DT, reference=policy_api.ReferenceClip.from_motion(ref),
            ))
            prev_a = np.zeros(29, dtype=np.float64)
            seq = []
            for s in range(ref.num_frames - 1):
                st = _make_step_inputs(rng, ref, s, prev_a)
                act = pol.act(st)
                seq.append((st, act))
                prev_a = act
            serial_steps[cid] = seq

        # Run batched with 2 slots across the 3 clips (slot 1 finishes clip_short at
        # step 6 and gets reassigned to clip_mid mid-rollout!)
        slot_cidx = [0, 1]
        slot_step = [0, 0]
        slot_active = [True, True]
        next_q = 2
        batched.reset_slots([0, 1], slot_cidx)
        prev_actions = np.zeros((2, 29), dtype=np.float64)
        max_err = 0.0

        while any(slot_active):
            cur_inputs = []
            gframes = []
            for e in range(2):
                if slot_active[e]:
                    cid = refs[slot_cidx[e]][0]
                    st, _ = serial_steps[cid][slot_step[e]]
                    cur_inputs.append(st)
                    gframes.append(bank.metas[slot_cidx[e]].start + slot_step[e] + 1)
                else:
                    cur_inputs.append(serial_steps[refs[slot_cidx[e]][0]][-1][0])
                    gframes.append(bank.metas[slot_cidx[e]].end - 1)

            kwargs = dict(
                joint_pos=np.stack([x.joint_pos for x in cur_inputs]),
                joint_vel=np.stack([x.joint_vel for x in cur_inputs]),
                base_ang_vel=np.stack([x.base_ang_vel for x in cur_inputs]),
                anchor_ori_b_prev=np.stack([x.anchor_ori_b_prev for x in cur_inputs]),
                prev_actions=prev_actions,
            )
            if use_torch:
                import torch as _t
                acts = batched.act_gpu(
                    global_frames=_t.tensor(gframes, dtype=_t.long),
                    **{k: _t.tensor(v) for k, v in kwargs.items()},
                ).detach().cpu().numpy().astype(np.float64)
            else:
                acts = batched.act_cpu(
                    global_frames_np=np.array(gframes, dtype=np.int64),
                    **kwargs,
                )

            re_slots, re_clips = [], []
            for e in range(2):
                if not slot_active[e]:
                    continue
                cid = refs[slot_cidx[e]][0]
                _, ref_act = serial_steps[cid][slot_step[e]]
                err = float(np.abs(acts[e] - ref_act).max())
                max_err = max(max_err, err)
                prev_actions[e] = acts[e]
                slot_step[e] += 1
                if slot_step[e] >= len(serial_steps[cid]):
                    if next_q < len(refs):
                        slot_cidx[e] = next_q
                        next_q += 1
                        slot_step[e] = 0
                        prev_actions[e] = 0.0
                        re_slots.append(e)
                        re_clips.append(slot_cidx[e])
                    else:
                        slot_active[e] = False
            if re_slots:
                batched.reset_slots(re_slots, re_clips)

        tol = 5e-6
        self.assertLess(max_err, tol, f"use_torch={use_torch} max_err={max_err}")

    def test_supermimic_batched_numpy_parity(self):
        self._run_supermimic_parity(use_torch=False)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_supermimic_batched_torch_parity(self):
        self._run_supermimic_parity(use_torch=True)

    def test_supermimic_batched_temporal_alignment_power(self):
        """Power check: BatchedSuperMimicPolicy must depend on frame-f anchor & frame."""
        q0 = sm.default_joint_pos_reference()
        scale = sm.action_scale_reference()
        ref = _synthetic_ref("clip_power", 12, seed=31)
        bank = B.PackedMotionBank([("clip_power", ref)])
        base = sm.SuperMimicPolicy(_make_supermimic_tensors(seed=5))
        batched = B.BatchedSuperMimicPolicy(
            base, bank, num_slots=1, default_joint_pos=q0, action_scale=scale,
            control_dt=C.STEP_DT, device="cpu", use_torch=False,
        )
        rng = np.random.default_rng(77)
        st = _make_step_inputs(rng, ref, 3, np.zeros(29, dtype=np.float64))

        pol_ok = sm.SuperMimicPolicy(base.tensors)
        pol_ok.reset(policy_api.ClipContext(
            clip="clip_power", joint_order=C.JOINT_ORDER,
            default_joint_pos=q0.copy(), action_scale=scale.copy(),
            control_dt=C.STEP_DT, reference=policy_api.ReferenceClip.from_motion(ref),
        ))
        for s in range(3):
            pol_ok.act(_make_step_inputs(rng, ref, s, np.zeros(29, dtype=np.float64)))
        serial_act = pol_ok.act(st)

        batched.reset_slots([0], [0])
        rng2 = np.random.default_rng(77)
        _ = _make_step_inputs(rng2, ref, 3, np.zeros(29, dtype=np.float64))
        for s in range(3):
            prev_st = _make_step_inputs(rng2, ref, s, np.zeros(29, dtype=np.float64))
            batched.act_cpu(
                global_frames_np=np.array([s + 1], dtype=np.int64),
                joint_pos=prev_st.joint_pos[None, :],
                joint_vel=prev_st.joint_vel[None, :],
                base_ang_vel=prev_st.base_ang_vel[None, :],
                anchor_ori_b_prev=prev_st.anchor_ori_b_prev[None, :],
                prev_actions=np.zeros((1, 29), dtype=np.float64),
            )
        act_good = batched.act_cpu(
            global_frames_np=np.array([4], dtype=np.int64),
            joint_pos=st.joint_pos[None, :],
            joint_vel=st.joint_vel[None, :],
            base_ang_vel=st.base_ang_vel[None, :],
            anchor_ori_b_prev=st.anchor_ori_b_prev[None, :],
            prev_actions=np.zeros((1, 29), dtype=np.float64),
        )[0]
        np.testing.assert_allclose(act_good, serial_act, atol=5e-6)

        batched.reset_slots([0], [0])
        rng3 = np.random.default_rng(77)
        _ = _make_step_inputs(rng3, ref, 3, np.zeros(29, dtype=np.float64))
        for s in range(3):
            prev_st = _make_step_inputs(rng3, ref, s, np.zeros(29, dtype=np.float64))
            batched.act_cpu(
                global_frames_np=np.array([s + 1], dtype=np.int64),
                joint_pos=prev_st.joint_pos[None, :],
                joint_vel=prev_st.joint_vel[None, :],
                base_ang_vel=prev_st.base_ang_vel[None, :],
                anchor_ori_b_prev=prev_st.anchor_ori_b_prev[None, :],
                prev_actions=np.zeros((1, 29), dtype=np.float64),
            )
        act_wrong_anchor = batched.act_cpu(
            global_frames_np=np.array([4], dtype=np.int64),
            joint_pos=st.joint_pos[None, :],
            joint_vel=st.joint_vel[None, :],
            base_ang_vel=st.base_ang_vel[None, :],
            anchor_ori_b_prev=st.anchor_ori_b[None, :],
            prev_actions=np.zeros((1, 29), dtype=np.float64),
        )[0]
        self.assertGreater(float(np.abs(act_good - act_wrong_anchor).max()), 1e-3)

    def _run_nablamimic_parity(self, use_torch: bool):
        q0 = nm.default_joint_pos_reference()
        scale = nm.action_scale_reference()
        refs = [
            ("clip_long", _synthetic_ref("clip_long", 20, seed=20)),
            ("clip_short", _synthetic_ref("clip_short", 6, seed=21)),
            ("clip_mid", _synthetic_ref("clip_mid", 10, seed=22)),
        ]
        bank = B.PackedMotionBank(refs)
        base = nm.NablaMimicPolicy(_make_nablamimic_params())
        batched = B.BatchedNablaMimicPolicy(
            base, bank, num_slots=2, default_joint_pos=q0, action_scale=scale,
            control_dt=C.STEP_DT, device="cpu", use_torch=use_torch,
        )

        rng = np.random.default_rng(456)
        serial_steps: dict[str, list[tuple[policy_api.StepInput, np.ndarray]]] = {}
        for cid, ref in refs:
            pol = nm.NablaMimicPolicy({"enc": base.enc, "dec": base.dec, "anchor_dec": 1})
            pol.reset(policy_api.ClipContext(
                clip=cid, joint_order=C.JOINT_ORDER,
                default_joint_pos=q0.copy(), action_scale=scale.copy(),
                control_dt=C.STEP_DT, reference=policy_api.ReferenceClip.from_motion(ref),
            ))
            prev_a = np.zeros(29, dtype=np.float64)
            seq = []
            for s in range(ref.num_frames - 1):
                st = _make_step_inputs(rng, ref, s, prev_a)
                act = pol.act(st)
                seq.append((st, act))
                prev_a = act
            serial_steps[cid] = seq

        slot_cidx = [0, 1]
        slot_step = [0, 0]
        slot_active = [True, True]
        next_q = 2
        batched.reset_slots([0, 1], slot_cidx)
        max_err = 0.0

        while any(slot_active):
            cur_inputs = []
            gframes = []
            for e in range(2):
                if slot_active[e]:
                    cid = refs[slot_cidx[e]][0]
                    st, _ = serial_steps[cid][slot_step[e]]
                    cur_inputs.append(st)
                    gframes.append(bank.metas[slot_cidx[e]].start + slot_step[e] + 1)
                else:
                    cur_inputs.append(serial_steps[refs[slot_cidx[e]][0]][-1][0])
                    gframes.append(bank.metas[slot_cidx[e]].end - 1)

            kwargs = dict(
                joint_pos=np.stack([x.joint_pos for x in cur_inputs]),
                joint_vel=np.stack([x.joint_vel for x in cur_inputs]),
                base_ang_vel=np.stack([x.base_ang_vel for x in cur_inputs]),
                anchor_ori_b_prev=np.stack([x.anchor_ori_b_prev for x in cur_inputs]),
            )
            if use_torch:
                import torch as _t
                acts = batched.act_gpu(
                    global_frames=_t.tensor(gframes, dtype=_t.long),
                    **{k: _t.tensor(v) for k, v in kwargs.items()},
                ).detach().cpu().numpy().astype(np.float64)
            else:
                acts = batched.act_cpu(
                    global_frames_np=np.array(gframes, dtype=np.int64),
                    **kwargs,
                )

            re_slots, re_clips = [], []
            for e in range(2):
                if not slot_active[e]:
                    continue
                cid = refs[slot_cidx[e]][0]
                _, ref_act = serial_steps[cid][slot_step[e]]
                err = float(np.abs(acts[e] - ref_act).max())
                max_err = max(max_err, err)
                slot_step[e] += 1
                if slot_step[e] >= len(serial_steps[cid]):
                    if next_q < len(refs):
                        slot_cidx[e] = next_q
                        next_q += 1
                        slot_step[e] = 0
                        re_slots.append(e)
                        re_clips.append(slot_cidx[e])
                    else:
                        slot_active[e] = False
            if re_slots:
                batched.reset_slots(re_slots, re_clips)

        tol = 5e-6
        self.assertLess(max_err, tol, f"use_torch={use_torch} max_err={max_err}")

    def test_nablamimic_batched_numpy_parity(self):
        self._run_nablamimic_parity(use_torch=False)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_nablamimic_batched_torch_parity(self):
        self._run_nablamimic_parity(use_torch=True)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_evaluate_batched_lpt_resume_and_scoring_mock(self):
        """End-to-end mock of ManagerBasedRlEnv verifying LPT queue, resume & scoring."""
        import torch as _t

        q0 = sm.default_joint_pos_reference()
        scale = sm.action_scale_reference()
        info = types.SimpleNamespace(
            joint_names=C.JOINT_ORDER,
            default_joint_pos=q0.copy(),
            action_scale=scale.copy(),
            qpos_adr=np.arange(7, 36, dtype=np.int64),
            dof_adr=np.arange(6, 35, dtype=np.int64),
            imu_lin_vel_adr=0,
            imu_ang_vel_adr=3,
        )
        refs = {
            "clip_a": _synthetic_ref("clip_a", 6, seed=1),   # 5 steps
            "clip_b": _synthetic_ref("clip_b", 10, seed=2),  # 9 steps (longest)
            "clip_c": _synthetic_ref("clip_c", 8, seed=3),   # 7 steps
            "clip_d": _synthetic_ref("clip_d", 5, seed=4),   # 4 steps
        }

        class FakeMotion:
            def __init__(self, n):
                self.joint_pos = _t.zeros((n, 29), dtype=_t.float32)
                self.joint_vel = _t.zeros((n, 29), dtype=_t.float32)
                self.body_pos_w = _t.zeros((n, 14, 3), dtype=_t.float32)
                self.body_quat_w = _t.zeros((n, 14, 4), dtype=_t.float32)
                self.body_lin_vel_w = _t.zeros((n, 14, 3), dtype=_t.float32)
                self.body_ang_vel_w = _t.zeros((n, 14, 3), dtype=_t.float32)
                self.time_step_total = n

        class FakeMotionCommand:
            def __init__(self, num_envs, env):
                self._env = env
                self._pending_forward = False
                self.body_indexes = _t.arange(14, dtype=_t.long)
                self.motion_anchor_body_index = 7
                self.motion = FakeMotion(6)
                self.time_steps = _t.zeros(num_envs, dtype=_t.long)
                self.robot_anchor_pos_w = _t.zeros((num_envs, 3), dtype=_t.float32)
                self.robot_anchor_quat_w = _t.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=_t.float32)

            @property
            def body_pos_w(self):
                return self.motion.body_pos_w[self.time_steps]

            @property
            def body_quat_w(self):
                return self.motion.body_quat_w[self.time_steps]

            @property
            def body_lin_vel_w(self):
                return self.motion.body_lin_vel_w[self.time_steps]

            @property
            def body_ang_vel_w(self):
                return self.motion.body_ang_vel_w[self.time_steps]

            @property
            def joint_pos(self):
                return self.motion.joint_pos[self.time_steps]

            @property
            def joint_vel(self):
                return self.motion.joint_vel[self.time_steps]

            def _write_reference_state_to_sim(self, *args):
                return None

            def update_relative_body_poses(self):
                return None

            def _resample_command(self, env_ids):
                self.time_steps[env_ids] = 0

            def _update_command(self, env_ids=None):
                if env_ids is None:
                    self.time_steps += 1
                else:
                    self.time_steps[env_ids] += 1

        class FakeRewardManager:
            def __init__(self, num_envs):
                self.active_terms = list(S.REWARD_TERM_NAMES)
                self._step_reward = _t.zeros((num_envs, len(self.active_terms)), dtype=_t.float32)

        class FakeTermManager:
            def __init__(self, num_envs):
                self._term_names = ["time_out", "root_height", "root_tilt", "nonfinite"]
                self._dones = {n: _t.zeros(num_envs, dtype=_t.bool) for n in self._term_names}

            def get_term(self, name):
                return self._dones[name]

        class FakeEnv:
            def __init__(self, cfg, device):
                self.num_envs = cfg.scene.num_envs
                self.device = device
                self._cmd = FakeMotionCommand(self.num_envs, self)
                self._manual_reset_pending = _t.zeros(self.num_envs, dtype=_t.bool)
                robot_obj = types.SimpleNamespace(
                    data=types.SimpleNamespace(
                        joint_pos=_t.zeros((self.num_envs, 29), dtype=_t.float32),
                        joint_vel=_t.zeros((self.num_envs, 29), dtype=_t.float32),
                    )
                )
                class _Scene(dict):
                    pass
                sc = _Scene({"robot": robot_obj})
                sc.env_origins = _t.zeros((self.num_envs, 3), dtype=_t.float32)
                self.scene = sc
                self.sim = types.SimpleNamespace(
                    mj_model=object(),
                    data=types.SimpleNamespace(
                        sensordata=_t.zeros((self.num_envs, 12), dtype=_t.float32),
                        qpos=_t.ones((self.num_envs, 36), dtype=_t.float32),
                        qvel=_t.zeros((self.num_envs, 35), dtype=_t.float32),
                    ),
                    forward=lambda: None,
                )
                self.command_manager = types.SimpleNamespace(get_term=lambda _n: self._cmd)
                self.action_manager = types.SimpleNamespace(
                    get_term=lambda _n: types.SimpleNamespace(
                        offset=_t.tensor(q0, dtype=_t.float32),
                        scale=_t.tensor(scale, dtype=_t.float32),
                    )
                )
                self.reward_manager = FakeRewardManager(self.num_envs)
                self.termination_manager = FakeTermManager(self.num_envs)
                self.local_step = _t.zeros(self.num_envs, dtype=_t.long)

            def reset(self, env_ids=None):
                if env_ids is None:
                    env_ids = _t.arange(self.num_envs, dtype=_t.long)
                self._cmd._resample_command(env_ids)
                self._cmd._update_command(env_ids)
                self.local_step[env_ids] = 0
                self._manual_reset_pending[env_ids] = False
                return {}, {}

            def step(self, action):
                assert not _t.any(self._manual_reset_pending), "unreset terminated env!"
                # Deterministic reward = 0.05 per step; fall if time_steps matches clip_c step 2
                self.local_step += 1
                self.reward_manager._step_reward.fill_(0.5)  # * 0.02 -> 0.01 per term * 8 = 0.08
                rew = self.reward_manager._step_reward.sum(dim=-1) * C.STEP_DT
                for n in self.termination_manager._term_names:
                    self.termination_manager._dones[n].zero_()
                # Make clip_c (which starts at 6+10 = 16 in LPT order: clip_b(10), clip_c(8), clip_a(6))
                # fail on its 3rd step (local_step == 3)
                fall = (self._cmd.time_steps == (10 + 3))
                self.termination_manager._dones["root_height"] = fall
                self._manual_reset_pending |= fall
                self._cmd._update_command(None)
                return {}, rew, fall, _t.zeros_like(fall), {}

            def close(self):
                return None

        class CustomPolicy:
            def __init__(self):
                self.clip = None

            def reset(self, ctx):
                self.clip = ctx.clip

            def act(self, step):
                return np.zeros(29, dtype=np.float64)

        fake_mjlab_envs = types.ModuleType("mjlab.envs")
        fake_mjlab_envs.ManagerBasedRlEnv = FakeEnv
        fake_mjlab_cmds = types.ModuleType("mjlab.tasks.tracking.mdp.commands")
        fake_mjlab_cmds.MotionCommand = FakeMotionCommand
        fake_mjlab_obs = types.ModuleType("mjlab.tasks.tracking.mdp.observations")
        fake_mjlab_obs.motion_anchor_pos_b = lambda env, _n: _t.zeros((env.num_envs, 3), dtype=_t.float32)
        fake_mjlab_obs.motion_anchor_ori_b = lambda env, _n: _t.zeros((env.num_envs, 6), dtype=_t.float32)
        fake_mjlab_math = types.ModuleType("mjlab.utils.lab_api.math")
        fake_mjlab_math.subtract_frame_transforms = lambda p1, q1, p2, q2: (
            _t.zeros((p1.shape[0], 3), dtype=_t.float32),
            _t.tensor([[1.0, 0.0, 0.0, 0.0]] * p1.shape[0], dtype=_t.float32),
        )
        fake_mjlab_math.matrix_from_quat = lambda q: _t.eye(3, dtype=_t.float32).unsqueeze(0).expand(q.shape[0], -1, -1)

        fake_cfg = types.SimpleNamespace(
            scene=types.SimpleNamespace(num_envs=1),
            commands={"motion": types.SimpleNamespace(motion_file="")},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parts = Path(tmpdir) / "parts"
            # Pre-seed clip_d in parts_dir to test resume skipping
            parts.mkdir(parents=True)
            cached_res = {
                "clip": "clip_d",
                "normalized_return": 0.42,
                "failed": False,
                "steps": 4,
                "alive_steps": 4,
                "fall_step": None,
                "fall_terms": [],
                "mean_term_score": {},
                "wall_s": 0.01,
                "device": "cpu",
                "performance": {},
                "runtime": {},
            }
            (parts / "clip_d.json").write_text(
                json.dumps({"clip": "clip_d", "metadata": {"tag": "v2"}, "result": cached_res}),
                encoding="utf-8",
            )

            modules_patch = {
                "mjlab": types.ModuleType("mjlab"),
                "mjlab.envs": fake_mjlab_envs,
                "mjlab.tasks": types.ModuleType("mjlab.tasks"),
                "mjlab.tasks.tracking": types.ModuleType("mjlab.tasks.tracking"),
                "mjlab.tasks.tracking.mdp": types.ModuleType("mjlab.tasks.tracking.mdp"),
                "mjlab.tasks.tracking.mdp.commands": fake_mjlab_cmds,
                "mjlab.tasks.tracking.mdp.observations": fake_mjlab_obs,
                "mjlab.utils": types.ModuleType("mjlab.utils"),
                "mjlab.utils.lab_api": types.ModuleType("mjlab.utils.lab_api"),
                "mjlab.utils.lab_api.math": fake_mjlab_math,
            }
            try:
                import mujoco  # noqa: F401, PLC0415
            except ImportError:
                modules_patch["mujoco"] = types.ModuleType("mujoco")
            with mock.patch.dict(sys.modules, modules_patch), \
                 mock.patch.object(B.RT, "probe_cuda", return_value={"torch_device": "cpu"}), \
                 mock.patch.object(B.RT, "require_cuda_device", side_effect=lambda d: "cpu"), \
                 mock.patch.object(B.RT, "inspect_runtime", return_value={"step_graph_captured": True}), \
                 mock.patch.object(B.V2CFG, "build_v2_env_cfg", return_value=fake_cfg), \
                 mock.patch.object(B, "get_reference", side_effect=lambda p, use_cache=False: refs[Path(p).stem]), \
                 mock.patch.object(B, "_sensor_adr", return_value=0):
                results, meta = B.evaluate_batched(
                    info, CustomPolicy, ["clip_a", "clip_b", "clip_c", "clip_d"],
                    device="cpu", batch_envs=2, checkpoint_dir=parts,
                    recovery_metadata={"tag": "v2"},
                )

            # Results must preserve exact requested clip order
            self.assertEqual([r.clip for r in results], ["clip_a", "clip_b", "clip_c", "clip_d"])
            # clip_d came from cache
            self.assertAlmostEqual(results[3].normalized_return, 0.42)
            self.assertEqual(meta["resumed_clips"], 1)
            self.assertEqual(meta["evaluated_clips"], 3)
            # clip_b (9 steps, 0.08/step -> 0.72 / 0.9 = 0.8)
            self.assertFalse(results[1].failed)
            self.assertEqual(results[1].alive_steps, 9)
            self.assertAlmostEqual(results[1].normalized_return, 0.8, places=5)
            # clip_c failed at step 3 (3 * 0.08 = 0.24 / (7 * 0.1) = 0.342857...)
            self.assertTrue(results[2].failed)
            self.assertEqual(results[2].fall_step, 3)
            self.assertEqual(results[2].fall_terms, ("root_height",))
            self.assertAlmostEqual(results[2].normalized_return, 0.24 / 0.7, places=5)
            # clip_a completed 5 steps
            self.assertFalse(results[0].failed)
            self.assertEqual(results[0].alive_steps, 5)
            self.assertAlmostEqual(results[0].normalized_return, 0.8, places=5)
            # Part files written for all clips
            for cid in ["clip_a", "clip_b", "clip_c", "clip_d"]:
                self.assertTrue((parts / f"{cid}.json").exists())

    def test_fast_eval_backend_dispatch(self):
        from scripts.v2 import fast_eval  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.json"
            fake_record = {
                "protocol_id": S.PROTOCOL_ID,
                "evaluation": {"aggregate_normalized_return": 0.5, "failed_clips": 0},
            }
            fake_summary = {
                "backend": "batched", "workers": 1, "batch_envs": 2, "policy_backend": "test",
                "clips": 2, "total_control_steps": 100, "executed_control_steps": 100,
                "eval_wall_s": 1.0, "rollout_wall_s": 0.8, "aggregate_steps_per_s": 100.0,
                "steady_steps": 90, "steady_wall_s": 0.7, "steady_aggregate_steps_per_s": 128.0,
                "gpu_utilization_pct": {},
            }
            # 1. Default (--workers auto, no --backend) -> routes to batched (Option B)
            with mock.patch.object(fast_eval, "run_batched", return_value=(fake_record, fake_summary)) as m_b, \
                 mock.patch.object(fast_eval, "run_parallel") as m_w:
                rc = fast_eval.main([
                    "--policy-file", "benchmark/policies/replay.py:make_policy",
                    "--clips", "walk1_subject1", "dance1_subject1",
                    "--out", str(out_path), "--quiet",
                ])
                self.assertEqual(rc, 0)
                self.assertEqual(m_b.call_count, 1)
                self.assertEqual(m_w.call_count, 0)

            # 2. Explicit --workers 4 (no --backend) -> routes to workers (Option A)
            with mock.patch.object(fast_eval, "run_batched") as m_b, \
                 mock.patch.object(fast_eval, "run_parallel", return_value=(fake_record, fake_summary)) as m_w:
                rc = fast_eval.main([
                    "--policy-file", "benchmark/policies/replay.py:make_policy",
                    "--workers", "4",
                    "--clips", "walk1_subject1", "dance1_subject1",
                    "--out", str(out_path), "--quiet",
                ])
                self.assertEqual(rc, 0)
                self.assertEqual(m_b.call_count, 0)
                self.assertEqual(m_w.call_count, 1)
                self.assertEqual(m_w.call_args[0][0].workers, 4)


def _gpu_stack_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
        import mjlab  # noqa: F401, PLC0415
    except Exception as exc:
        return False, f"BLOCKED: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return False, "BLOCKED: torch.cuda.is_available() == False"
    try:
        B.RT.probe_cuda("cuda:0")
    except Exception as exc:
        return False, f"BLOCKED: {type(exc).__name__}: {exc}"
    return True, ""


GPU_STACK_OK, GPU_STACK_REASON = _gpu_stack_status()


@unittest.skipUnless(GPU_STACK_OK, GPU_STACK_REASON or "BLOCKED: CUDA/mjlab unavailable")
class BatchedGpuIntegrationTests(unittest.TestCase):
    """Tier-2 real CUDA physics parity tests for Option B batched evaluation."""

    def test_batched_replay_matches_serial_on_gpu(self):
        from benchmark.evaluator import spec as v1spec  # noqa: PLC0415
        from benchmark.policies.replay import ReplayPolicy  # noqa: PLC0415
        from benchmark.v2 import runner as v2runner  # noqa: PLC0415

        clips = ["walk1_subject1", "dance1_subject1"]
        csv_ok = all((C.DATA_DIR / f"{c}.csv").exists() for c in clips)
        if not csv_ok:
            self.skipTest("BLOCKED: LAFAN1 CSVs not present")

        info = v1spec.build_model()
        # 1. Pre-fall window (30 steps): verify tight physics + reward parity.
        serial_pre = v2runner.evaluate_all(
            info, ReplayPolicy(), clips, use_cache=True, device="cuda:0", max_steps=30,
        )
        batched_pre, meta = B.evaluate_batched(
            info, ReplayPolicy, clips, use_cache=True, device="cuda:0",
            batch_envs=2, policy_device="cuda", max_steps=30,
        )
        self.assertEqual(meta["batch_envs"], 2)
        self.assertEqual([r.clip for r in batched_pre], clips)
        for rs, rb in zip(serial_pre, batched_pre):
            self.assertEqual(rb.failed, rs.failed, rs.clip)
            self.assertEqual(rb.alive_steps, rs.alive_steps, rs.clip)
            self.assertEqual(rb.fall_terms, rs.fall_terms, rs.clip)
            self.assertLess(abs(rb.normalized_return - rs.normalized_return), 5e-4, rs.clip)

        # 2. Fall window (80 steps): mjwarp GPU contact-solver atomics can shift
        #    a boundary-threshold fall by ±1 step even across two serial runs.
        serial = v2runner.evaluate_all(
            info, ReplayPolicy(), clips, use_cache=True, device="cuda:0", max_steps=80,
        )
        batched, _ = B.evaluate_batched(
            info, ReplayPolicy, clips, use_cache=True, device="cuda:0",
            batch_envs=2, policy_device="cuda", max_steps=80,
        )
        for rs, rb in zip(serial, batched):
            self.assertEqual(rb.failed, rs.failed, rs.clip)
            self.assertLessEqual(abs(rb.alive_steps - rs.alive_steps), 1, rs.clip)
            self.assertEqual(rb.fall_terms, rs.fall_terms, rs.clip)
            self.assertLess(abs(rb.normalized_return - rs.normalized_return), 5e-3, rs.clip)


if __name__ == "__main__":
    unittest.main()
