"""Unit and parity tests for the official ``HybridRobotics/supermimic`` V2 adapter.

These tests verify:
1. ``OfficialSuperMimicPolicy`` tensor parsing, shape checks, contract checks,
   exact ``EmpiricalNormalization`` (``eps=1e-2`` added to ``_std``),
   ``LayerNorm(1e-5) -> GELU``, strided goal window (``stride=5``, ``10``
   frames, clamped at clip end), and observation/goal algebraic reconstruction
   without privileged state.
2. Multi-clip dynamic slot-reassignment parity between ``OfficialSuperMimicPolicy``
   and ``BatchedOfficialSuperMimicPolicy`` (both NumPy CPU and PyTorch GPU/CPU).
3. Exact parity against a PyTorch ``nn.Sequential(Linear, LayerNorm, GELU)``
   reference when ``torch`` is installed (e.g. on Colab G4).
4. Closed-loop ``TrackingEnv`` observation and goal reconstruction against
   simulator ``MjData`` when the LAFAN1 dataset is present.
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
from benchmark.evaluator import mathx, motion, policy_api, spec  # noqa: E402
from benchmark.evaluator.motion import MotionReference  # noqa: E402
from benchmark.policies import official_supermimic as osm  # noqa: E402
from benchmark.policies import supermimic as sm  # noqa: E402
from benchmark.v2 import batched as B  # noqa: E402
from benchmark.v2 import spec as S  # noqa: E402

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SHORT_CLIP = "fallAndGetUp3_subject1"
SHORT_CSV = C.DATA_DIR / f"{SHORT_CLIP}.csv"


def _make_official_tensors(
    seed: int = 0,
    hidden_dims: tuple[int, ...] = (64, 48, 40, 32),
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t: dict[str, np.ndarray] = {}

    def _fill_mlp(prefix: str, in_dim: int, out_dim: int) -> None:
        curr = in_dim
        idx = 0
        for h in hidden_dims:
            t[f"{prefix}.{idx}.weight"] = rng.normal(0.0, 0.1, (h, curr)).astype(np.float32)
            t[f"{prefix}.{idx}.bias"] = rng.normal(0.0, 0.03, (h,)).astype(np.float32)
            t[f"{prefix}.{idx + 1}.weight"] = rng.uniform(0.7, 1.3, (h,)).astype(np.float32)
            t[f"{prefix}.{idx + 1}.bias"] = rng.normal(0.0, 0.05, (h,)).astype(np.float32)
            curr = h
            idx += 3
        t[f"{prefix}.{idx}.weight"] = rng.normal(0.0, 0.1, (out_dim, curr)).astype(np.float32)
        t[f"{prefix}.{idx}.bias"] = rng.normal(0.0, 0.03, (out_dim,)).astype(np.float32)

    _fill_mlp("encoders.motion", osm.GOAL_DIM, osm.Z_DIM)
    _fill_mlp("decoder", osm.Z_DIM + osm.HIST * osm.OBS_DIM, osm.NUM_J)

    t["obs_normalizer._mean"] = rng.normal(0.0, 0.15, (1, osm.OBS_DIM)).astype(np.float32)
    t["obs_normalizer._std"] = rng.uniform(0.4, 2.0, (1, osm.OBS_DIM)).astype(np.float32)
    t["goal_normalizers.motion._mean"] = rng.normal(0.0, 0.15, (1, osm.GOAL_DIM)).astype(np.float32)
    t["goal_normalizers.motion._std"] = rng.uniform(0.4, 2.0, (1, osm.GOAL_DIM)).astype(np.float32)
    return t


def _synthetic_ref(clip_id: str, num_frames: int, seed: int) -> MotionReference:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, (num_frames - 1) * C.STEP_DT, num_frames, dtype=np.float64)
    q0 = osm.default_joint_pos_reference()
    joint_pos = q0[None, :] + 0.15 * np.sin(t[:, None] + rng.uniform(0, 2, (1, 29)))
    joint_vel = np.gradient(joint_pos, C.STEP_DT, axis=0)

    nbody = 16
    body_pos_w = np.zeros((num_frames, nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, nbody, 4), dtype=np.float32)
    angles = 0.3 * np.sin(1.7 * t + seed)
    angles_r = 0.15 * np.cos(1.1 * t + seed)
    body_quat_w[:, :, 0] = np.cos(angles / 2.0)[:, None]
    body_quat_w[:, :, 3] = np.sin(angles / 2.0)[:, None]
    body_quat_w[:, 0, 1] = np.sin(angles_r / 2.0) * 0.3
    body_quat_w /= np.linalg.norm(body_quat_w, axis=-1, keepdims=True)

    return MotionReference(
        clip_id=clip_id,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((num_frames, nbody, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, nbody, 3), dtype=np.float32),
        npz_path=Path(f"/tmp/{clip_id}.npz"),
    )


def _ctx(clip_id: str, ref: MotionReference | policy_api.ReferenceClip) -> policy_api.ClipContext:
    ref_view = (
        ref
        if isinstance(ref, policy_api.ReferenceClip)
        else policy_api.ReferenceClip.from_motion(ref)
    )
    return policy_api.ClipContext(
        clip=clip_id,
        joint_order=C.JOINT_ORDER,
        default_joint_pos=osm.default_joint_pos_reference(),
        action_scale=osm.action_scale_reference(),
        control_dt=C.STEP_DT,
        reference=ref_view,
    )


def _rand_step(rng, ref: MotionReference, step: int, prev_a: np.ndarray) -> policy_api.StepInput:
    k = min(step + 1, ref.num_frames - 1)
    q_rob = ref.joint_pos[k].astype(np.float64) + rng.normal(0.0, 0.08, 29)
    r_pelvis_rob = mathx.matrix_from_quat(mathx.normalize(rng.normal(size=4)))
    r_torso_rob = r_pelvis_rob @ sm.waist_rotation(q_rob)
    r_pelvis_ref = mathx.matrix_from_quat(ref.body_quat_w[k, 0].astype(np.float64))
    r_torso_ref = r_pelvis_ref @ sm.waist_rotation(ref.joint_pos[k].astype(np.float64))
    ori_b = sm.mat_to_6d(r_torso_rob.T @ r_torso_ref)
    return policy_api.StepInput(
        step=step,
        reference_index=k,
        time_s=step * C.STEP_DT,
        joint_pos=q_rob,
        joint_vel=ref.joint_vel[k].astype(np.float64) + rng.normal(0.0, 0.3, 29),
        base_lin_vel=rng.normal(0.0, 0.2, 3),
        base_ang_vel=rng.normal(0.0, 0.2, 3),
        anchor_pos_b=rng.normal(0.0, 0.1, 3),
        anchor_ori_b=ori_b,
        previous_action=prev_a.copy(),
        reference=policy_api.ReferenceClip.from_motion(ref),
        anchor_pos_b_prev=rng.normal(0.0, 0.1, 3),
        anchor_ori_b_prev=np.zeros(6, dtype=np.float64),
    )


class OfficialAlgebraicTests(unittest.TestCase):
    def test_observation_and_goal_match_world_frame_ground_truth(self):
        rng = np.random.default_rng(7)
        ref = _synthetic_ref("alg_clip", 30, seed=5)
        pol = osm.OfficialSuperMimicPolicy(_make_official_tensors(seed=1))
        ctx = _ctx("alg_clip", ref)
        pol.reset(ctx)

        for step_idx in (0, 1, 5, 25, 28):
            k = min(step_idx + 1, ref.num_frames - 1)
            q_rob = ref.joint_pos[k].astype(np.float64) + rng.normal(0.0, 0.15, 29)
            q_rob[list(sm._WAIST_IDX)] += rng.normal(0.0, 0.3, 3)  # noqa: SLF001
            r_pelvis_rob = mathx.matrix_from_quat(mathx.normalize(rng.normal(size=4)))
            r_torso_rob = r_pelvis_rob @ sm.waist_rotation(q_rob)

            r_ref_all = mathx.matrix_from_quat(ref.body_quat_w[:, 0].astype(np.float64))
            r_torso_ref_k = r_ref_all[k] @ sm.waist_rotation(ref.joint_pos[k].astype(np.float64))
            ori_b = sm.mat_to_6d(r_torso_rob.T @ r_torso_ref_k)
            prev_a = rng.normal(0.0, 0.5, 29)
            qd = rng.normal(0.0, 1.0, 29)
            ang_vel = rng.normal(0.0, 0.5, 3)

            step = policy_api.StepInput(
                step=step_idx,
                reference_index=k,
                time_s=step_idx * C.STEP_DT,
                joint_pos=q_rob,
                joint_vel=qd,
                base_lin_vel=np.zeros(3),
                base_ang_vel=ang_vel,
                anchor_pos_b=np.zeros(3),
                anchor_ori_b=ori_b,
                previous_action=prev_a,
                reference=ctx.reference,
                anchor_pos_b_prev=np.zeros(3),
                anchor_ori_b_prev=np.zeros(6),
            )

            obs = pol.observation(step)
            self.assertEqual(obs.shape, (osm.OBS_DIM,))
            expected_grav = -r_pelvis_rob[2, :]
            expected_obs = np.concatenate(
                [expected_grav, ang_vel, q_rob - ctx.default_joint_pos, qd, prev_a]
            )
            np.testing.assert_allclose(obs, expected_obs, atol=1e-12)

            pelvis_rel = pol._pelvis_rel_at(step, k)  # noqa: SLF001
            goal = pol.goal(k, pelvis_rel)
            self.assertEqual(goal.shape, (osm.GOAL_DIM,))
            idx = np.minimum(k + np.arange(osm.GOAL_FRAMES) * osm.GOAL_STRIDE, ref.num_frames - 1)
            expected_content = np.concatenate([ref.joint_pos[idx], ref.joint_vel[idx]], axis=-1).reshape(-1)
            expected_rel_ori = sm.mat_to_6d(
                np.einsum("ji,kjl->kil", r_pelvis_rob, r_ref_all[idx])
            ).reshape(-1)
            expected_goal = np.concatenate([expected_content, expected_rel_ori])
            np.testing.assert_allclose(goal, expected_goal, atol=1e-12)

    def test_history_seeding_sliding_and_reset(self):
        rng = np.random.default_rng(11)
        ref = _synthetic_ref("hist_clip", 12, seed=2)
        pol = osm.OfficialSuperMimicPolicy(_make_official_tensors(seed=3))
        pol.reset(_ctx("hist_clip", ref))
        self.assertIsNone(pol._hist)  # noqa: SLF001

        st0 = _rand_step(rng, ref, 0, np.zeros(29))
        a0 = pol.act(st0)
        self.assertEqual(pol._hist.shape, (osm.HIST, osm.OBS_DIM))  # noqa: SLF001
        obs0 = pol.observation(st0)
        np.testing.assert_allclose(pol._hist, np.repeat(obs0[None, :], osm.HIST, axis=0), atol=1e-14)  # noqa: SLF001

        st1 = _rand_step(rng, ref, 1, a0)
        _ = pol.act(st1)
        obs1 = pol.observation(st1)
        np.testing.assert_allclose(pol._hist[:-1], np.repeat(obs0[None, :], osm.HIST - 1, axis=0), atol=1e-14)  # noqa: SLF001
        np.testing.assert_allclose(pol._hist[-1], obs1, atol=1e-14)  # noqa: SLF001

        pol.reset(_ctx("hist_clip", ref))
        self.assertIsNone(pol._hist)  # noqa: SLF001
        a0_again = pol.act(st0)
        np.testing.assert_array_equal(a0, a0_again)

    def test_contract_and_missing_tensor_validation(self):
        bad = _make_official_tensors()
        del bad["goal_normalizers.motion._std"]
        with self.assertRaises(sm.SuperMimicContractError):
            osm.OfficialSuperMimicPolicy(bad)

        pol = osm.OfficialSuperMimicPolicy(_make_official_tensors())
        ref = _synthetic_ref("c_clip", 10, seed=0)
        bad_ctx = policy_api.ClipContext(
            clip="c_clip",
            joint_order=C.JOINT_ORDER,
            default_joint_pos=osm.default_joint_pos_reference(),
            action_scale=osm.action_scale_reference() * 1.05,
            control_dt=C.STEP_DT,
            reference=policy_api.ReferenceClip.from_motion(ref),
        )
        with self.assertRaises(sm.SuperMimicContractError):
            pol.reset(bad_ctx)

    def test_protocol_sha256_untouched(self):
        self.assertEqual(
            S.protocol_sha256(),
            "e5a5c0335fcc239edae4a085eb66b9ac5b93d4c44917c7168d21761c12675ba0",
        )


class OfficialBatchedParityTests(unittest.TestCase):
    def _run_parity(self, use_torch: bool):
        rng = np.random.default_rng(99)
        refs = [
            ("clip_long", _synthetic_ref("clip_long", 60, seed=10)),
            ("clip_short", _synthetic_ref("clip_short", 8, seed=11)),
            ("clip_mid", _synthetic_ref("clip_mid", 25, seed=12)),
        ]
        bank = B.PackedMotionBank(refs)
        tensors = _make_official_tensors(seed=42)
        base = osm.OfficialSuperMimicPolicy(tensors)
        q0 = osm.default_joint_pos_reference()
        scale = osm.action_scale_reference()

        batched = B.BatchedOfficialSuperMimicPolicy(
            base,
            bank,
            num_slots=2,
            default_joint_pos=q0,
            action_scale=scale,
            control_dt=C.STEP_DT,
            device="cpu",
            use_torch=use_torch,
        )

        # Serial ground truth per clip
        serial_seq: dict[str, list[tuple[policy_api.StepInput, np.ndarray]]] = {}
        for cid, ref in refs:
            pol = osm.OfficialSuperMimicPolicy(tensors)
            pol.reset(_ctx(cid, ref))
            prev_a = np.zeros(29, dtype=np.float64)
            seq = []
            for s in range(ref.num_frames - 1):
                st = _rand_step(rng, ref, s, prev_a)
                act = pol.act(st)
                seq.append((st, act))
                prev_a = act
            serial_seq[cid] = seq

        # Simulate 2 slots: clip 0 & clip 1 start; when clip 1 ends, slot 1 switches to clip 2
        slot_cidx = [0, 1]
        slot_step = [0, 0]
        batched.reset_slots([0, 1], slot_cidx)

        max_err = 0.0
        while slot_cidx[0] is not None or slot_cidx[1] is not None:
            gframes = np.array(
                [
                    (bank.metas[c].start + min(slot_step[e] + 1, bank.metas[c].num_frames - 1))
                    if c is not None else 0
                    for e, c in enumerate(slot_cidx)
                ],
                dtype=np.int64,
            )
            steps = [
                serial_seq[bank.metas[c].clip_id][slot_step[e]][0] if c is not None else None
                for e, c in enumerate(slot_cidx)
            ]

            def _stack(getter, dim):
                arr = np.zeros((2, dim), dtype=np.float64)
                for e, st in enumerate(steps):
                    if st is not None:
                        arr[e] = getter(st)
                return arr

            q_b = _stack(lambda st: st.joint_pos, 29)
            qd_b = _stack(lambda st: st.joint_vel, 29)
            ang_b = _stack(lambda st: st.base_ang_vel, 3)
            ori_b = _stack(lambda st: st.anchor_ori_b, 6)
            pa_b = _stack(lambda st: st.previous_action, 29)

            if use_torch:
                with torch.no_grad():
                    acts = batched.act_gpu(
                        global_frames=torch.from_numpy(gframes),
                        joint_pos=torch.from_numpy(q_b).float(),
                        joint_vel=torch.from_numpy(qd_b).float(),
                        base_ang_vel=torch.from_numpy(ang_b).float(),
                        anchor_ori_b=torch.from_numpy(ori_b).float(),
                        prev_actions=torch.from_numpy(pa_b).float(),
                    ).cpu().numpy()
                tol = 1e-4  # float32 StepInput cast in GPU path vs float64 serial inputs
            else:
                acts = batched.act_cpu(
                    global_frames_np=gframes,
                    joint_pos=q_b,
                    joint_vel=qd_b,
                    base_ang_vel=ang_b,
                    anchor_ori_b=ori_b,
                    prev_actions=pa_b,
                )
                tol = 1e-6

            for e, c in enumerate(slot_cidx):
                if c is None:
                    continue
                expected = serial_seq[bank.metas[c].clip_id][slot_step[e]][1]
                err = float(np.abs(acts[e] - expected).max())
                max_err = max(max_err, err)
                self.assertLess(err, tol, f"slot={e} clip={c} step={slot_step[e]} err={err}")
                slot_step[e] += 1
                if slot_step[e] >= bank.metas[c].n_steps:
                    if c == 1:
                        slot_cidx[e] = 2
                        slot_step[e] = 0
                        batched.reset_slots([e], [2])
                    else:
                        slot_cidx[e] = None

        self.assertLess(max_err, 1e-4 if use_torch else 1e-6)

    def test_batched_cpu_parity(self):
        self._run_parity(use_torch=False)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
    def test_batched_torch_parity(self):
        self._run_parity(use_torch=True)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
    def test_numpy_forward_matches_pytorch_reference(self):
        tensors = _make_official_tensors(seed=17)
        pol = osm.OfficialSuperMimicPolicy(tensors)

        def _build_torch_mlp(prefix: str, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
            layers: list[nn.Module] = []
            curr = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
                curr = h
            layers.append(nn.Linear(curr, out_dim))
            mod = nn.Sequential(*layers)
            sd = {k[len(prefix) + 1 :]: torch.from_numpy(v) for k, v in tensors.items() if k.startswith(prefix + ".")}
            mod.load_state_dict(sd)
            return mod.eval()

        enc = _build_torch_mlp("encoders.motion", osm.GOAL_DIM, (64, 48, 40, 32), osm.Z_DIM)
        dec = _build_torch_mlp("decoder", osm.Z_DIM + osm.HIST * osm.OBS_DIM, (64, 48, 40, 32), osm.NUM_J)
        obs_m = torch.from_numpy(tensors["obs_normalizer._mean"].reshape(-1))
        obs_s = torch.from_numpy(tensors["obs_normalizer._std"].reshape(-1))
        goal_m = torch.from_numpy(tensors["goal_normalizers.motion._mean"].reshape(-1))
        goal_s = torch.from_numpy(tensors["goal_normalizers.motion._std"].reshape(-1))

        rng = np.random.default_rng(31)
        for _ in range(5):
            hist_np = rng.normal(0.0, 0.8, (osm.HIST, osm.OBS_DIM)).astype(np.float32)
            goal_np = rng.normal(0.0, 0.8, (osm.GOAL_DIM,)).astype(np.float32)
            out_np = pol.forward(hist_np, goal_np)
            with torch.no_grad():
                gn = (torch.from_numpy(goal_np).unsqueeze(0) - goal_m) / (goal_s + 1e-2)
                z = torch.tanh(enc(gn))
                hn = ((torch.from_numpy(hist_np).unsqueeze(0) - obs_m) / (obs_s + 1e-2)).flatten(start_dim=1)
                out_pt = dec(torch.cat([z, hn], dim=-1)).squeeze(0).numpy()
            np.testing.assert_allclose(out_np, out_pt, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(SHORT_CSV.exists(), "LAFAN1 dataset not present")
class OfficialSimulatorReconstructionTest(unittest.TestCase):
    """Compare rebuilt 93-dim obs & 640-dim goal directly against ``TrackingEnv`` ``MjData``."""

    def test_matches_mjdata_over_rollout(self):
        from benchmark.evaluator.env import TrackingEnv

        info = spec.build_model()
        ref = motion.get_reference(SHORT_CSV)
        env = TrackingEnv(ref)
        pol = osm.OfficialSuperMimicPolicy(_make_official_tensors(seed=0))
        pol.reset(_ctx(SHORT_CLIP, env.reference_view))

        worst_obs = 0.0
        worst_goal = 0.0
        ref_r_all = mathx.matrix_from_quat(ref.body_quat_w[:, 0].astype(np.float64))

        for _ in range(30):
            step = env.step_input()
            k = step.reference_index
            rebuilt_obs = pol.observation(step)
            rebuilt_goal = pol.goal(k, pol._pelvis_rel_at(step, k))  # noqa: SLF001

            qpos = env._env.sim.data.qpos[0].cpu().numpy()  # noqa: SLF001
            qvel = env._env.sim.data.qvel[0].cpu().numpy()  # noqa: SLF001
            sensordata = env._env.sim.data.sensordata[0].cpu().numpy()  # noqa: SLF001
            r_pelvis = mathx.matrix_from_quat(
                env._env.scene["robot"].data.body_link_quat_w[0, 0].cpu().numpy()  # noqa: SLF001
            )
            gt_obs = np.concatenate(
                [
                    -r_pelvis[2, :],
                    sensordata[info.imu_ang_vel_adr : info.imu_ang_vel_adr + 3],
                    qpos[info.qpos_adr] - info.default_joint_pos,
                    qvel[info.dof_adr],
                    env._action,  # noqa: SLF001
                ]
            )
            worst_obs = max(worst_obs, float(np.abs(rebuilt_obs - gt_obs).max()))

            idx = np.minimum(k + np.arange(osm.GOAL_FRAMES) * osm.GOAL_STRIDE, ref.num_frames - 1)
            gt_content = np.concatenate([ref.joint_pos[idx], ref.joint_vel[idx]], axis=-1).reshape(-1)
            gt_rel_ori = sm.mat_to_6d(
                np.einsum("ji,kjl->kil", r_pelvis, ref_r_all[idx])
            ).reshape(-1)
            gt_goal = np.concatenate([gt_content, gt_rel_ori])
            worst_goal = max(worst_goal, float(np.abs(rebuilt_goal - gt_goal).max()))

            target = ref.joint_pos[min(k, ref.num_frames - 1)]
            if env.step((target - info.default_joint_pos) / info.action_scale).terminated:
                break

        self.assertLess(worst_obs, 1e-5, f"obs mismatch {worst_obs:.2e}")
        self.assertLess(worst_goal, 1e-5, f"goal mismatch {worst_goal:.2e}")


if __name__ == "__main__":
    unittest.main()
