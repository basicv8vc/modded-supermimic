"""Benchmark V2: protocol, math, source-level and configuration parity tests.

Everything here is GPU-free.  Three tiers:

* always run (numpy + stdlib only): V1 preservation, V2 identity/protocol,
  executable math, normalization, V2 record validation, frozen oracle snapshot;
* run when mjlab 1.6.0 is importable: static cfg verification + glue math
  consistency (CPU only, no MuJoCo Warp / CUDA);
* run when the two pinned upstream checkouts *and* torch/jax/mjx are present:
  dynamic parity against the real SuperMimic / NablaMimic source code.

``V2_SUPERMIMIC_ROOT`` / ``V2_NABLAMIMIC_ROOT`` select the checkouts;
``scripts/v2/fetch_upstream.py`` fetches them at the pinned commits.
"""

from __future__ import annotations

import json
import math
import random
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from benchmark import core as v1_core  # noqa: E402
from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.v2 import config as V2C  # noqa: E402
from benchmark.v2 import spec as S  # noqa: E402
import v2_support as O  # noqa: E402

REAL_ORACLES = O.real_oracle_available()
SNAPSHOT = O.load_snapshot()


def _mjlab_available() -> bool:
    try:
        import mjlab  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. V1 must be preserved verbatim
# ---------------------------------------------------------------------------

class V1PreservationTests(unittest.TestCase):
    def test_v1_protocol_id_and_sha_unchanged(self):
        self.assertEqual(S.load_v1_protocol()["protocol_id"], "modded-supermimic-g1-lafan1-v1")
        self.assertEqual(S.v1_protocol_sha256(), S.V1_PROTOCOL_SHA256)

    def test_v1_constants_still_have_its_own_rubric(self):
        self.assertEqual(C.EVALUATOR_IMPLEMENTATION, "canonical-mjlab-g1-v1")
        self.assertIn("self_collisions", C.REWARDS)
        self.assertEqual(C.REWARDS["self_collisions"]["weight"], -10.0)
        self.assertEqual(C.REWARDS["joint_limit"]["weight"], -10.0)
        self.assertEqual(C.TERM_EE_BODY_POS_THRESHOLD, 0.25)

    def test_v1_runner_still_clamps_and_core_still_rejects_out_of_range(self):
        runner_src = (REPO / "benchmark/evaluator/runner.py").read_text()
        self.assertIn("min(max(raw, 0.0), 1.0)", runner_src)
        rec = v1_core.audit_record(_v1_record(1.4))
        self.assertIn("normalized_return must be in [0,1]", " ".join(rec["errors"]))
        self.assertIn("normalized_return must be in [0,1]",
                      " ".join(v1_core.audit_record(_v1_record(-0.2))["errors"]))

    def test_v1_historical_reports_still_present(self):
        for name in ("supermimic_canonical_eval_cli.json",
                     "supermimic_canonical_eval_mjlab_reuse.json",
                     "nablamimic_canonical_eval_mjlab.json"):
            self.assertTrue((REPO / "reports" / name).is_file(), name)


def _v1_record(ret: float) -> dict:
    ids = S.load_v1_protocol()["clip_ids"]
    return {
        "protocol_id": "modded-supermimic-g1-lafan1-v1",
        "algorithm": "test", "training_commit": "a" * 40, "seed": 0,
        "hardware": {"gpu_name": "RTX PRO 6000 Blackwell", "gpu_count": 1},
        "evaluator": {"implementation": "canonical-mjlab-g1-v1", "mujoco": "3.11.0",
                      "mujoco_warp": "3.11.0", "warp_lang": "1.16.0", "robot_dof": 29,
                      "self_collision": True, "mjlab_commit": "a" * 40,
                      "robot_model_sha256": "b" * 64, "dataset_manifest_sha256": "b" * 64},
        "checkpoints": [{"checkpoint_sha256": "b" * 64,
                         "training_wall_s": 1.0, "setup_wall_s": 1.0, "eval_wall_s": 1.0,
                         "per_clip": [{"clip": c, "normalized_return": ret, "failed": False} for c in ids]}],
    }


# ---------------------------------------------------------------------------
# 2. V2 identity / protocol
# ---------------------------------------------------------------------------

class V2IdentityTests(unittest.TestCase):
    def test_protocol_id_is_independent(self):
        cfg = S.load_protocol_v2()
        self.assertEqual(cfg["protocol_id"], "modded-supermimic-g1-lafan1-v2")
        self.assertNotEqual(cfg["protocol_id"], S.load_v1_protocol()["protocol_id"])
        self.assertIsNone(cfg["supersedes"])
        self.assertFalse(cfg["v1_scores_comparable"])

    def test_protocol_sha256_is_stable_and_distinct(self):
        sha = S.protocol_sha256()
        self.assertEqual(len(sha), 64)
        self.assertNotEqual(sha, S.v1_protocol_sha256())

    def test_evaluator_identity_is_independent(self):
        ident = S.evaluator_identity()
        self.assertEqual(ident["evaluator_implementation"], "canonical-mjlab-g1-v2-original-aligned")
        self.assertNotEqual(ident["evaluator_implementation"], C.EVALUATOR_IMPLEMENTATION)
        self.assertEqual(ident["protocol_sha256"], S.protocol_sha256())
        self.assertEqual(ident["v1_protocol_sha256"], S.V1_PROTOCOL_SHA256)

    def test_dependency_on_pr12_and_pr13_is_recorded(self):
        deps = S.load_protocol_v2()["depends_on_prs"]
        self.assertEqual([d["number"] for d in deps], [12, 13])
        self.assertEqual(deps[0]["head"], "f5bf4e0d881837b84ecb0654c88d7d0d3bfc361e")
        self.assertEqual(deps[1]["head"], "66a6e52664195fee3c446924980ae586bc55c25d")

    def test_pinned_reference_commits(self):
        refs = S.load_protocol_v2()["reference_implementations"]
        self.assertEqual(refs["supermimic"]["commit"], O.SUPERMIMIC_COMMIT)
        self.assertEqual(refs["nablamimic"]["commit"], O.NABLAMIMIC_COMMIT)

    def test_same_40_clips_as_v1(self):
        self.assertEqual(S.load_protocol_v2()["clip_ids"], S.load_v1_protocol()["clip_ids"])

    def test_registry_records_are_audited_and_valid(self):
        reg = json.loads((REPO / "reports/v2/registry.json").read_text())
        self.assertEqual(reg["status"], "G4_BASELINE_EVALUATION_COMPLETE")
        self.assertTrue(reg["target_hardware"]["met"])
        acc = json.loads((REPO / "reports/v2/gpu_runtime_status.json").read_text())
        self.assertEqual(acc["acceptance_status"], "PASS")
        self.assertGreaterEqual(len(reg["records"]), 2)
        for entry in reg["records"]:
            rec_path = REPO / "reports/v2" / entry["path"]
            self.assertTrue(rec_path.is_file(), entry["path"])
            rec = json.loads(rec_path.read_text())
            audit = S.audit_v2_record(rec)
            self.assertEqual(audit["errors"], [], entry["path"])
            self.assertTrue(audit["status"].startswith("FORMAT_VALID"), entry["path"])
            self.assertFalse(audit["clipping_applied"], entry["path"])
            self.assertAlmostEqual(
                audit["aggregate_normalized_return"],
                entry["aggregate_normalized_return"],
                places=12,
            )
            self.assertEqual(audit["failed_clips"], entry["failed_clips"])
            self.assertTrue(rec.get("submission_eligible"))
            self.assertTrue(rec.get("target_hardware_met"))
            self.assertEqual(rec.get("report_scope"), "acceptance")
        record_paths = {e["path"] for e in reg["records"]}
        for diag_name in reg.get("diagnostic_artifacts", []):
            self.assertNotIn(diag_name, record_paths)
            if diag_name.startswith("diag_g4_"):
                diag = json.loads((REPO / "reports/v2" / diag_name).read_text())
                self.assertIs(diag.get("submission_eligible"), False)


class V2ProtocolSpecTests(unittest.TestCase):
    def test_reward_terms_weights_stds(self):
        cfg = S.load_protocol_v2()
        terms = cfg["reward"]["terms"]
        self.assertEqual(tuple(terms), S.REWARD_TERM_NAMES)
        for name in S.TRACKING_WEIGHTS:
            self.assertEqual(terms[name]["weight"], S.TRACKING_WEIGHTS[name])
            self.assertEqual(terms[name]["std"], S.TRACKING_STD[name])
        self.assertEqual(terms["action_rate_l2"]["weight"], -0.1)
        self.assertEqual(terms["joint_limit"]["weight"], -2.0)
        self.assertEqual(terms["joint_limit"]["soft_limit_factor"], 0.9)

    def test_no_self_collision_penalty_but_full_self_collision_physics(self):
        cfg = S.load_protocol_v2()
        self.assertIn("self_collisions", cfg["reward"]["excluded_terms"])
        self.assertNotIn("self_collisions", cfg["reward"]["terms"])
        self.assertTrue(cfg["full_self_collision"])
        self.assertFalse(cfg["self_collision_reward_penalty"])

    def test_termination_terms_and_thresholds(self):
        cfg = S.load_protocol_v2()
        self.assertEqual(set(cfg["termination"]["terms"]), set(S.TERMINATION_TERM_NAMES))
        self.assertNotIn("ee_body_pos", cfg["termination"]["terms"])
        self.assertIn("ee_body_pos", cfg["termination"]["excluded_terms"])
        self.assertEqual(cfg["termination"]["terms"]["root_height"]["threshold"], 0.25)
        self.assertEqual(cfg["termination"]["terms"]["root_tilt"]["threshold"], 0.8)
        self.assertEqual(cfg["termination"]["terms"]["root_height"]["body"], "pelvis")
        self.assertEqual(cfg["termination"]["terms"]["root_tilt"]["body"], "pelvis")
        self.assertEqual(cfg["termination"]["clip_end"], "success (not a failure)")

    def test_normalization_has_no_clipping(self):
        n = S.load_protocol_v2()["normalization"]
        self.assertEqual(n["clipping"], "none")
        self.assertTrue(n["negative_allowed"])
        self.assertTrue(n["above_one_allowed"])
        self.assertIn("(T - 1) * max_step_reward", n["per_clip"])
        self.assertEqual(n["denominator"], "full clip theoretical maximum (unchanged by early termination)")

    def test_max_step_reward_matches_both_originals(self):
        self.assertAlmostEqual(S.MAX_STEP_REWARD, 0.1, places=12)
        self.assertAlmostEqual(S.MAX_STEP_REWARD, (4 * 1.0 + 2 * 0.5) * 0.02, places=12)


# ---------------------------------------------------------------------------
# 3. Executable math (pure numpy)
# ---------------------------------------------------------------------------

class V2MathTests(unittest.TestCase):
    def test_quat_angle_sq_matches_supermimic_acos_trace_form(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            q1 = rng.normal(size=4); q1 /= np.linalg.norm(q1)
            q2 = rng.normal(size=4); q2 /= np.linalg.norm(q2)
            ours = float(S.quat_angle_sq(q1, q2))
            tr = float(np.trace(S.quat_to_mat(q1).T @ S.quat_to_mat(q2)))
            sm = math.acos(float(np.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6))) ** 2
            self.assertAlmostEqual(ours, sm, places=6)

    def test_quat_angle_sq_is_zero_at_identity_and_symmetric(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            a = rng.normal(size=4); a /= np.linalg.norm(a)
            b = rng.normal(size=4); b /= np.linalg.norm(b)
            self.assertLess(abs(float(S.quat_angle_sq(a, a))), 1e-10)
            self.assertAlmostEqual(float(S.quat_angle_sq(a, b)), float(S.quat_angle_sq(b, a)), places=8)

    def test_yaw_quat_matches_yaw_matrix_from_projected_x_axis(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            R = S.quat_to_mat(q)
            yaw = math.atan2(R[1, 0], R[0, 0])
            expect = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
            got = S.yaw_quat(q)[0]
            self.assertLess(np.max(np.abs(got - expect)), 1e-12)

    def test_gravity_b_z_equals_minus_R22_and_mjlab_glue(self):
        rng = np.random.default_rng(19)
        for _ in range(100):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            self.assertAlmostEqual(S.gravity_b_z(q), -S.quat_to_mat(q)[2, 2], places=12)
            self.assertAlmostEqual(S.gravity_b_z(q), float(V2C._gravity_b_z(q)), places=12)

    def test_reanchor_is_identity_at_perfect_tracking(self):
        d = O.make_case("perfect")
        rel_p, rel_q = S.reference_reanchor(O.as_state(d))
        self.assertLess(np.max(np.abs(rel_p - d["ref_body_pos"])), 1e-9)
        self.assertLess(np.max(np.abs(rel_q - d["ref_body_quat"])), 1e-9)

    def test_perfect_tracking_gives_exactly_the_ceiling(self):
        d = O.make_case("perfect")
        terms = S.reward_terms(O.as_state(d))
        self.assertAlmostEqual(sum(v for k, v in terms.items() if k in S.TRACKING_WEIGHTS),
                               S.MAX_STEP_REWARD, places=10)
        for name, w in S.TRACKING_WEIGHTS.items():
            self.assertAlmostEqual(terms[name], w * S.CONTROL_DT, places=10)

    def test_action_rate_and_joint_limit_definitions(self):
        d = O.make_case("drift")
        terms = S.reward_terms(O.as_state(d))
        expected_rate = S.CONTROL_DT * (-0.1) * float(np.sum((d["action"] - d["prev_action"]) ** 2))
        expected_lim = S.CONTROL_DT * (-2.0) * float(np.sum(
            np.maximum(d["joint_pos"] - d["soft_hi"], 0) + np.maximum(d["soft_lo"] - d["joint_pos"], 0)))
        self.assertAlmostEqual(terms["action_rate_l2"], expected_rate, places=12)
        self.assertAlmostEqual(terms["joint_limit"], expected_lim, places=12)
        self.assertLess(terms["joint_limit"], 0.0)

    def test_soft_limits_are_derived_from_hard_limits_with_factor_09(self):
        d = O.make_case()
        mid = (d["q_lo"] + d["q_hi"]) / 2
        half = (d["q_hi"] - d["q_lo"]) / 2
        np.testing.assert_allclose(mid - half * 0.9, d["soft_lo"], atol=1e-12)
        np.testing.assert_allclose(mid + half * 0.9, d["soft_hi"], atol=1e-12)

    def test_reward_is_invariant_to_common_xy_translation(self):
        """xy re-anchoring must cancel a common global xy translation exactly."""
        d = O.make_case("drift")
        base = S.reward_errors(O.as_state(d))
        d2 = dict(d)
        shift = np.array([1.3, -0.7, 0.0])
        d2["robot_body_pos"] = d["robot_body_pos"] + shift
        d2["ref_body_pos"] = d["ref_body_pos"] + shift
        got = S.reward_errors(O.as_state(d2))
        for key, value in base.items():
            self.assertAlmostEqual(got[key], value, places=12, msg=key)

    def test_reward_is_invariant_to_common_global_yaw_for_planar_state(self):
        """yaw re-anchoring cancels a common global yaw exactly for planar states.

        All three implementations use ``yaw_quat(robot_anchor * ref_anchor^-1)``;
        a left-multiplied global yaw commutes with that expression only when the
        anchor orientations differ by a pure yaw.  This is a shared property of
        the original mathematics, not something V2 changes.
        """
        def pure_yaw(a):
            return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])

        d = O.make_case("drift")
        planar = dict(d)
        planar["robot_body_quat"] = np.tile(pure_yaw(0.55), (O.B, 1))
        planar["ref_body_quat"] = np.tile(pure_yaw(0.31), (O.B, 1))
        base = S.reward_errors(O.as_state(planar))
        yaw_q = pure_yaw(0.4)
        d2 = dict(planar)
        d2["robot_body_pos"] = planar["robot_body_pos"] @ S.quat_to_mat(yaw_q).T
        d2["ref_body_pos"] = planar["ref_body_pos"] @ S.quat_to_mat(yaw_q).T
        d2["robot_body_quat"] = S.quat_mul(yaw_q[None, :], planar["robot_body_quat"])
        d2["ref_body_quat"] = S.quat_mul(yaw_q[None, :], planar["ref_body_quat"])
        got = S.reward_errors(O.as_state(d2))
        for key, value in base.items():
            self.assertAlmostEqual(got[key], value, places=12, msg=key)

    def test_tilted_anchor_yaw_extraction_is_only_approximately_invariant(self):
        """Documented shared limitation: yaw extraction is not tilt-invariant.

        With roll/pitch on the anchor, ``yaw_quat(yaw_q * a * b^-1 * yaw_q^-1)``
        differs slightly from ``yaw_quat(a * b^-1)``.  SM / NM / mjlab all use
        this form, so V2 deliberately keeps it instead of "fixing" it.
        """
        d = O.make_case("drift")
        base = S.reward_errors(O.as_state(d))["motion_body_pos"]
        yaw_q = np.array([math.cos(0.2), 0.0, 0.0, math.sin(0.2)])
        d2 = dict(d)
        d2["robot_body_pos"] = d["robot_body_pos"] @ S.quat_to_mat(yaw_q).T
        d2["ref_body_pos"] = d["ref_body_pos"] @ S.quat_to_mat(yaw_q).T
        d2["robot_body_quat"] = S.quat_mul(yaw_q[None, :], d["robot_body_quat"])
        d2["ref_body_quat"] = S.quat_mul(yaw_q[None, :], d["ref_body_quat"])
        got = S.reward_errors(O.as_state(d2))["motion_body_pos"]
        self.assertGreater(abs(got - base), 1e-12)   # not exactly invariant
        self.assertLess(abs(got - base), 1e-3)       # but a small, shared effect


# ---------------------------------------------------------------------------
# 4. Official mjlab cfg verification (static, CPU)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_mjlab_available(), "mjlab 1.6.0 not importable")
class V2ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = V2C.build_v2_env_cfg(play=True)

    def test_cfg_has_no_problems(self):
        self.assertEqual(V2C.verify_v2_cfg(self.cfg), [])

    def test_eight_official_reward_terms_with_original_weights(self):
        self.assertEqual(tuple(self.cfg.rewards), S.REWARD_TERM_NAMES)
        self.assertNotIn("self_collisions", self.cfg.rewards)
        self.assertEqual(float(self.cfg.rewards["joint_limit"].weight), -2.0)
        for name, fn in V2C.OFFICIAL_REWARD_FUNCS.items():
            self.assertEqual(self.cfg.rewards[name].func.__name__, fn)

    def test_terminations_are_the_three_v2_terms_plus_time_out(self):
        self.assertEqual(tuple(self.cfg.terminations),
                         ("time_out", "root_height", "root_tilt", "nonfinite"))
        self.assertTrue(self.cfg.terminations["time_out"].time_out)

    def test_self_collision_physics_is_kept(self):
        from mjlab.tasks.tracking.config.g1.env_cfgs import (  # noqa: PLC0415
            unitree_g1_flat_tracking_env_cfg,
        )
        official = unitree_g1_flat_tracking_env_cfg(play=True)
        self.assertEqual([s.name for s in self.cfg.scene.sensors],
                         [s.name for s in official.scene.sensors])
        self.assertEqual(set(self.cfg.scene.entities), set(official.scene.entities))
        self.assertEqual(len(self.cfg.scene.sensors), 1)
        self.assertEqual(self.cfg.scene.sensors[0].name, "self_collision")

    def test_determinism_overrides_match_pr12(self):
        self.assertEqual(self.cfg.scene.num_envs, 1)
        self.assertFalse(self.cfg.auto_reset)
        self.assertEqual(self.cfg.events, {})
        self.assertEqual(self.cfg.commands["motion"].sampling_mode, "start")
        self.assertEqual(self.cfg.commands["motion"].joint_position_range, (0.0, 0.0))
        self.assertFalse(self.cfg.observations["actor"].enable_corruption)


@unittest.skipUnless(_torch_available(), "torch not importable")
class GlueMathConsistencyTests(unittest.TestCase):
    """The pelvis glue must equal ``spec.termination`` exactly."""

    def setUp(self):
        import torch  # noqa: PLC0415
        self.torch = torch
        self.d = O.make_case("drift")

    def _fake_env(self, dz=0.0, tilt=None, nan=False):
        import torch  # noqa: PLC0415
        import types  # noqa: PLC0415
        d = self.d
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
        q = d["robot_root_quat"]
        if tilt is not None:
            q = S.mat_to_quat(S.quat_to_mat(d["robot_root_quat"]) @ O._rotation_x(tilt))
        qpos = t(np.concatenate([np.array([0.0, 0.0, d["ref_root_z"] + dz]), q, d["joint_pos"]]))[None]
        qvel = torch.zeros((1, 35), dtype=torch.float64)
        if nan:
            qpos[0, 0] = float("nan")
        sim = types.SimpleNamespace(data=types.SimpleNamespace(qpos=qpos, qvel=qvel))
        # mjlab MotionCommand buffers are (num_envs, num_bodies, 3/4).
        command = types.SimpleNamespace(
            cfg=types.SimpleNamespace(body_names=S.TRACKED_BODIES),
            body_pos_w=t(d["ref_body_pos"])[None],
            body_quat_w=t(d["ref_body_quat"])[None],
        )
        return types.SimpleNamespace(
            sim=sim,
            command_manager=types.SimpleNamespace(get_term=lambda name: command),
        )

    def test_glue_height_matches_spec_at_and_around_threshold(self):
        for dz in (0.0, 0.249, 0.25, 0.251, -0.3):
            env = self._fake_env(dz=dz)
            glue = bool(V2C.bad_root_height(env, "motion")[0])
            spec_term = O.v2_termination(self.d, dz=dz)
            self.assertEqual(glue, "root_height" in spec_term.reasons, f"dz={dz}")

    def test_glue_tilt_matches_spec(self):
        for tilt in (0.0, 0.6, 1.7):
            env = self._fake_env(tilt=tilt)
            glue = bool(V2C.bad_root_tilt(env, "motion")[0])
            spec_term = O.v2_termination(self.d, tilt=tilt)
            self.assertEqual(glue, "root_tilt" in spec_term.reasons, f"tilt={tilt}")

    def test_glue_nonfinite_matches_spec(self):
        env = self._fake_env(nan=True)
        self.assertTrue(bool(V2C.bad_nonfinite(env)[0]))
        self.assertIn("nonfinite", O.v2_termination(self.d, nan=True).reasons)
        self.assertFalse(bool(V2C.bad_nonfinite(self._fake_env())[0]))


# ---------------------------------------------------------------------------
# 5. Reward parity -- frozen oracle snapshot (always runs)
# ---------------------------------------------------------------------------

@unittest.skipIf(SNAPSHOT is None, "v2_oracle_snapshot.json missing")
class RewardSnapshotParityTests(unittest.TestCase):
    def test_snapshot_provenance(self):
        self.assertEqual(SNAPSHOT["supermimic_commit"], O.SUPERMIMIC_COMMIT)
        self.assertEqual(SNAPSHOT["nablamimic_commit"], O.NABLAMIMIC_COMMIT)
        self.assertEqual(SNAPSHOT["oracle_sha256"], O.ORACLE_SHA256)

    def test_v2_reproduces_recorded_v2_terms(self):
        for kind, rec in SNAPSHOT["cases"].items():
            got = O.v2_reward_terms(O.make_case(kind))
            for name, value in rec["v2"]["terms"].items():
                self.assertAlmostEqual(got[name], value, places=12, msg=f"{kind}/{name}")

    def test_v2_equals_nablamimic_on_all_eight_terms(self):
        for kind, rec in SNAPSHOT["cases"].items():
            v2 = O.v2_reward_terms(O.make_case(kind))
            for nm_name, v2_name in O.NM_TO_V2.items():
                self.assertAlmostEqual(v2[v2_name], rec["nablamimic"]["terms"][nm_name],
                                       delta=1e-7, msg=f"{kind}/{nm_name}")

    def test_v2_equals_supermimic_when_anchor_bodies_coincide(self):
        rec = SNAPSHOT["cases"]["coincident_anchor"]
        v2 = O.v2_reward_terms(O.make_case("coincident_anchor"))
        self.assertAlmostEqual(sum(v2.values()), rec["supermimic"]["total"], delta=1e-7)

    def test_supermimic_pelvis_anchor_divergence_is_real_and_documented(self):
        rec = SNAPSHOT["cases"]["drift"]
        gap = abs(rec["v2"]["total"] - rec["supermimic"]["total"])
        self.assertGreater(gap, 1e-5)  # the divergence is NOT numerical noise
        self.assertLess(gap, 0.05)

    def test_perfect_tracking_is_a_common_point_for_all_three(self):
        rec = SNAPSHOT["cases"]["perfect"]
        # SuperMimic's acos clamp leaves a 3.75e-7 residual at zero angle.
        self.assertLess(abs(rec["v2"]["total"] - rec["nablamimic"]["total"]), 1e-7)
        self.assertLess(abs(rec["v2"]["total"] - rec["supermimic"]["total"]), 1e-6)


# ---------------------------------------------------------------------------
# 6. Reward parity -- real upstream source (dynamic)
# ---------------------------------------------------------------------------

@unittest.skipUnless(REAL_ORACLES, "pinned upstream sources / torch / jax / mjx unavailable")
class RewardRealSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sm = O.load_supermimic()
        cls.nm = O.load_nablamimic()

    def test_source_integrity(self):
        integrity = O.source_integrity()
        self.assertTrue(all(v == "OK" for v in integrity.values()), integrity)

    def test_v2_matches_nablamimic_term_by_term(self):
        for kind in ("perfect", "coincident_anchor", "drift"):
            d = O.make_case(kind)
            v2 = O.v2_reward_terms(d)
            nm_total, nm = O.nm_reward(d, self.nm)
            for nm_name, v2_name in O.NM_TO_V2.items():
                self.assertAlmostEqual(v2[v2_name], nm[nm_name], delta=1e-7,
                                       msg=f"{kind}/{nm_name}")
            self.assertAlmostEqual(sum(v2.values()), nm_total, delta=1e-7, msg=kind)

    def test_v2_matches_supermimic_when_anchors_coincide(self):
        for kind in ("perfect", "coincident_anchor"):
            d = O.make_case(kind)
            v2 = O.v2_reward_terms(d)
            sm_total, sm = O.sm_reward(d, self.sm)
            self.assertAlmostEqual(sum(v2.values()), sm_total, delta=1e-6, msg=kind)
            self.assertAlmostEqual(sum(v for k, v in v2.items() if k in S.TRACKING_WEIGHTS),
                                   sm["track"], delta=1e-6, msg=kind)

    def test_action_rate_and_joint_limit_match_supermimic_in_every_case(self):
        for kind in ("perfect", "coincident_anchor", "drift"):
            d = O.make_case(kind)
            v2 = O.v2_reward_terms(d)
            _, sm = O.sm_reward(d, self.sm)
            self.assertAlmostEqual(v2["action_rate_l2"], S.CONTROL_DT * (-0.1) * sm["action_rate"],
                                   places=10, msg=kind)
            self.assertAlmostEqual(v2["joint_limit"], S.CONTROL_DT * (-2.0) * sm["joint_limit"],
                                   places=10, msg=kind)

    def test_supermimic_pelvis_vs_nablamimic_torso_divergence(self):
        d = O.make_case("drift")
        sm_total, _ = O.sm_reward(d, self.sm)
        nm_total, _ = O.nm_reward(d, self.nm)
        self.assertGreater(abs(sm_total - nm_total), 1e-5)

    def test_body_velocity_errors_use_world_frame_not_reanchored(self):
        d = O.make_case("drift")
        errs = S.reward_errors(O.as_state(d))
        self.assertAlmostEqual(
            errs["motion_body_lin_vel"],
            float(np.mean(np.sum((d["ref_body_lin_vel"] - d["robot_body_lin_vel"]) ** 2, -1))),
            places=12)


# ---------------------------------------------------------------------------
# 7. Termination parity
# ---------------------------------------------------------------------------

class TerminationPureTests(unittest.TestCase):
    def setUp(self):
        self.d = O.make_case("drift")

    def test_normal_state_does_not_terminate(self):
        self.assertFalse(O.v2_termination(self.d).terminated)

    def test_height_strict_boundary(self):
        self.assertFalse(O.v2_termination(self.d, dz=0.249).terminated)
        self.assertFalse(O.v2_termination(self.d, dz=0.25).terminated)
        self.assertEqual(O.v2_termination(self.d, dz=0.251).reasons, ("root_height",))
        self.assertIn("root_height", O.v2_termination(self.d, dz=-0.251).reasons)

    def test_tilt_boundary(self):
        self.assertNotIn("root_tilt", O.v2_termination(self.d, tilt=0.6).reasons)
        self.assertIn("root_tilt", O.v2_termination(self.d, tilt=1.7).reasons)

    def test_nonfinite(self):
        self.assertEqual(O.v2_termination(self.d, nan=True).reasons, ("nonfinite",))

    def test_large_wrist_ankle_deviation_does_not_terminate(self):
        """V1's ee_body_pos would fire here; V2 must not."""
        d = self.d
        term = O.v2_termination(d, ee_deviation=0.9)
        self.assertFalse(term.terminated, term.reasons)
        self.assertGreater(0.9, C.TERM_EE_BODY_POS_THRESHOLD)

    def test_clip_end_is_success_not_failure(self):
        ok = S.TerminationResult(False, ())
        self.assertTrue(S.clip_end_success(step=100, n_steps=100, term=ok))
        self.assertFalse(S.clip_end_success(step=99, n_steps=100, term=ok))
        self.assertFalse(S.clip_end_success(step=100, n_steps=100,
                                            term=S.TerminationResult(True, ("root_height",))))

    def test_v2_has_no_ee_term_in_protocol_or_config(self):
        cfg = S.load_protocol_v2()
        self.assertNotIn("ee_body_pos", cfg["termination"]["terms"])
        self.assertNotIn("ee_body_pos", V2C.V2_TERMINATIONS)
        self.assertIn("ee_body_pos", V2C.V2_REMOVED_TERMINATIONS)


@unittest.skipUnless(REAL_ORACLES, "pinned upstream sources / torch / jax / mjx unavailable")
class TerminationRealSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sm = O.load_supermimic()
        cls.nm = O.load_nablamimic()
        cls.d = O.make_case("drift")

    def test_height_and_tilt_boundaries_match_both_originals(self):
        for kw in ({}, {"dz": 0.249}, {"dz": 0.25}, {"dz": 0.251},
                   {"dz": -0.251}, {"tilt": 0.6}, {"tilt": 1.7}):
            sm = O.sm_termination(self.d, mdp=self.sm, **kw)
            nm = O.nm_termination(self.d, nenv=self.nm, **kw)
            v2 = O.v2_termination(self.d, **kw).terminated
            self.assertEqual(sm, nm, kw)
            self.assertEqual(sm, v2, kw)

    def test_nonfinite_is_a_failure_in_nablamimic_and_v2_but_not_supermimic(self):
        self.assertFalse(O.sm_termination(self.d, mdp=self.sm, nan=True))
        self.assertTrue(O.nm_termination(self.d, nenv=self.nm, nan=True))
        self.assertEqual(O.v2_termination(self.d, nan=True).reasons, ("nonfinite",))

    def test_wrist_ankle_deviation_does_not_terminate_in_either_original(self):
        # Both originals only look at pelvis z / tilt; changing body positions
        # cannot affect their termination at all.
        self.assertFalse(O.sm_termination(self.d, mdp=self.sm))
        self.assertFalse(O.nm_termination(self.d, nenv=self.nm))
        self.assertFalse(O.v2_termination(self.d, ee_deviation=0.9).terminated)

    def test_clip_end_never_marked_failed_by_originals(self):
        # supermimic/env.py:221 failed_eval |= term & ~done_eval ; timeout only
        # sets done_eval.  nablamimic/evaluate.py:70 clip_ended only clears alive.
        sm_src = (O.supermimic_root() / "supermimic/env.py").read_text()
        nm_src = (O.nablamimic_root() / "src/nablamimic/evaluate.py").read_text()
        self.assertIn("failed_eval |= term & ~self.done_eval", sm_src)
        self.assertIn("alive = alive & ~term & ~f[\"clip_ended\"](s2)", nm_src)


# ---------------------------------------------------------------------------
# 8. Normalization parity
# ---------------------------------------------------------------------------

class NormalizationParityTests(unittest.TestCase):
    N = 10
    DEN = S.MAX_STEP_REWARD * 10  # (T-1) * max_step_reward with T-1 = 10

    def test_full_clip_denominator_and_ceiling(self):
        r = [0.1] * self.N
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N), 1.0)

    def test_early_termination_zeroes_remaining_rewards(self):
        r = [0.1] * self.N
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N, terminated_at=3), 0.4)
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N, terminated_at=0), 0.1)
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N, terminated_at=self.N - 1), 1.0)

    def test_terminating_step_reward_is_counted(self):
        r = [0.05] * 5 + [0.3] + [0.1] * 4
        expected = (0.05 * 5 + 0.3) / self.DEN
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N, terminated_at=5), expected)

    def test_negative_normalized_return_is_preserved(self):
        r = [-0.05] * self.N
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N), -0.5)

    def test_above_one_normalized_return_is_preserved(self):
        r = [0.15] * self.N
        self.assertAlmostEqual(S.rollout_normalized_return(r, self.N), 1.5)

    def test_v1_clamp_is_not_applied_in_v2(self):
        for raw in (-0.5, 1.5, 0.3):
            v2 = S.rollout_normalized_return([raw * S.MAX_STEP_REWARD] * self.N, self.N)
            self.assertAlmostEqual(v2, raw)
            self.assertAlmostEqual(S.v1_clamped_normalized_return(v2), min(max(raw, 0), 1))
        self.assertNotEqual(S.rollout_normalized_return([-0.05] * self.N, self.N),
                            S.v1_clamped_normalized_return(-0.5))

    def test_per_clip_aggregation_is_unweighted_mean(self):
        self.assertAlmostEqual(S.aggregate([1.0, -0.5, 1.5, 0.0]), 0.5)
        self.assertEqual(S.failed_clips([True, False, True]), 2)

    def test_clip_end_handling_uses_T_minus_1_steps(self):
        for T in (2, 11, 161):
            n = T - 1
            self.assertAlmostEqual(S.rollout_normalized_return([0.1] * n, n), 1.0)
            self.assertTrue(S.clip_end_success(n, n, S.TerminationResult(False, ())))

    def test_matches_supermimic_and_nablamimic_formula(self):
        # supermimic/evaluate.py:84  norm = ret / ((clip_len-1) * MAX_STEP_REWARD)
        # nablamimic/evaluate.py:90  tsum.sum(1) / (full_len * env.max_reward)
        rng = random.Random(5)
        for _ in range(50):
            n = rng.randint(2, 200)
            r = np.array([rng.uniform(-0.3, 0.12) for _ in range(n)])
            term_at = rng.choice([None, rng.randrange(n)])
            masked = r.copy()
            if term_at is not None:
                masked[term_at + 1:] = 0.0
            expected = masked.sum() / (n * 0.1)
            self.assertAlmostEqual(S.rollout_normalized_return(r, n, term_at), expected, places=12)

    @unittest.skipUnless(REAL_ORACLES, "pinned upstream sources unavailable")
    def test_original_source_lines_still_define_the_full_clip_denominator(self):
        sm = (O.supermimic_root() / "supermimic/evaluate.py").read_text()
        nm = (O.nablamimic_root() / "src/nablamimic/evaluate.py").read_text()
        self.assertIn("norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)", sm)
        self.assertIn("full_len = env.lengths - 1", nm)
        self.assertIn("tsum.sum(1) / (full_len * env.max_reward)", nm)
        self.assertNotIn("clamp", nm)
        self.assertNotIn("clip(", sm)


# ---------------------------------------------------------------------------
# 8b. V2 rollout driver contract (static, GPU-free)
# ---------------------------------------------------------------------------

class V2RunnerContractTests(unittest.TestCase):
    """The runtime driver must not reintroduce V1's clamp or V1's failures."""

    @classmethod
    def setUpClass(cls):
        cls.src = (REPO / "benchmark/v2/runner.py").read_text()

    def test_no_per_clip_clamp(self):
        self.assertNotIn("min(max(raw, 0.0), 1.0)", self.src)
        self.assertIn("NO [0,1] clamp", self.src)

    def test_full_clip_step_count(self):
        self.assertIn("n_steps = ref.num_frames - 1", self.src)

    def test_normalization_calls_the_v2_spec(self):
        self.assertIn("S.rollout_normalized_return(rewards, n_steps, terminated_at)", self.src)

    def test_early_termination_zeroes_remaining_rewards(self):
        self.assertIn("rewards[step + 1:] = 0.0", self.src)

    def test_time_out_is_excluded_from_failures(self):
        self.assertIn('if name == "time_out":', self.src)
        self.assertIn("failed=terminated_at is not None", self.src)

    def test_record_uses_v2_identity_not_v1_core(self):
        self.assertIn('"protocol_id": cfg["protocol_id"]', self.src)
        self.assertIn("self_collision_reward_penalty", self.src)
        self.assertIn("aggregate_normalized_return", self.src)
        self.assertNotIn("core.audit_record", self.src)
        self.assertNotIn("from .. import core", self.src)


# ---------------------------------------------------------------------------
# 9. V2 record validation
# ---------------------------------------------------------------------------

class V2RecordValidationTests(unittest.TestCase):
    def _record(self, ret: float, failed: int = 0) -> dict:
        ids = S.load_protocol_v2()["clip_ids"]
        return {
            "protocol_id": S.PROTOCOL_ID,
            "checkpoints": [{
                "per_clip": [{"clip": c, "normalized_return": ret, "failed": i < failed}
                             for i, c in enumerate(ids)],
            }],
            "evaluator": {
                "implementation": S.EVALUATOR_IMPLEMENTATION,
                "self_collision": True,
                "self_collision_reward_penalty": False,
                "mujoco": "3.11.0", "mujoco_warp": "3.11.0", "warp_lang": "1.16.0",
            },
        }

    def test_accepts_negative_and_above_one(self):
        for ret in (-0.5, 1.5, 0.7):
            res = S.audit_v2_record(self._record(ret))
            self.assertEqual(res["errors"], [], res["errors"])
            self.assertEqual(res["status"], "FORMAT_VALID_PENDING_PHYSICS_AUDIT")
            self.assertAlmostEqual(res["aggregate_normalized_return"], ret)
            self.assertFalse(res["clipping_applied"])

    def test_rejects_v1_record(self):
        res = S.audit_v2_record(_v1_record(0.8))
        self.assertEqual(res["status"], "INVALID")
        self.assertIn("protocol_id mismatch (a V1 record is not a V2 record)", res["errors"])

    def test_requires_full_self_collision_physics_but_no_penalty(self):
        bad = self._record(0.5)
        bad["evaluator"]["self_collision"] = False
        self.assertIn("full self-collision physics must be enabled", S.audit_v2_record(bad)["errors"])
        bad = self._record(0.5)
        bad["evaluator"]["self_collision_reward_penalty"] = True
        self.assertIn("V2 must declare self_collision_reward_penalty=false",
                      S.audit_v2_record(bad)["errors"])

    def test_requires_exactly_the_40_protocol_clips(self):
        bad = self._record(0.5)
        bad["checkpoints"][0]["per_clip"].pop()
        self.assertEqual(S.audit_v2_record(bad)["status"], "INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
