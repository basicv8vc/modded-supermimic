"""GPU runtime integration tests for Benchmark V2 (task prompt §10).

Two tiers:

* GPU-free contract tests always run (CPU-fallback rejection, smoke validation,
  V2 identity, V1 regression protection).
* Real CUDA runtime tests run only when a CUDA/Warp/MuJoCo Warp stack is
  actually usable.  Otherwise they are reported as **BLOCKED** (skip reason
  starts with ``BLOCKED:``) -- they are never silently counted as PASS.

Run the equivalent standalone check with
``python scripts/v2/gpu_runtime_check.py``; it writes
``reports/v2/gpu_runtime_status.json`` with an explicit PASS/FAIL/BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from benchmark.v2 import parity as P  # noqa: E402
from benchmark.v2 import runtime as RT  # noqa: E402
from benchmark.v2 import spec as S  # noqa: E402


def _cuda_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        return False, f"BLOCKED: torch not installed ({exc})"
    if not torch.cuda.is_available():
        return False, "BLOCKED: torch.cuda.is_available() == False"
    try:
        RT.probe_cuda("cuda:0")
    except RT.CudaRuntimeUnavailable as exc:
        return False, f"BLOCKED: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"BLOCKED: CUDA probe error: {type(exc).__name__}: {exc}"
    return True, ""


GPU_OK, GPU_REASON = _cuda_status()

# Tier 1 is *CUDA*-free, not dependency-free: a couple of contract tests still
# need torch (device selection) or mujoco (manifest hashes).  Guard them so a
# plain CPU host reports BLOCKED/skipped instead of an ERROR -- otherwise the
# module docstring's "always runs" promise is false on a clean machine.
try:
    import torch as _torch  # noqa: F401, PLC0415

    TORCH_OK, TORCH_REASON = True, ""
except ImportError as exc:  # pragma: no cover - host dependent
    TORCH_OK, TORCH_REASON = False, f"BLOCKED: torch not installed ({exc})"

try:
    import mujoco as _mujoco  # noqa: F401, PLC0415

    MUJOCO_OK, MUJOCO_REASON = True, ""
except ImportError as exc:  # pragma: no cover - host dependent
    MUJOCO_OK, MUJOCO_REASON = False, f"BLOCKED: mujoco not installed ({exc})"


# ---------------------------------------------------------------------------
# Tier 1: contract tests that do not need a GPU
# ---------------------------------------------------------------------------

class GpuFreeContractTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_OK, TORCH_REASON or "BLOCKED: torch unavailable")
    def test_cpu_fallback_is_rejected_not_silently_used(self):
        res = P.check_cpu_fallback_rejection()
        self.assertEqual(res["status"], "PASS", res)
        self.assertTrue(res["checks"]["cpu_device_string_rejected"])
        self.assertTrue(res["checks"]["cpu_warp_state_rejected"])

    def test_v2_runner_has_no_cpu_device_literal(self):
        src = (REPO / "benchmark/v2/runner.py").read_text()
        self.assertIn('device=self.device', src)
        self.assertIn("RT.inspect_runtime(self._env)", src)
        self.assertNotIn('device="cpu"', src)
        self.assertNotIn("device='cpu'", src)

    def test_v1_cpu_evaluator_is_still_untouched(self):
        src = (REPO / "benchmark/evaluator/env.py").read_text()
        self.assertIn('device="cpu"', src)  # V1 deliberately unchanged
        self.assertEqual(S.v1_protocol_sha256(), S.V1_PROTOCOL_SHA256)

    def test_smoke_record_is_valid_and_never_submission_eligible(self):
        rec = self._smoke_record(1)
        audit = S.audit_v2_smoke(rec)
        self.assertEqual(audit["status"], "SMOKE_VALID", audit["errors"])
        self.assertFalse(S.is_submission_eligible(rec))
        self.assertEqual(S.audit_v2_record(rec)["status"], "INVALID")

    def test_smoke_accepts_one_or_two_clips_only(self):
        for n in (1, 2):
            rec = self._smoke_record(n)
            self.assertEqual(S.audit_v2_smoke(rec)["status"], "SMOKE_VALID")
        self.assertEqual(S.audit_v2_smoke(self._smoke_record(3))["status"], "INVALID")
        self.assertEqual(S.audit_v2_smoke(self._smoke_record(0))["status"], "INVALID")

    def test_smoke_requires_cuda_provenance_not_a_cli_gpu_name(self):
        rec = self._smoke_record(1)
        rec["hardware"]["gpu_name"] = "FAKE RTX PRO 6000 typed by user"
        rec["evaluation"]["provenance"]["actual_torch_device"] = "cpu"
        audit = S.audit_v2_smoke(rec)
        self.assertEqual(audit["status"], "INVALID")
        self.assertTrue(any("CUDA" in e for e in audit["errors"]))

    def test_formal_benchmark_gate_still_requires_40_clips(self):
        ids = S.load_protocol_v2()["clip_ids"]
        rec = {
            "protocol_id": S.PROTOCOL_ID,
            "evaluation": {"mode": "benchmark", "submission_eligible": True},
            "submission_eligible": True,
            "checkpoints": [{"per_clip": [
                {"clip": c, "normalized_return": 0.3, "failed": False} for c in ids]}],
            "evaluator": {"implementation": S.EVALUATOR_IMPLEMENTATION,
                          "self_collision": True,
                          "self_collision_reward_penalty": False,
                          "mujoco": "3.11.0", "mujoco_warp": "3.11.0",
                          "warp_lang": "1.16.0"},
        }
        self.assertTrue(S.is_submission_eligible(rec))
        rec["checkpoints"][0]["per_clip"].pop()
        self.assertFalse(S.is_submission_eligible(rec))
        rec["checkpoints"][0]["per_clip"].append(
            {"clip": ids[-1], "normalized_return": 0.3, "failed": False})
        rec["submission_eligible"] = False           # smoke cannot be padded in
        self.assertFalse(S.is_submission_eligible(rec))

    def test_controlled_baseline_comparison_valid_and_not_submission_eligible(self):
        ids = S.load_protocol_v2()["clip_ids"]
        rec = {
            "protocol_id": S.PROTOCOL_ID,
            "evaluation": {"mode": "benchmark", "submission_eligible": False, "comparison_valid": True},
            "submission_eligible": False,
            "comparison_valid": True,
            "target_hardware_met": False,
            "checkpoints": [{"per_clip": [
                {"clip": c, "normalized_return": 0.3, "failed": False} for c in ids]}],
            "evaluator": {"implementation": S.EVALUATOR_IMPLEMENTATION,
                          "self_collision": True,
                          "self_collision_reward_penalty": False,
                          "mujoco": "3.11.0", "mujoco_warp": "3.11.0",
                          "warp_lang": "1.16.0"},
        }
        self.assertFalse(S.is_submission_eligible(rec))
        self.assertTrue(S.is_comparison_valid(rec))

    @unittest.skipUnless(MUJOCO_OK, MUJOCO_REASON or "BLOCKED: mujoco unavailable")
    def test_v2_clip_result_serialization_roundtrip(self):
        from benchmark.v2 import runner as v2runner
        r1 = v2runner.V2ClipResult(
            clip="walk1_subject1",
            normalized_return=0.12345,
            failed=True,
            steps=13064,
            alive_steps=52,
            fall_step=52,
            fall_terms=("root_height",),
            mean_term_score={"motion_body_pos": 0.05},
            wall_s=1.23,
            device="cuda:0",
            performance={"steady_simulation_fps": 65.0},
            runtime={"device": "cuda:0"},
        )
        d = r1.to_dict()
        r2 = v2runner.V2ClipResult.from_dict(d)
        self.assertEqual(r1.clip, r2.clip)
        self.assertEqual(r1.normalized_return, r2.normalized_return)
        self.assertEqual(r1.failed, r2.failed)
        self.assertEqual(r1.fall_terms, r2.fall_terms)
        self.assertEqual(r1.steps, r2.steps)
        self.assertEqual(r1.alive_steps, r2.alive_steps)

    def test_cli_distinguishes_smoke_from_benchmark(self):
        src = (REPO / "scripts/v2/run_eval.py").read_text()
        self.assertIn("audit_v2_smoke", src)
        self.assertIn("audit_v2_record", src)
        self.assertIn("submission_eligible", src)
        self.assertIn("ap.error(f\"smoke requires", src)
        self.assertIn("full 40 protocol clips", src)

    def test_cli_benchmark_gate_requires_target_hardware(self):
        # Regression (A100 controlled baseline): benchmark mode must NOT
        # unconditionally stamp submission_eligible=true.  The G4 formal
        # submission gate is target-hardware-qualified; non-target hardware
        # (A100/sm_80, L4/sm_89) yields comparison_valid=true instead.
        src = (REPO / "scripts/v2/run_eval.py").read_text()
        self.assertNotIn('record["submission_eligible"] = (mode == "benchmark")', src)
        self.assertIn("comparison_valid", src)
        self.assertIn("target_hardware_met", src)
        self.assertIn("BENCHMARK_COMPARISON_VALID", src)
        self.assertIn("BENCHMARK_SUBMISSION_VALID", src)

    def test_v2_protocol_and_identity_unchanged(self):
        cfg = S.load_protocol_v2()
        self.assertEqual(cfg["protocol_id"], S.PROTOCOL_ID)
        self.assertEqual(len(cfg["clip_ids"]), 40)
        self.assertEqual(S.protocol_sha256(), S.V2_PROTOCOL_SHA256)

    def test_gpu_runtime_check_script_reports_blocked_without_cuda(self):
        # The script must exist and contain an explicit BLOCKED path.
        src = (REPO / "scripts/v2/gpu_runtime_check.py").read_text()
        self.assertIn('"status": "BLOCKED"', src)
        self.assertIn("return 3", src)
        self.assertIn("PASS", src)

    def test_runtime_relative_reference_override_keeps_v2_math_identical(self):
        """The runtime entry point must be the *same* V2 math, not a new formula."""
        rng = np.random.default_rng(11)
        B = len(S.TRACKED_BODIES)

        def unit_q(shape):
            q = rng.standard_normal(shape)
            return q / np.linalg.norm(q, axis=-1, keepdims=True)

        st = S.V2State(
            robot_body_pos=rng.standard_normal((B, 3)) * 0.2,
            robot_body_quat=unit_q((B, 4)),
            robot_body_lin_vel=rng.standard_normal((B, 3)) * 0.1,
            robot_body_ang_vel=rng.standard_normal((B, 3)) * 0.1,
            ref_body_pos=rng.standard_normal((B, 3)) * 0.2,
            ref_body_quat=unit_q((B, 4)),
            ref_body_lin_vel=rng.standard_normal((B, 3)) * 0.1,
            ref_body_ang_vel=rng.standard_normal((B, 3)) * 0.1,
            robot_root_z=0.8, robot_root_quat=np.array([1.0, 0, 0, 0]),
            ref_root_z=0.8, ref_root_quat=np.array([1.0, 0, 0, 0]),
            joint_pos=rng.standard_normal(29) * 0.1,
            action=rng.standard_normal(29) * 0.1,
            prev_action=rng.standard_normal(29) * 0.1,
            soft_lo=np.full(29, -1.0), soft_hi=np.full(29, 1.0),
        )
        rel_p, rel_q = S.reference_reanchor(st)
        plain = S.reward_terms(st)
        overridden = S.reward_terms_from_relative_reference(st, rel_p, rel_q)
        for name in S.REWARD_TERM_NAMES:
            self.assertAlmostEqual(plain[name], overridden[name], places=14, msg=name)

        # A changed reference must affect ONLY the two relative-body terms.
        shifted = S.reward_terms_from_relative_reference(st, rel_p + 0.05, rel_q)
        self.assertNotAlmostEqual(shifted["motion_body_pos"],
                                  plain["motion_body_pos"], places=6)
        for name in ("motion_global_root_pos", "motion_global_root_ori",
                     "motion_body_lin_vel", "motion_body_ang_vel",
                     "action_rate_l2", "joint_limit"):
            self.assertAlmostEqual(shifted[name], plain[name], places=14, msg=name)

    def test_tilt_boundary_deviations_are_exact(self):
        """0.79 / 0.80 / 0.81 must be exact deviations for the strict > rule."""
        for d in (0.79, 0.80, 0.81):
            ref_q, robot_q = P._tilt_quat_pair(d)
            diff = abs(S.gravity_b_z(np.asarray(robot_q))
                       - S.gravity_b_z(np.asarray(ref_q)))
            self.assertAlmostEqual(diff, d, places=11, msg=f"d={d}")

    @unittest.skipUnless(MUJOCO_OK, MUJOCO_REASON or "BLOCKED: mujoco unavailable")
    def test_provenance_reports_measured_fps_and_runtime_detection(self):
        """Regression: os.uname().release, measured FPS, frozen protocol SHA."""
        fake = {
            "gpu_name": "runtime-detected", "compute_capability_string": "sm_89",
            "gpu_count": 1, "torch_device": "cuda:0", "warp_device": "cuda:0",
            "warp_cuda_devices": ["cuda:0"], "simulation_backend": "mujoco_warp",
            "cuda_runtime": "13.0", "driver_version": "580.82.07",
            "versions": {"mujoco": "3.11.0", "mujoco_warp": "3.11.0",
                         "warp_lang": "1.16.0", "mjlab": "1.6.0"},
            "version_drift": {}, "target_gpu_match": False, "sm_120_confirmed": False,
        }
        prov = RT.collect_provenance(
            probe=fake,
            runtime={"simulation_backend": "mujoco_warp",
                     "step_graph_captured": True, "wp_data_qpos_device": "cuda:0"},
            performance={"steady_simulation_fps": 65.2},
        )
        self.assertEqual(prov["simulation_fps"], 65.2)
        self.assertIn("measured", prov["simulation_fps_source"])
        self.assertTrue(prov["platform"])
        self.assertEqual(prov["v2_protocol_sha256"], S.V2_PROTOCOL_SHA256)
        self.assertTrue(prov["cuda_graph_step_captured"])
        self.assertEqual(prov["actual_torch_device"], "cuda:0")

    def test_graph_capture_missing_is_not_a_cpu_fallback(self):
        """mjlab disables CUDA graphs by itself; that is throughput, not CPU physics.

        ``Simulation._should_use_cuda_graph`` returns False when the Warp mempool
        is off or the driver is older than 12.4.  Treating a missing graph as
        "CPU fallback" would abort a genuine GPU run on newer hardware.
        """
        import types as _t

        Data = type("Data", (), {"__module__": "mujoco_warp._src.types"})
        data = Data()
        data.qpos = _t.SimpleNamespace(device=_t.SimpleNamespace(is_cuda=True))
        sim = _t.SimpleNamespace(
            device="cuda:0",
            wp_device=_t.SimpleNamespace(is_cuda=True, arch=120),
            wp_data=data, use_cuda_graph=False, step_graph=None)
        env = _t.SimpleNamespace(device="cuda:0", sim=sim)

        info = RT.inspect_runtime(env)
        self.assertTrue(info["no_cpu_fallback"])
        self.assertFalse(info["step_graph_captured"])
        self.assertEqual(info["graph_capture_status"], "disabled_by_runtime")
        self.assertTrue(info["warnings"], "a missing graph must be surfaced")

    def test_cpu_simulation_state_is_still_rejected(self):
        """The relaxed graph rule must not weaken the real CPU-fallback guard."""
        import types as _t

        Data = type("Data", (), {"__module__": "mujoco_warp._src.types"})
        data = Data()
        data.qpos = _t.SimpleNamespace(device=_t.SimpleNamespace(is_cuda=False))
        sim = _t.SimpleNamespace(
            device="cpu",
            wp_device=_t.SimpleNamespace(is_cuda=False, arch=0),
            wp_data=data, use_cuda_graph=False, step_graph=None)
        env = _t.SimpleNamespace(device="cpu", sim=sim)
        with self.assertRaises(RT.CudaRuntimeUnavailable):
            RT.inspect_runtime(env)

    def test_acceptance_requires_target_hardware_not_just_parity(self):
        import importlib.util

        p = REPO / "scripts/v2/gpu_runtime_check.py"
        spec = importlib.util.spec_from_file_location("v2_gpu_check_under_test", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # The preflight L4 run: parity passed, hardware did not.
        self.assertEqual(mod.acceptance_status("PASS", False, False),
                         ("BLOCKED_TARGET_HARDWARE", False))
        self.assertEqual(mod.acceptance_status("PASS", True, False),
                         ("BLOCKED_TARGET_HARDWARE", False))
        self.assertEqual(mod.acceptance_status("PASS", True, True), ("PASS", True))
        self.assertEqual(mod.acceptance_status("FAIL", True, True), ("FAIL", True))

    def test_source_fingerprint_covers_implementation(self):
        fp = RT.source_fingerprint()
        self.assertIn("combined", fp)
        self.assertNotIn("missing", fp.values())
        self.assertGreaterEqual(len(fp) - 1, 6)

    @staticmethod
    def _smoke_record(n: int) -> dict:
        ids = S.load_protocol_v2()["clip_ids"][:n]
        return {
            "protocol_id": S.PROTOCOL_ID,
            "mode": "smoke",
            "submission_eligible": False,
            "hardware": {"gpu_name": "runtime-detected", "declared_gpu_name": "user-typed"},
            "checkpoints": [{"per_clip": [
                {"clip": c, "normalized_return": 0.12, "failed": False} for c in ids]}],
            "evaluator": {"implementation": S.EVALUATOR_IMPLEMENTATION,
                          "self_collision": True,
                          "self_collision_reward_penalty": False},
            "evaluation": {
                "mode": "smoke", "submission_eligible": False,
                "provenance": {"actual_torch_device": "cuda:0",
                               "actual_simulation_backend": "mujoco_warp",
                               "cuda_graph_step_captured": True},
            },
        }


# ---------------------------------------------------------------------------
# Tier 2: real CUDA / mjlab GPU runtime
# ---------------------------------------------------------------------------

@unittest.skipUnless(GPU_OK, GPU_REASON or "BLOCKED: CUDA unavailable")
class GpuRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = P.run_gpu_check("walk1_subject1", device="cuda:0", steps=6)

    def test_cuda_runtime_selection(self):
        r = self.report
        self.assertTrue(r["runtime"]["env_device"].startswith("cuda"))
        self.assertTrue(r["runtime"]["warp_device_is_cuda"])
        self.assertTrue(r["runtime"]["wp_data_qpos_is_cuda"])
        # A missing CUDA graph is a throughput fact, not a CPU fallback: mjlab
        # disables graphs by itself when the Warp mempool is off or the driver
        # is old (sim.py:_should_use_cuda_graph).  Require a *recorded* status,
        # not a captured graph.
        self.assertIn(r["runtime"]["graph_capture_status"],
                      ("captured", "disabled_by_runtime", "enabled_but_uncaptured"))
        self.assertTrue(r["runtime"]["no_cpu_fallback"])
        self.assertIn("cuda", r["runtime"]["wp_data_qpos_device"])

    def test_cpu_fallback_rejection_at_runtime(self):
        self.assertEqual(self.report["checks"]["cpu_fallback_rejection"]["status"], "PASS")

    def test_cuda_kernel_really_executed(self):
        k = self.report["kernel_execution"]
        self.assertTrue(k["state_advanced_by_gpu_kernel"], k)
        self.assertTrue(k["qpos_finite_after_step"])
        self.assertGreater(k["qpos_max_abs_delta"], 0.0)

    def test_real_mjlab_reward_parity(self):
        r = self.report["checks"]["reward"]
        self.assertEqual(r["status"], "PASS", json.dumps(r["checks"], indent=2))
        self.assertTrue(r["checks"]["tolerance"],
                        f"max|err|={r['max_abs_error']:.3e} maxrel={r['max_rel_error']:.3e}")
        self.assertEqual(tuple(sorted(r["static"]["weights_runtime"])),
                         tuple(sorted(S.REWARD_TERM_NAMES)))
        self.assertTrue(all(r["checks"].values()), r["checks"])

    def test_reward_oracle_uses_runtime_state_not_reward_outputs(self):
        st = self.report["checks"]["reward"]["static"]
        self.assertIn("MotionCommand.body_pos_relative_w",
                      st["relative_reference_source"])
        self.assertIn("independent math", st["oracle_expected_value_source"])
        self.assertTrue(self.report["checks"]["reward"]["checks"]["body_indexing_match"])
        self.assertTrue(self.report["checks"]["reward"]["checks"]["root_anchor_match"])
        self.assertTrue(self.report["checks"]["reward"]["checks"]["dt_scaling_match"])
        self.assertTrue(self.report["checks"]["reward"]["checks"]["action_history_matches"])

    def test_reference_reanchor_formula_matches_cuda_buffers(self):
        r = self.report["checks"]["reference_reanchor"]
        self.assertEqual(r["status"], "PASS", json.dumps(r["rows"], indent=2))
        self.assertTrue(all(r["checks"].values()), r["checks"])
        self.assertLessEqual(r["max_abs_error"], P.REANCHOR_TOLERANCE)

    def test_termination_boundary_cases_cover_strict_thresholds(self):
        r = self.report["checks"]["termination"]
        names = [c["case"] for c in r["cases"]]
        for expected in ("height_exactly_0.25_strict_gt",
                         "tilt_exactly_0.8_strict_gt",
                         "height_0.251_above_threshold",
                         "tilt_0.81_above_threshold",
                         "nonfinite_qpos", "wrist_ankle_deviation_only"):
            self.assertIn(expected, names)
        self.assertTrue(all(c["match"] for c in r["cases"]),
                        json.dumps(r["cases"], indent=2))
        self.assertTrue(r["structural"]["termination_uses_current_reference_frame"])
        self.assertTrue(r["structural"]["clip_end_success_not_a_failure"])

    def test_real_mjlab_termination_parity(self):
        r = self.report["checks"]["termination"]
        self.assertEqual(r["status"], "PASS", json.dumps(r["cases"], indent=2))
        self.assertTrue(r["structural"]["ee_body_pos_absent"])
        self.assertTrue(r["structural"]["self_collision_reward_penalty_absent"])
        self.assertTrue(r["structural"]["self_collision_physics_contact_sensor_present"])
        self.assertTrue(r["structural"]["torso_anchor_not_used_for_termination"])
        self.assertTrue(all(c["match"] for c in r["cases"]))

    def test_correct_frame_alignment(self):
        r = self.report["checks"]["frame_alignment"]
        self.assertEqual(r["status"], "PASS", r["checks"])
        self.assertTrue(all(r["checks"].values()), r["checks"])
        # PR #13: frame 0 is the initial condition, first scored frame is 1.
        self.assertEqual(r["details"]["first_scored_reference_frame"],
                         P.FIRST_SCORED_REFERENCE_FRAME)
        self.assertTrue(r["checks"]["v2_matches_v1_pr13_frame_semantics"])
        self.assertTrue(r["checks"]["full_clip_frames_match_motion"])

    def test_v2_record_identity_and_v1_regression(self):
        r = self.report["checks"]["identity_v1"]
        self.assertEqual(r["status"], "PASS", r["checks"])

    def test_steady_state_fps_and_wall_time_recorded(self):
        perf = self.report["performance"]
        self.assertGreater(perf["steady_simulation_fps"], 0.0)
        self.assertGreater(perf["total_rollout_wall_s"], 0.0)
        self.assertGreater(perf["steady_steps"], perf["warmup_steps"])
        self.assertGreater(perf["first_step_wall_s"], 0.0)
        self.assertGreaterEqual(perf["steps"], 200)
        # Physics vs policy-glue split: a slow single env must be diagnosable
        # (task prompt §8) instead of just "slow".
        self.assertGreater(perf["sim_step_fps"], 0.0)
        self.assertGreaterEqual(perf["sim_step_steps"], 1)
        self.assertIn(perf["bottleneck"], ("physics", "policy_glue"))
        self.assertIn("policy_glue_overhead_s_per_step", perf)
        for key in ("cuda_init_probe_s", "reference_load_s",
                    "cuda_env_build_s", "first_kernel_jit_wall_s", "total_wall_s"):
            self.assertIn(key, self.report["timing"])
        tel = self.report["telemetry"]
        self.assertGreaterEqual(tel["samples"], 1)
        self.assertIn("gpu_memory_used_mib", tel)
        self.assertIn("cpu_utilization_pct", tel)

    def test_single_clip_smoke_cli_exits_zero(self):
        # Only run the CLI when a real dataset clip is present.
        from benchmark.evaluator import constants as C  # noqa: PLC0415
        clip_csv = C.DATA_DIR / "walk1_subject1.csv"
        if not clip_csv.exists():
            self.skipTest("BLOCKED: LAFAN1 CSV data not available on this host")
        # Distinct output so the integration test never clobbers the
        # task's smoke_gpu.json deliverable.
        out = REPO / "reports/v2/smoke_gpu_cli_test.json"
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts/v2/run_eval.py"),
             "--smoke-policy", "replay", "--clips", "walk1_subject1",
             "--cache-references", "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, timeout=1800,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        rec = json.loads(out.read_text())
        self.assertEqual(rec["mode"], "smoke")
        self.assertFalse(rec["submission_eligible"])
        self.assertEqual(S.audit_v2_smoke(rec)["status"], "SMOKE_VALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
