#!/usr/bin/env python3
"""Aggregate the CUDA runtime artifacts into one V2 GPU performance report.

    python scripts/v2/gpu_performance_report.py

Reads only the JSON artifacts produced by ``gpu_runtime_check.py`` and
``run_eval.py`` (no CUDA / mjlab / torch needed), so the report is
reproducible from the committed diagnostic outputs.  Hardware fields come
from runtime detection stored in those artifacts -- never from a CLI string.

Deliverable for task prompt §8 / §9 / §12: separates CUDA init, reference
loading, env build, kernel JIT/warmup and real steady-state rollout, and
records GPU utilisation / memory / CPU utilisation plus per-clip wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2_REPORTS = REPO / "reports" / "v2"

SMOKE_FILES = ("smoke_gpu.json", "smoke_gpu_dance1.json", "smoke_gpu_2clips.json")


def _load(name: str) -> dict:
    path = V2_REPORTS / name
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runtime_report() -> tuple[str, dict]:
    """Prefer the acceptance artifact; fall back to a committed preflight run.

    ``gpu_runtime_status.json`` only exists after a run on the target GPU.  The
    L4 / sm_89 preflight is archived under a name that says so, and is still a
    valid source for timing/telemetry -- but the caller must be able to see
    which one it got, hence the returned label.
    """
    for name in ("gpu_runtime_status.json", "preflight_l4_sm89_2026-11-12.json"):
        path = V2_REPORTS / name
        if path.exists():
            return name, json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(
        "no runtime artifact: expected reports/v2/gpu_runtime_status.json "
        "(target GPU) or the archived preflight JSON")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip_rows(smokes: dict[str, dict]) -> list[dict]:
    out = []
    for name, rec in smokes.items():
        ev = rec.get("evaluation") or {}
        perf = ev.get("performance") or {}
        diag = ev.get("per_clip_diagnostics") or {}
        rows = rec["checkpoints"][0]["per_clip"]
        for row in rows:
            clip = row["clip"]
            p = perf.get(clip) or {}
            d = diag.get(clip) or {}
            out.append({
                "source": name,
                "clip": clip,
                "normalized_return": row["normalized_return"],
                "failed": row["failed"],
                "full_clip_control_steps": d.get("steps"),
                "alive_steps": d.get("alive_steps"),
                "fall_step": d.get("fall_step"),
                "fall_terms": d.get("fall_terms"),
                # checkpoints[0].eval_wall_s is sum(r.wall_s) over that run's
                # clips (1 or 2); it is NOT a per-clip value in the 2-clip run.
                "eval_wall_s_sum_of_clips_in_this_run": rec["checkpoints"][0].get("eval_wall_s"),
                "clip_wall_s_from_parts": float(sum(
                    p.get(k) or 0.0 for k in ("cuda_env_build_s", "reset_wall_s",
                                              "total_step_wall_s", "reference_load_s"))),
                "rollout_step_wall_s": p.get("total_step_wall_s"),
                "warmup_wall_s": p.get("warmup_wall_s"),
                "steady_simulation_fps": p.get("steady_simulation_fps"),
                "cuda_env_build_s": p.get("cuda_env_build_s"),
                "reference_load_s": p.get("reference_load_s"),
            })
    return out


def build_report() -> dict:
    status_name, status = _load_runtime_report()
    smokes = {name: _load(name) for name in SMOKE_FILES}
    primary = smokes["smoke_gpu.json"]
    prov = (primary.get("evaluation") or {}).get("provenance") or {}
    timing = status.get("timing") or {}
    perf = status.get("performance") or {}
    kernel = status.get("kernel_execution") or {}
    tel = status.get("telemetry") or {}

    artifacts = [status_name, *SMOKE_FILES]
    return {
        "report": "modded-supermimic-g1-lafan1-v2-cuda-runtime-performance",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_id": primary.get("protocol_id"),
        "protocol_sha256": prov.get("v2_protocol_sha256"),
        "evaluator_git_sha": prov.get("evaluator_git_sha"),
        "gpu_runtime_status": status.get("status"),
        "runtime_parity_checks": {k: v.get("status")
                                  for k, v in (status.get("checks") or {}).items()},
        "reward_runtime_parity": {
            "max_abs_error": status.get("reward_max_abs_error"),
            "max_rel_error": status.get("reward_max_rel_error"),
            "per_term_max_abs_error": status.get("reward_per_term_max_abs_error"),
            "tolerance": {"atol": 2e-5, "rtol": 3e-4},
        },
        "target_hardware": {
            "requested": "NVIDIA RTX PRO 6000 Blackwell / sm_120 (Colab G4, 96GB)",
            "status": ("MET" if (status.get("target_gpu_match")
                                 and status.get("sm_120_confirmed"))
                       else ("BLOCKED_TARGET_HARDWARE"
                             if status.get("status") == "PASS" else "BLOCKED")),
            "reason": (status.get("_note")
                       or "target_gpu_match=%s sm_120_confirmed=%s (runtime-detected)"
                       % (status.get("target_gpu_match"), status.get("sm_120_confirmed"))),
            "source_artifact": status_name,
            "sm_120_confirmed": status.get("sm_120_confirmed"),
            "target_gpu_match": status.get("target_gpu_match"),
            "frozen_stack_sm_120_compatibility": (
                "VERIFIED" if status.get("sm_120_confirmed")
                else "NOT VERIFIED (no sm_120 device obtained)"),
        },
        "verified_hardware_runtime_detected": {
            "gpu_name": prov.get("gpu_name") or status.get("versions", {}).get("gpu_name"),
            "gpu_compute_capability": prov.get("gpu_compute_capability"),
            "actual_torch_device": prov.get("actual_torch_device"),
            "actual_warp_device": prov.get("actual_warp_device"),
            "warp_cuda_devices": prov.get("warp_cuda_devices"),
            "actual_simulation_backend": prov.get("actual_simulation_backend"),
            "simulation_state_device": prov.get("simulation_state_device"),
            "cuda_graph_step_captured": prov.get("cuda_graph_step_captured"),
            "cuda_runtime": prov.get("cuda_runtime"),
            "driver_version": prov.get("driver_version"),
            "no_cpu_fallback": status.get("no_cpu_fallback"),
            "runtime_device_details": status.get("runtime"),
        },
        "dependency_stack": {
            "versions": status.get("versions"),
            "version_drift": status.get("version_drift"),
            "mjlab_commit": prov.get("mjlab_commit"),
            "robot_model_sha256": prov.get("robot_model_sha256"),
            "dataset_manifest_sha256": prov.get("dataset_manifest_sha256"),
            "motion_converter_version": prov.get("motion_converter_version"),
            "note": "Frozen versions installed unchanged; no wheel was upgraded "
                    "or substituted to make sm_120 work.",
        },
        "timing_breakdown_s": {
            "cuda_init_probe_s": timing.get("cuda_init_probe_s"),
            "reference_load_s": timing.get("reference_load_s"),
            "cuda_env_build_s": timing.get("cuda_env_build_s"),
            "first_kernel_jit_wall_s": timing.get("first_kernel_jit_wall_s"),
            "warmup_20_steps_wall_s": perf.get("warmup_wall_s"),
            "steady_rollout_wall_s": perf.get("steady_wall_s"),
            "total_runtime_check_wall_s": timing.get("total_wall_s"),
            "note": "CUDA init / env build / reference load / kernel JIT are "
                    "one-off compilation costs and are EXCLUDED from the "
                    "steady-state FPS below.",
        },
        "steady_state_simulation": {
            "measured_fps": perf.get("steady_simulation_fps"),
            "steps": perf.get("steps"),
            "warmup_steps_excluded": perf.get("warmup_steps"),
            "steady_steps": perf.get("steady_steps"),
            "steady_wall_s": perf.get("steady_wall_s"),
            "step_wall_s_mean": perf.get("step_wall_s_mean"),
            "step_wall_s_max": perf.get("step_wall_s_max"),
            "first_step_wall_s": perf.get("first_step_wall_s"),
            "manual_resets": perf.get("manual_resets"),
            "nominal_control_fps": prov.get("nominal_control_fps"),
            "simulation_fps_source": prov.get("simulation_fps_source"),
        },
        "kernel_execution_proof": kernel,
        "telemetry": {
            "samples": tel.get("samples"),
            "interval_s": tel.get("interval_s"),
            "gpu_utilization_pct": tel.get("gpu_utilization_pct"),
            "gpu_memory_used_mib": tel.get("gpu_memory_used_mib"),
            "cpu_utilization_pct": tel.get("cpu_utilization_pct"),
            "smoke_telemetry": {name: (rec["evaluation"].get("telemetry") or {})
                                for name, rec in smokes.items()},
            "note": "Single-env GPU utilisation is low because one mjlab env at "
                    "50 Hz leaves the GPU mostly idle between kernels; this is "
                    "expected and is a single-environment measurement, not a "
                    "batched-throughput claim.",
        },
        "per_clip": _clip_rows(smokes),
        "smoke_records": {
            name: {
                "mode": rec.get("mode"),
                "submission_eligible": rec.get("submission_eligible"),
                "clips": [r["clip"] for r in rec["checkpoints"][0]["per_clip"]],
                "runtime_parity": (rec["evaluation"].get("runtime_parity") or {}).get("status"),
                "evaluation_wall_s": rec["evaluation"].get("evaluation_wall_s"),
            }
            for name, rec in smokes.items()
        },
        "blocked_items": [
            "RTX PRO 6000 Blackwell / sm_120 GPU: Colab G4 assignment HTTP 503.",
            "sm_120 compatibility of frozen warp-lang 1.16.0 / CUDA 13.0 stack: "
            "cannot be tested without a Blackwell device.",
            "Formal 40-clip evaluation and V2 leaderboard scores: out of scope "
            "(no training, no checkpoint changes, no 40-clip run).",
        ],
        "artifacts_sha256": {name: _sha256(V2_REPORTS / name) for name in artifacts},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(V2_REPORTS / "gpu_performance.json"))
    args = ap.parse_args(argv)
    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={report['gpu_runtime_status']} "
          f"gpu={report['verified_hardware_runtime_detected']['gpu_name']} "
          f"target={report['target_hardware']['status']} "
          f"steady_fps={report['steady_state_simulation']['measured_fps']:.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
