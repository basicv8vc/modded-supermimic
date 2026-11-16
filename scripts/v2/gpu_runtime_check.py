#!/usr/bin/env python3
"""Standalone GPU runtime integration check for Benchmark V2.

    python scripts/v2/gpu_runtime_check.py --clip walk1_subject1

Writes ``reports/v2/gpu_runtime_status.json`` with status PASS / FAIL /
BLOCKED.  A host without CUDA is reported **BLOCKED**, never PASS (task
prompt §10).

Exit codes: 0 = PASS on the target hardware, 4 = parity PASS but the GPU is
not the RTX PRO 6000 / sm_120 target (``BLOCKED_TARGET_HARDWARE``),
2 = FAIL, 3 = BLOCKED.

This is the entry point used for the CUDA acceptance on the GPU runtime; the
same checks are exposed as unit tests in ``tests/test_v2_gpu_runtime.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.v2 import runtime as RT  # noqa: E402
from benchmark.v2 import spec as S  # noqa: E402


def acceptance_status(parity_status: str, target_gpu_match: bool,
                      sm_120_confirmed: bool) -> tuple[str, bool]:
    """Separate "parity passed" from "the task's acceptance is met".

    ``checks`` answers *"does the V2 math match the real GPU runtime?"*.
    Acceptance additionally requires the RTX PRO 6000 / sm_120 target.  A parity
    pass on an L4 is real evidence but must never be reported as PASS, otherwise
    the report silently claims hardware that was never used.
    """
    target_ok = bool(target_gpu_match) and bool(sm_120_confirmed)
    if parity_status != "PASS":
        return parity_status, target_ok
    return ("PASS" if target_ok else "BLOCKED_TARGET_HARDWARE"), target_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="reports/v2/gpu_runtime_status.json")
    ap.add_argument("--scope", default="acceptance",
                    choices=("acceptance", "preflight"),
                    help="preflight = run on non-target hardware; never counts as "
                         "the RTX PRO 6000 acceptance artifact")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result: dict = {"status": "BLOCKED", "checked_at_utc": stamp,
                    "device_requested": args.device, "clip": args.clip,
                    "report_scope": args.scope,
                    "source_fingerprint": RT.source_fingerprint(),
                    "protocol_sha256": S.protocol_sha256(),
                    "checks": {}, "errors": []}

    # --- CUDA availability: no GPU => BLOCKED, explicitly NOT PASS ---------
    try:
        probe = RT.probe_cuda(args.device)
    except RT.CudaRuntimeUnavailable as exc:
        result["errors"].append(f"CUDA unavailable: {exc}")
        result["reason"] = "BLOCKED: no CUDA/Warp-CUDA runtime on this host"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"])
        print(f"wrote {out_path}")
        return 3
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"CUDA probe error: {type(exc).__name__}: {exc}")
        result["reason"] = "BLOCKED: CUDA probe failed"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"])
        return 3

    result["hardware"] = RT.format_hardware_report(probe)
    result["versions"] = probe["versions"]
    result["version_drift"] = probe["version_drift"]
    result["target_gpu_match"] = probe["target_gpu_match"]
    result["sm_120_confirmed"] = probe["sm_120_confirmed"]
    print("=== CUDA hardware (runtime detection) ===")
    print(result["hardware"])

    # --- Real mjlab GPU runtime parity ------------------------------------
    from benchmark.v2 import parity as PARITY  # noqa: PLC0415
    try:
        report = PARITY.run_gpu_check(args.clip, device=args.device, steps=args.steps)
    except RT.CudaRuntimeUnavailable as exc:
        result["errors"].append(f"GPU runtime unavailable: {exc}")
        result["reason"] = "BLOCKED: mjlab GPU runtime could not be constructed"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"])
        return 3
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["status"] = "FAIL"
        result["reason"] = "GPU runtime check raised an exception"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["reason"], exc)
        return 2

    result["checks"] = {k: {"status": v.get("status"), "checks": v.get("checks")}
                        for k, v in report["checks"].items()}
    result["runtime"] = report.get("runtime")
    # Refresh the human-readable hardware report with *constructed-env*
    # runtime detection (CUDA step graph, physics array device, real backend).
    result["hardware"] = RT.format_hardware_report(probe, report.get("runtime"))
    result["kernel_execution"] = report.get("kernel_execution")
    result["telemetry"] = report.get("telemetry")
    result["performance"] = report.get("performance")
    result["timing"] = report.get("timing")
    result["no_cpu_fallback"] = report.get("no_cpu_fallback")
    result["reward_max_abs_error"] = report["checks"]["reward"]["max_abs_error"]
    result["reward_max_rel_error"] = report["checks"]["reward"]["max_rel_error"]
    result["reward_per_term_max_abs_error"] = report.get("reward_per_term_max_abs_error")
    result["reward_static"] = report["checks"]["reward"]["static"]
    result["termination_cases"] = report["checks"]["termination"]["cases"]
    result["frame_alignment"] = report["checks"]["frame_alignment"]
    result["status"] = report["status"]
    # ``checks`` answers "does the V2 math match the real GPU runtime?".
    # Acceptance *also* requires the task's target hardware: a parity pass on an
    # L4 / sm_89 is genuinely useful evidence, but it is NOT a Blackwell pass.
    # Keep the two verdicts separate so neither can be mistaken for the other.
    target_ok = bool(result.get("target_gpu_match")) and bool(result.get("sm_120_confirmed"))
    result["target_hardware_met"] = target_ok
    result["acceptance_status"], target_ok = acceptance_status(
        result["status"], result.get("target_gpu_match"), result.get("sm_120_confirmed"))

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"\nreward   : {report['checks']['reward']['status']} "
          f"max|err|={result['reward_max_abs_error']:.3e} "
          f"max rel={result['reward_max_rel_error']:.3e}")
    print(f"terminate: {report['checks']['termination']['status']}")
    print(f"frame    : {report['checks']['frame_alignment']['status']}")
    print(f"reanchor : {report['checks']['reference_reanchor']['status']} "
          f"max|err|={report['checks']['reference_reanchor']['max_abs_error']:.3e}")
    print(f"identity : {report['checks']['identity_v1']['status']}")
    print(f"fallback : {report['checks']['cpu_fallback_rejection']['status']}")
    print(f"kernel   : {json.dumps(report.get('kernel_execution'))}")
    print(f"per-term : {json.dumps(result['reward_per_term_max_abs_error'])}")
    perf = report["performance"]
    print(f"fps      : {perf.get('steady_simulation_fps', 0):.1f} "
          f"({perf.get('steady_steps')} steady steps, "
          f"warmup={perf.get('warmup_wall_s', 0):.3f}s, "
          f"rollout={perf.get('total_rollout_wall_s', 0):.2f}s)")
    print(f"timing   : {json.dumps(result['timing'])}")
    print("=== Runtime-verified hardware (post-construction) ===")
    print(result["hardware"])
    if not target_ok:
        print("\n!! TARGET HARDWARE NOT MET: parity verdict is "
              f"{result['status']} but GPU is {result['hardware'].splitlines()[1].strip()}"
              " -- this run is NOT the RTX PRO 6000 / sm_120 acceptance.")
    print(f"status   : {result['status']} -> acceptance {result['acceptance_status']}"
          f" -> wrote {out_path}")
    if result["status"] != "PASS":
        return 2
    return 0 if target_ok else 4


if __name__ == "__main__":
    sys.exit(main())
