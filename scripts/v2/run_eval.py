#!/usr/bin/env python3
"""Run the V2 (original-evaluation-aligned) G1 LAFAN1 evaluation on CUDA.

    python scripts/v2/run_eval.py --dry-run                    # GPU-free identity check
    python scripts/v2/run_eval.py --smoke-policy replay --clips walk1_subject1 \
        --out reports/v2/smoke_gpu.json                       # 1-2 clip runtime smoke
    python scripts/v2/run_eval.py --policy-file my_algo/infer.py:make_policy \
        --checkpoint my_algo/ckpt.pt --algorithm MyAlgo --training-commit <sha> \
        --out reports/v2/<name>.json                         # formal 40-clip benchmark

Two **distinct** modes (task prompt §4):

* ``smoke``: 1-2 clips, scripted smoke policy, validates CUDA physics + scoring
  execution, writes an independent JSON with ``submission_eligible=false`` and
  is validated by ``benchmark.v2.spec.audit_v2_smoke``.  Never a leaderboard
  entry.  Exit 0 on success, non-zero on any failure.
* ``benchmark``: the exact 40 protocol clips, validated by the untouched
  ``audit_v2_record`` (all Integrity Gates still apply).  Never run by this
  task's smoke acceptance.

V2's formal evaluator is CUDA-only.  ``--device`` defaults to ``cuda:0`` and a
missing CUDA/Warp/MuJoCo Warp backend raises ``CudaRuntimeUnavailable`` --
there is no silent CPU fallback.  Hardware provenance always comes from
runtime detection; ``--gpu-name`` is only a *declared* label and is never used
as proof (task prompt §9).

V2 shares the V1 policy contract (``benchmark/evaluator/policy_api.py``),
including PR #13's reference-frame indexing fix.  V1 is untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from benchmark.v2 import spec as S  # noqa: E402


class SmokePolicy:
    """Scripted policies for a runtime smoke only; never a submission."""

    kind = "v2_smoke_scripted"

    def __init__(self, mode: str):
        self._mode = mode
        self._context = None

    def reset(self, context) -> None:
        self._context = context

    def act(self, step):
        if self._mode == "hold":
            return np.zeros(len(self._context.default_joint_pos))
        target = self._context.reference.joint_pos[step.reference_index]
        return (target - self._context.default_joint_pos) / self._context.action_scale


def _load_factory(spec_str: str):
    target, _, name = spec_str.partition(":")
    name = name or "make_policy"
    path = Path(target)
    if path.suffix == ".py" or path.exists():
        mspec = importlib.util.spec_from_file_location(f"_v2_policy_{path.stem}", path)
        if mspec is None or mspec.loader is None:
            raise SystemExit(f"cannot import policy module: {path}")
        module = importlib.util.module_from_spec(mspec)
        sys.modules[mspec.name] = module
        mspec.loader.exec_module(module)
    else:
        module = __import__(target, fromlist=[name])
    factory = getattr(module, name, None)
    if not callable(factory):
        raise SystemExit(f"{spec_str}: no callable {name!r}")
    return factory


def _resolve_mode(args, ap) -> str:
    if args.mode:
        mode = args.mode
    elif args.smoke_policy or args.clips or args.limit:
        mode = "smoke"
    else:
        mode = "benchmark"
    if mode == "smoke":
        if not args.smoke_policy:
            ap.error("smoke mode requires --smoke-policy (scripted smoke policy)")
        if args.checkpoint or args.policy_file:
            ap.error("smoke mode cannot use --checkpoint/--policy-file")
    else:
        if args.smoke_policy:
            ap.error("benchmark mode cannot use --smoke-policy")
        if args.limit:
            ap.error("benchmark mode cannot use --limit (all 40 clips are required)")
        if not (args.checkpoint or args.policy_file):
            ap.error("benchmark mode requires --checkpoint and/or --policy-file")
    return mode


def _resolve_clips(args, mode: str, ap) -> list[str]:
    full = list(S.load_protocol_v2()["clip_ids"])
    if args.clips:
        clips = list(args.clips)
        unknown = set(clips) - set(full)
        if unknown:
            ap.error(f"unknown clip ids: {sorted(unknown)}")
    else:
        clips = full
    if args.limit:
        clips = clips[:args.limit]
    if mode == "smoke":
        if not S.SMOKE_MIN_CLIPS <= len(clips) <= S.SMOKE_MAX_CLIPS:
            ap.error(f"smoke requires {S.SMOKE_MIN_CLIPS}-{S.SMOKE_MAX_CLIPS} clips, got {len(clips)}")
    else:
        if len(clips) != 40 or set(clips) != set(full) or len(set(clips)) != 40:
            ap.error(f"benchmark requires the full 40 protocol clips, got {len(clips)}")
    return clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the V2 identity / rubric and exit (no GPU needed)")
    ap.add_argument("--mode", choices=("smoke", "benchmark"), default=None,
                    help="explicit mode; auto-detected from --smoke-policy/--clips")
    ap.add_argument("--policy-file", metavar="PATH:FACTORY")
    ap.add_argument("--checkpoint")
    ap.add_argument("--smoke-policy", choices=("hold", "replay"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--algorithm", default="unknown")
    ap.add_argument("--training-commit", default="0" * 40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-name", default="declared-only",
                    help="declared label only; NEVER used as hardware evidence")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--device", default="cuda:0",
                    help="CUDA device (default cuda:0); CPU is rejected")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-runtime-parity", action="store_true",
                    help="skip GPU runtime parity (smoke mode only)")
    ap.add_argument("--training-wall-s", type=float, default=0.0)
    ap.add_argument("--setup-wall-s", type=float, default=0.0)
    ap.add_argument("--cache-references", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    identity = S.evaluator_identity()
    if args.dry_run:
        cfg = S.load_protocol_v2()
        probe = None
        try:
            from benchmark.v2 import runtime as RT  # noqa: PLC0415
            probe = RT.probe_cuda(args.device)
        except Exception as exc:
            probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({"identity": identity,
                          "cuda_probe": probe,
                          "reward_terms": cfg["reward"]["terms"],
                          "excluded_reward_terms": cfg["reward"]["excluded_terms"],
                          "termination_terms": cfg["termination"]["terms"],
                          "excluded_termination_terms": cfg["termination"]["excluded_terms"],
                          "normalization": cfg["normalization"],
                          "modes": {"smoke": f"{S.SMOKE_MIN_CLIPS}-{S.SMOKE_MAX_CLIPS} clips",
                                    "benchmark": "40 clips"}}, indent=2))
        return 0

    mode = _resolve_mode(args, ap)
    clips = _resolve_clips(args, mode, ap)
    if args.out is None:
        args.out = ("reports/v2/smoke_gpu.json" if mode == "smoke"
                    else "reports/v2/eval.json")
    profile = not args.no_profile

    # Heavy imports are intentionally deferred so --dry-run works without CUDA.
    from benchmark.evaluator import policy as policy_mod  # noqa: PLC0415
    from benchmark.evaluator import spec as v1spec  # noqa: PLC0415
    from benchmark.evaluator.constants import ROBOT_DOF  # noqa: PLC0415
    from benchmark.v2 import parity as PARITY  # noqa: PLC0415
    from benchmark.v2 import runner as v2runner  # noqa: PLC0415
    from benchmark.v2 import runtime as RT  # noqa: PLC0415

    # P0: hard-fail (non-zero exit) instead of a silent CPU fallback.
    try:
        probe = RT.probe_cuda(args.device)
    except RT.CudaRuntimeUnavailable as exc:
        print(f"BLOCKED: CUDA runtime unavailable: {exc}", file=sys.stderr)
        return 3
    if not args.quiet:
        print("=== CUDA hardware (runtime detection) ===")
        print(RT.format_hardware_report(probe))
        print(f"mode={mode} clips={len(clips)} device={args.device}", flush=True)

    started = time.perf_counter()
    info = v1spec.build_model()
    if mode == "smoke":
        pol, ckpt_sha = SmokePolicy(args.smoke_policy), "0" * 64
        if args.algorithm == "unknown":
            args.algorithm = "smoke-scripted"
    else:
        path = Path(args.checkpoint)
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        ckpt_sha = policy_mod.sha256_file(path)
        if args.policy_file:
            pol = _load_factory(args.policy_file)(str(path))
        else:
            pol = policy_mod.load_policy(path, info)
    setup_s = args.setup_wall_s + (time.perf_counter() - started)

    # ---- GPU runtime parity (real RewardManager / TerminationManager) ----
    parity_report: dict = {"status": "SKIPPED"}
    if mode == "smoke" and not args.no_runtime_parity:
        try:
            parity_report = PARITY.run_gpu_check(clips[0], device=args.device, steps=6)
        except Exception as exc:
            print(f"BLOCKED: GPU runtime parity could not run: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        if not args.quiet:
            print(f"runtime parity: {parity_report['status']} "
                  f"reward max|err|={parity_report['checks']['reward']['max_abs_error']:.3e} "
                  f"termination={parity_report['checks']['termination']['status']}", flush=True)

    def progress(res):
        if not args.quiet:
            print(f"[{res.clip:24s}] ret={res.normalized_return:+.5f} "
                  f"{'FAILED' if res.failed else 'ok':7s} "
                  f"alive={res.alive_steps}/{res.steps} {res.wall_s:.1f}s "
                  f"fps={res.performance.get('steady_simulation_fps', 0):.1f}", flush=True)

    out = Path(args.out)
    ckpt_dir = out.parent / f".parts_{out.stem}"
    recovery_meta = {
        "algorithm": args.algorithm,
        "checkpoint_sha256": ckpt_sha,
        "protocol_sha256": identity["protocol_sha256"],
        "device": args.device,
        "evaluator_git_sha": identity.get("evaluator_git_sha") or "unknown",
    }

    telemetry = RT.GpuTelemetry(interval=0.4).start()
    try:
        results = v2runner.evaluate_all(info, pol, clips, use_cache=args.cache_references,
                                        progress=progress, device=args.device,
                                        profile=profile, checkpoint_dir=ckpt_dir,
                                        recovery_metadata=recovery_meta)
        eval_wall_s = time.perf_counter() - started - setup_s
    finally:
        telemetry_summary = telemetry.stop()

    runtime_info = dict(results[0].runtime) if results and results[0].runtime else {}
    measured = (parity_report.get("performance", {}).get("steady_simulation_fps")
                if isinstance(parity_report, dict) else None)
    provenance = RT.collect_provenance(
        probe=probe, runtime=runtime_info,
        kernel=(parity_report.get("kernel_execution") if isinstance(parity_report, dict) else None),
        telemetry=telemetry_summary,
        performance={"per_clip": {r.clip: r.performance for r in results},
                     "evaluation_wall_s": eval_wall_s,
                     "steady_simulation_fps": measured},
        extra={"evaluation_mode": mode},
    )
    record = v2runner.build_record(
        algorithm=args.algorithm, training_commit=args.training_commit,
        seed=args.seed, gpu_name=args.gpu_name, gpu_count=args.gpu_count,
        checkpoint_sha256=ckpt_sha, training_wall_s=args.training_wall_s,
        setup_wall_s=setup_s, results=results, mode=mode,
        provenance=provenance, telemetry=telemetry_summary,
        extra={"clip_subset": None if len(clips) == 40 else clips,
               "protocol_sha256": identity["protocol_sha256"],
               "robot_dof": ROBOT_DOF,
               "hardware_report": RT.format_hardware_report(probe, runtime_info),
               "runtime_parity": parity_report,
               "evaluation_wall_s": eval_wall_s,
               "cuda_init_s": (parity_report.get("timing", {}).get("cuda_env_build_s")
                               if isinstance(parity_report, dict) else None),
               "version_drift": probe["version_drift"]},
    )
    # Explicit top-level markers: smoke is never a leaderboard submission.
    target_ok = bool(probe["target_gpu_match"]) and bool(probe["sm_120_confirmed"])
    submission_ok = bool(mode == "benchmark" and target_ok)
    comparison_ok = bool(mode == "benchmark")
    record["mode"] = mode
    record["target_hardware_met"] = target_ok
    record["submission_eligible"] = submission_ok
    record["comparison_valid"] = comparison_ok
    record["evaluation"]["submission_eligible"] = submission_ok
    record["evaluation"]["comparison_valid"] = comparison_ok
    record["evaluation"]["evaluation_wall_s"] = eval_wall_s
    record["evaluation"]["mode"] = mode
    # "Ran on a GPU" is not the same as "ran on the task's target GPU". Stamp
    # the scope so an L4/sm_89 or A100/sm_80 artifact can never be read as the Blackwell
    # acceptance (same rule as gpu_runtime_check.acceptance_status).
    record["report_scope"] = ("acceptance" if target_ok
                              else ("controlled_baseline_non_target_hardware" if mode == "benchmark"
                                    else "preflight_non_target_hardware"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if mode == "smoke":
        audit = S.audit_v2_smoke(record)
        parity_ok = (parity_report.get("status") == "PASS"
                     or args.no_runtime_parity)
        ok = (audit["status"] == "SMOKE_VALID" and parity_ok
              and not S.is_submission_eligible(record))
        status = "SMOKE_PASS" if ok else "SMOKE_FAIL"
    else:
        audit = S.audit_v2_record(record)
        is_sub = S.is_submission_eligible(record)
        is_comp = S.is_comparison_valid(record)
        if target_ok:
            ok = audit["status"].startswith("FORMAT_VALID") and is_sub
            status = "BENCHMARK_SUBMISSION_VALID" if ok else "BENCHMARK_INVALID"
        else:
            ok = audit["status"].startswith("FORMAT_VALID") and is_comp and not is_sub
            status = "BENCHMARK_COMPARISON_VALID" if ok else "BENCHMARK_INVALID"

    if not args.quiet:
        print(f"\nprotocol        : {S.PROTOCOL_ID}")
        print(f"mode            : {mode} (submission_eligible={record['submission_eligible']})")
        print(f"clips           : {len(results)}")
        print(f"aggregate       : {S.aggregate(r.normalized_return for r in results):+.5f} (unclamped)")
        print(f"failed clips    : {S.failed_clips(r.failed for r in results)}")
        print(f"wrote           : {out}")
        print(f"audit           : {audit['status']} -> {status}")
        for error in audit["errors"]:
            print(f"  - {error}")
        for r in results:
            p = r.performance
            print(f"  {r.clip}: steady_fps={p.get('steady_simulation_fps', 0):.1f} "
                  f"wall={r.wall_s:.2f}s warmup={p.get('warmup_wall_s', 0):.2f}s "
                  f"build={p.get('cuda_env_build_s', 0):.2f}s")
        print(f"telemetry       : {json.dumps(telemetry_summary['gpu_utilization_pct'])}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
