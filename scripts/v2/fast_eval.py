#!/usr/bin/env python3
"""Fast V2 evaluation: vectorized batched simulation (Option B) and parallel workers (Option A).

Backends:

* ``batched`` (Option B, default when ``--workers`` is ``auto``): runs all clips
  inside a single CUDA ``ManagerBasedRlEnv`` with ``num_envs=B`` (up to 40
  concurrent clips), packed GPU reference bank, LPT dynamic scheduling, and
  zero-copy GPU policy inference for SuperMimic / NablaMimic / Replay (plus
  slot fallback for arbitrary custom policies).
* ``workers`` (Option A): shards clips across independent OS worker processes,
  each running the untouched serial ``num_envs=1`` path.

    # Option B: batched evaluation (default, 1-4 min per model on A100)
    python scripts/v2/fast_eval.py --policy-file benchmark/policies/supermimic.py:make_policy \
        --checkpoint ckpt.pt --algorithm SuperMimic-vibed --training-commit <sha> \
        --training-wall-s 5212.8 --seed 0 --cache-references \
        --parts-dir /content/drive/.../parts --out reports/v2/a100_supermimic_seed0.json

    # Option A: worker-process sharding (explicit --workers N or --backend workers)
    python scripts/v2/fast_eval.py --backend workers --workers 12 ...

Numerical parity vs the serial evaluator is checked per clip with
``scripts/v2/compare_evals.py``.  Scoring math, physics config and
protocol_v2.json are never touched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.v2 import spec as S  # noqa: E402


# --------------------------------------------------------------------------
# policy factory loader (identical semantics to scripts/v2/run_eval.py)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# worker entry (subprocess; one per shard)
# --------------------------------------------------------------------------
def run_worker(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    from benchmark.evaluator import policy as policy_mod  # noqa: PLC0415
    from benchmark.evaluator import spec as v1spec  # noqa: PLC0415
    from benchmark.v2 import runner as v2runner  # noqa: PLC0415

    info = v1spec.build_model()
    ckpt = spec["checkpoint"]
    if isinstance(ckpt, str) and ckpt.lower() in {"none", "null", "-"}:
        ckpt = None
    pol = _load_factory(spec["policy_file"])(ckpt)
    t0 = time.perf_counter()
    results = v2runner.evaluate_all(
        info, pol, spec["clips"], use_cache=spec["use_cache"],
        progress=lambda r: print(f"[w{spec['worker_id']} {r.clip:24s}] "
                                f"ret={r.normalized_return:+.5f} "
                                f"{'FAILED' if r.failed else 'ok':7s} "
                                f"alive={r.alive_steps}/{r.steps} {r.wall_s:.1f}s "
                                f"fps={r.performance.get('steady_simulation_fps', 0):.1f}",
                                flush=True),
        device=spec["device"], profile=False,
        checkpoint_dir=Path(spec["parts_dir"]) if spec["parts_dir"] else None,
        recovery_metadata=spec["recovery"],
        max_steps=spec.get("probe_steps"),
    )
    steady_steps = sum(int(r.performance.get("steady_steps", 0)) for r in results)
    steady_wall = sum(float(r.performance.get("steady_wall_s", 0.0)) for r in results)
    payload = {
        "worker_id": spec["worker_id"],
        "clips": spec["clips"],
        "wall_s": time.perf_counter() - t0,
        "steady_steps": steady_steps,
        "steady_wall_s": steady_wall,
        "results": [r.to_dict() for r in results],
    }
    out = Path(spec["shard_out"])
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(out)
    return 0


# --------------------------------------------------------------------------
# parent: shard, spawn, merge, record
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# worker-count auto configuration (VRAM + CPU aware)
# --------------------------------------------------------------------------
# Measured per-worker GPU footprint on A100 (telemetry, 2026-11-13):
#   w1 peak 658 MiB total; w12 peak 3579 MiB total (~300 MiB/worker at
#   concurrent peak).  768 MiB/worker is a conservative ceiling that also
#   covers per-process policy/checkpoint buffers.
WORKER_VRAM_MIB = 768.0
VRAM_BUDGET_FRAC = 0.8   # leave headroom for fragmentation / driver
CPU_RESERVE = 0          # measured optimum on A100 VM: w == nproc (w12=382
                         # steps/s; w16 oversubscribed with no gain), and the
                         # parent only p.wait()s while workers run.


def probe_vram_mib(device: str = "cuda:0") -> tuple[float, float] | None:
    """Return ``(total_mib, free_mib)``; ``None`` when unprobeable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        total_s, free_s = out.stdout.strip().splitlines()[0].split(",")[:2]
        return float(total_s), float(free_s)
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415
        free_b, total_b = torch.cuda.mem_get_info(device)
        return total_b / 2**20, free_b / 2**20
    except Exception:
        return None


def resolve_max_workers(device: str = "cuda:0", n_clips: int = 40,
                        per_worker_mib: float = WORKER_VRAM_MIB) -> dict:
    """Pick the max useful worker count from VRAM and CPU budgets.

    Returns the chosen count plus the reasoning so the decision is auditable
    in the result record.  VRAM-unprobeable environments degrade to the CPU
    limit rather than failing.
    """
    vram = probe_vram_mib(device)
    if vram is not None:
        total_mib, free_mib = vram
        vram_budget = free_mib * VRAM_BUDGET_FRAC
        w_vram = max(1, int(vram_budget // per_worker_mib))
    else:
        total_mib = free_mib = float("nan")
        w_vram = 1 << 20  # unbounded; CPU will bind
    cpus = os.cpu_count() or 1
    w_cpu = max(1, cpus - CPU_RESERVE)
    w = min(w_vram, w_cpu, max(1, n_clips))
    return {
        "workers": int(w),
        "per_worker_mib": per_worker_mib,
        "vram_total_mib": total_mib,
        "vram_free_mib": free_mib,
        "w_vram_limit": w_vram if w_vram < (1 << 20) else None,
        "cpu_count": cpus,
        "w_cpu_limit": w_cpu,
        "binding_constraint": ("vram" if w_vram <= w_cpu else "cpu"),
        "n_clips_cap": n_clips,
    }


def _shard_clips(clips: list[str], workers: int) -> list[list[str]]:
    return [clips[i::workers] for i in range(workers)]


def run_parallel(args, clips: list[str], recovery: dict) -> tuple[dict, dict]:
    from benchmark.evaluator import constants as C  # noqa: PLC0415
    from benchmark.evaluator import policy as policy_mod  # noqa: PLC0415
    from benchmark.v2 import runner as v2runner  # noqa: PLC0415
    from benchmark.v2 import runtime as RT  # noqa: PLC0415

    started = time.perf_counter()
    ckpt_sha = policy_mod_sha(args.checkpoint)

    probe = RT.probe_cuda(args.device)
    if not args.quiet:
        print(RT.format_hardware_report(probe))
        print(f"fast_eval: workers={args.workers} clips={len(clips)} "
              f"device={args.device}", flush=True)

    out = Path(args.out)
    work = out.parent / f".fast_{out.stem}"
    work.mkdir(parents=True, exist_ok=True)
    shards = _shard_clips(clips, args.workers)
    if not args.quiet:
        print("shards:", {i: len(s) for i, s in enumerate(shards)}, flush=True)

    telemetry = RT.GpuTelemetry(interval=0.4).start()
    procs: list[subprocess.Popen] = []
    shard_outs: list[Path] = []
    for i, shard in enumerate(shards):
        if not shard:
            shard_outs.append(None)  # type: ignore[arg-type]
            continue
        parts_dir = (Path(args.parts_dir) / f"w{i}" if args.parts_dir
                     else work / "parts" / f"w{i}")
        parts_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "worker_id": i,
            "clips": shard,
            "policy_file": args.policy_file,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "use_cache": bool(args.cache_references),
            "parts_dir": str(parts_dir),
            "recovery": recovery,
            "shard_out": str(work / f"shard_{i}.json"),
            "probe_steps": args.probe_steps,
        }
        spec_path = work / f"shard_{i}.spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        log_path = work / f"shard_{i}.log"
        log_f = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        p = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()),
             "--_worker", str(spec_path)],
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append(p)
        shard_outs.append(work / f"shard_{i}.json")

    wall0 = time.perf_counter()
    failures = []
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failures.append((i, rc))
    eval_wall_s = time.perf_counter() - wall0
    telemetry_summary = telemetry.stop()
    if failures:
        for i, rc in failures:
            log = work / f"shard_{i}.log"
            tail = log.read_text(encoding="utf-8")[-2000:] if log.exists() else ""
            print(f"--- worker {i} rc={rc} log tail ---\n{tail}", file=sys.stderr)
        raise SystemExit(f"workers failed: {failures}")

    # merge in protocol clip order
    by_clip: dict[str, dict] = {}
    steady_steps_total = 0
    steady_wall_total = 0.0
    steady_wall_max = 0.0
    for so in shard_outs:
        if so is None:
            continue
        payload = json.loads(so.read_text(encoding="utf-8"))
        steady_steps_total += int(payload.get("steady_steps", 0))
        w_wall = float(payload.get("steady_wall_s", 0.0))
        steady_wall_total += w_wall
        steady_wall_max = max(steady_wall_max, w_wall)
        for row in payload["results"]:
            by_clip[row["clip"]] = row
    missing = [c for c in clips if c not in by_clip]
    if missing:
        raise SystemExit(f"shard merge: missing clips {missing}")
    results = [v2runner.V2ClipResult.from_dict(by_clip[c]) for c in clips]

    runtime_info = dict(results[0].runtime) if results and results[0].runtime else {}
    total_steps = sum(r.steps for r in results)
    aggregate_steps_per_s = total_steps / eval_wall_s if eval_wall_s > 0 else 0.0

    provenance = RT.collect_provenance(
        probe=probe, runtime=runtime_info, kernel=None,
        telemetry=telemetry_summary,
        performance={"per_clip": {r.clip: r.performance for r in results},
                     "evaluation_wall_s": eval_wall_s,
                     "steady_simulation_fps": aggregate_steps_per_s},
        extra={"evaluation_mode": "benchmark"},
    )
    record = v2runner.build_record(
        algorithm=args.algorithm, training_commit=args.training_commit,
        seed=args.seed, gpu_name=args.gpu_name, gpu_count=args.gpu_count,
        checkpoint_sha256=ckpt_sha, training_wall_s=args.training_wall_s,
        setup_wall_s=args.setup_wall_s, results=results, mode="benchmark",
        provenance=provenance, telemetry=telemetry_summary,
        extra={
            "clip_subset": None if len(clips) == 40 else clips,
            "protocol_sha256": S.protocol_sha256(),
            "robot_dof": C.ROBOT_DOF,
            "hardware_report": RT.format_hardware_report(probe, runtime_info),
            "runtime_parity": {"status": "SKIPPED"},
            "evaluation_wall_s": eval_wall_s,
            "cuda_init_s": None,
            "version_drift": probe["version_drift"],
            "fast_eval": {
                "implementation": "scripts/v2/fast_eval.py",
                "option": "A_independent_parallel_workers",
                "workers": args.workers,
                "num_envs_per_worker": 1,
                "worker_resolution": getattr(args, "worker_resolution", None),
                "shards": {i: len(s) for i, s in enumerate(shards)},
                "eval_wall_s": eval_wall_s,
                "total_control_steps": total_steps,
                "aggregate_steps_per_s": aggregate_steps_per_s,
                "parts_dir": args.parts_dir or str(work / "parts"),
            },
        },
    )
    target_ok = bool(probe["target_gpu_match"]) and bool(probe["sm_120_confirmed"])
    record["mode"] = "benchmark"
    record["target_hardware_met"] = target_ok
    record["submission_eligible"] = bool(target_ok)  # A100 -> False
    record["comparison_valid"] = True
    record["evaluation"]["submission_eligible"] = record["submission_eligible"]
    record["evaluation"]["comparison_valid"] = True
    record["evaluation"]["evaluation_wall_s"] = eval_wall_s
    record["evaluation"]["mode"] = "benchmark"
    record["report_scope"] = ("acceptance" if target_ok
                              else "controlled_baseline_non_target_hardware")
    summary = {
        "workers": args.workers,
        "clips": len(clips),
        "total_control_steps": total_steps,
        "eval_wall_s": eval_wall_s,
        "aggregate_steps_per_s": aggregate_steps_per_s,
        "steady_steps": steady_steps_total,
        "steady_wall_s": steady_wall_max,
        "steady_wall_s_sum": steady_wall_total,
        # concurrent workers: aggregate throughput over the overlapping window
        "steady_aggregate_steps_per_s": (steady_steps_total / steady_wall_max
                                         if steady_wall_max > 0 else 0.0),
        "probe_steps": args.probe_steps,
        "gpu_utilization_pct": telemetry_summary.get("gpu_utilization_pct"),
    }
    return record, summary


def run_batched(args, clips: list[str], recovery: dict) -> tuple[dict, dict]:
    from benchmark.evaluator import constants as C  # noqa: PLC0415
    from benchmark.evaluator import spec as v1spec  # noqa: PLC0415
    from benchmark.v2 import batched as v2batched  # noqa: PLC0415
    from benchmark.v2 import runner as v2runner  # noqa: PLC0415
    from benchmark.v2 import runtime as RT  # noqa: PLC0415

    ckpt_sha = policy_mod_sha(args.checkpoint)
    probe = RT.probe_cuda(args.device)
    batch_res = v2batched.resolve_batch_envs(args.device, len(clips), requested=args.batch_envs)
    if not args.quiet:
        print(RT.format_hardware_report(probe))
        print(
            f"fast_eval(B)    : batch_envs={batch_res['batch_envs']} clips={len(clips)} "
            f"policy_device={args.policy_device} device={args.device}",
            flush=True,
        )

    info = v1spec.build_model()
    ckpt = args.checkpoint
    if isinstance(ckpt, str) and ckpt.lower() in {"none", "null", "-"}:
        ckpt = None
    factory = _load_factory(args.policy_file)

    out = Path(args.out)
    parts_dir = Path(args.parts_dir) if args.parts_dir else (out.parent / f".fast_{out.stem}" / "parts_batched")
    parts_dir.mkdir(parents=True, exist_ok=True)

    def _on_clip(r):
        if not args.quiet:
            print(
                f"[b{batch_res['batch_envs']} {r.clip:24s}] "
                f"ret={r.normalized_return:+.5f} "
                f"{'FAILED' if r.failed else 'ok':7s} "
                f"alive={r.alive_steps}/{r.steps} {r.wall_s:.1f}s",
                flush=True,
            )

    telemetry = RT.GpuTelemetry(interval=0.4).start()
    wall0 = time.perf_counter()
    try:
        results, batch_meta = v2batched.evaluate_batched(
            info,
            lambda: factory(ckpt),
            clips,
            use_cache=bool(args.cache_references),
            progress=_on_clip,
            device=args.device,
            batch_envs=args.batch_envs,
            policy_device=args.policy_device,
            checkpoint_dir=parts_dir,
            recovery_metadata=recovery,
            max_steps=args.probe_steps,
        )
        eval_wall_s = time.perf_counter() - wall0
    finally:
        telemetry_summary = telemetry.stop()

    runtime_info = dict(results[0].runtime) if results and results[0].runtime else {}
    total_steps = sum(r.steps for r in results)
    executed_steps = int(batch_meta.get("executed_steps", sum(r.alive_steps for r in results)))
    rollout_wall_s = float(batch_meta.get("rollout_wall_s", eval_wall_s))
    aggregate_steps_per_s = executed_steps / eval_wall_s if eval_wall_s > 0 else 0.0
    steady_steps = int(batch_meta.get("steady_steps", 0))
    steady_wall = float(batch_meta.get("steady_wall_s", 0.0))
    steady_fps = float(batch_meta.get("steady_aggregate_steps_per_s", 0.0))

    provenance = RT.collect_provenance(
        probe=probe, runtime=runtime_info, kernel=None,
        telemetry=telemetry_summary,
        performance={"per_clip": {r.clip: r.performance for r in results},
                     "evaluation_wall_s": eval_wall_s,
                     "steady_simulation_fps": steady_fps or aggregate_steps_per_s},
        extra={"evaluation_mode": "benchmark"},
    )
    record = v2runner.build_record(
        algorithm=args.algorithm, training_commit=args.training_commit,
        seed=args.seed, gpu_name=args.gpu_name, gpu_count=args.gpu_count,
        checkpoint_sha256=ckpt_sha, training_wall_s=args.training_wall_s,
        setup_wall_s=args.setup_wall_s, results=results, mode="benchmark",
        provenance=provenance, telemetry=telemetry_summary,
        extra={
            "clip_subset": None if len(clips) == 40 else clips,
            "protocol_sha256": S.protocol_sha256(),
            "robot_dof": C.ROBOT_DOF,
            "hardware_report": RT.format_hardware_report(probe, runtime_info),
            "runtime_parity": {"status": "SKIPPED"},
            "evaluation_wall_s": eval_wall_s,
            "cuda_init_s": batch_meta.get("cuda_env_build_s"),
            "version_drift": probe["version_drift"],
            "fast_eval": {
                "implementation": "benchmark/v2/batched.py",
                "option": "B_batched_simulation",
                "workers": f"batched(num_envs={batch_meta.get('batch_envs', batch_res['batch_envs'])})",
                "batch_envs": batch_meta.get("batch_envs", batch_res["batch_envs"]),
                "batch_resolution": batch_meta.get("batch_resolution", batch_res),
                "policy_backend": batch_meta.get("policy_backend"),
                "eval_wall_s": eval_wall_s,
                "rollout_wall_s": rollout_wall_s,
                "total_control_steps": total_steps,
                "executed_control_steps": executed_steps,
                "aggregate_steps_per_s": aggregate_steps_per_s,
                "steady_aggregate_steps_per_s": steady_fps,
                "parts_dir": str(parts_dir),
            },
        },
    )
    target_ok = bool(probe["target_gpu_match"]) and bool(probe["sm_120_confirmed"])
    record["mode"] = "benchmark"
    record["target_hardware_met"] = target_ok
    record["submission_eligible"] = bool(target_ok)  # A100 -> False
    record["comparison_valid"] = True
    record["evaluation"]["submission_eligible"] = record["submission_eligible"]
    record["evaluation"]["comparison_valid"] = True
    record["evaluation"]["evaluation_wall_s"] = eval_wall_s
    record["evaluation"]["mode"] = "benchmark"
    record["report_scope"] = ("acceptance" if target_ok
                              else "controlled_baseline_non_target_hardware")
    summary = {
        "backend": "batched",
        "workers": 1,
        "batch_envs": batch_meta.get("batch_envs", batch_res["batch_envs"]),
        "policy_backend": batch_meta.get("policy_backend"),
        "clips": len(clips),
        "total_control_steps": total_steps,
        "executed_control_steps": executed_steps,
        "eval_wall_s": eval_wall_s,
        "rollout_wall_s": rollout_wall_s,
        "aggregate_steps_per_s": aggregate_steps_per_s,
        "steady_steps": steady_steps,
        "steady_wall_s": steady_wall,
        "steady_aggregate_steps_per_s": steady_fps,
        "probe_steps": args.probe_steps,
        "gpu_utilization_pct": telemetry_summary.get("gpu_utilization_pct"),
    }
    return record, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--_worker", metavar="SPEC_JSON", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--policy-file", metavar="PATH:FACTORY")
    ap.add_argument("--checkpoint")
    ap.add_argument("--algorithm", default="unknown")
    ap.add_argument("--training-commit", default="0" * 40)
    ap.add_argument("--training-wall-s", type=float, default=0.0)
    ap.add_argument("--setup-wall-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-name", default="declared-only")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", choices=("auto", "batched", "workers"), default="auto",
                    help="execution backend: 'batched' (Option B, single-process multi-env), "
                         "'workers' (Option A, multi-process num_envs=1), or 'auto' "
                         "(batched unless an explicit integer --workers N is passed)")
    ap.add_argument("--batch-envs", default="auto",
                    help="parallel envs for Option B batched simulation, or 'auto' (default: up to 40)")
    ap.add_argument("--policy-device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="where batched policy inference runs (auto/cuda=GPU zero-copy for native "
                         "policies, cpu=vectorized NumPy)")
    ap.add_argument("--workers", default="auto",
                    help="worker process count for Option A, or 'auto': "
                         "min(VRAM budget / per-worker footprint, CPU cores, n_clips)")
    ap.add_argument("--probe-steps", type=int, default=None,
                    help="truncate each clip to N steps (throughput probes only)")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="subset for throughput probes / parity runs")
    ap.add_argument("--cache-references", action="store_true")
    ap.add_argument("--parts-dir", default=None,
                    help="per-clip persistence root (use a Drive path for crash safety)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args._worker:
        return run_worker(args._worker)

    if not args.out:
        ap.error("--out is required")
    if not args.policy_file:
        ap.error("--policy-file is required")
    if not args.checkpoint:
        args.checkpoint = "none"

    full = list(S.load_protocol_v2()["clip_ids"])
    clips = list(args.clips) if args.clips else full
    unknown = set(clips) - set(full)
    if unknown:
        ap.error(f"unknown clip ids: {sorted(unknown)}")
    if len(set(clips)) != len(clips):
        ap.error("duplicate clip ids")

    workers_raw = str(args.workers).strip()
    workers_is_auto = workers_raw.lower() in {"auto", "0"}
    backend = args.backend
    if backend == "auto":
        backend = "batched" if workers_is_auto else "workers"

    recovery = {
        "algorithm": args.algorithm,
        "checkpoint_sha256": policy_mod_sha(args.checkpoint),
        "protocol_sha256": S.protocol_sha256(),
        "device": args.device,
        "evaluator_git_sha": S.evaluator_identity().get("evaluator_git_sha") or "unknown",
    }

    if backend == "batched":
        record, summary = run_batched(args, clips, recovery)
    else:
        # resolve worker count (auto = min(VRAM budget, CPU cores, n_clips))
        worker_resolution: dict | None = None
        if workers_is_auto:
            worker_resolution = resolve_max_workers(args.device, n_clips=len(clips))
            args.workers = worker_resolution["workers"]
            if not args.quiet:
                print(f"auto workers    : {args.workers} "
                      f"(binding={worker_resolution['binding_constraint']}, "
                      f"vram_free={worker_resolution['vram_free_mib']:.0f}MiB, "
                      f"w_vram<={worker_resolution['w_vram_limit']}, "
                      f"w_cpu<={worker_resolution['w_cpu_limit']})", flush=True)
        else:
            try:
                args.workers = int(workers_raw)
            except ValueError:
                ap.error("--workers must be an integer or 'auto'")
        if args.workers < 1:
            ap.error("--workers must be >= 1")
        args.worker_resolution = worker_resolution
        record, summary = run_parallel(args, clips, recovery)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    audit = S.audit_v2_record(record) if len(clips) == 40 else {"status": "PARTIAL_NO_AUDIT", "errors": []}
    ok = (len(clips) != 40) or audit["status"].startswith("FORMAT_VALID")
    if not args.quiet:
        if summary.get("backend") == "batched":
            print(f"\nfast_eval       : backend=batched batch_envs={summary['batch_envs']} "
                  f"policy={summary.get('policy_backend')} clips={summary['clips']}")
            print(f"control steps   : {summary['executed_control_steps']} executed "
                  f"({summary['total_control_steps']} clip-horizon)")
        else:
            print(f"\nfast_eval       : backend=workers workers={summary['workers']} clips={summary['clips']}")
            print(f"control steps   : {summary['total_control_steps']}")
        print(f"eval wall       : {summary['eval_wall_s']:.1f}s (end-to-end)")
        print(f"aggregate steps/s: {summary['aggregate_steps_per_s']:.1f} (end-to-end)")
        print(f"steady steps/s  : {summary['steady_aggregate_steps_per_s']:.1f} "
              f"({summary['steady_steps']} steps / {summary['steady_wall_s']:.1f}s)")
        print(f"gpu util        : {summary['gpu_utilization_pct']}")
        print(f"aggregate return: {record['evaluation']['aggregate_normalized_return']:+.5f}")
        print(f"failed clips    : {record['evaluation']['failed_clips']}")
        print(f"wrote           : {out}")
        print(f"audit           : {audit['status']}")
        for error in audit.get("errors", []):
            print(f"  - {error}")
    return 0 if ok else 2


def policy_mod_sha(path: str) -> str:
    if not path or path.lower() in {"none", "null", "-"}:
        return "none"
    from benchmark.evaluator import policy as policy_mod  # noqa: PLC0415
    return policy_mod.sha256_file(Path(path))


if __name__ == "__main__":
    sys.exit(main())
