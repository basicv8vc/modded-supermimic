#!/usr/bin/env python
"""Run the canonical evaluation and measure what the policy itself costs.

Identical rollout to ``scripts/run_canonical_eval.py`` (same ``runner`` calls,
same record schema); the only addition is a timer around ``act`` so the policy's
share of the wall clock is separable from MuJoCo's.

    python scripts/profile_supermimic.py \
        --checkpoint ~/Downloads/.../ckpt_015000.pt \
        --out reports/supermimic_canonical_eval.json \
        --timing-out reports/supermimic_timing.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark import core  # noqa: E402
from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import manifest, policy as policy_mod, runner, spec  # noqa: E402
from benchmark.evaluator.protocol import protocol_clip_ids  # noqa: E402
from benchmark.policies.supermimic import SuperMimicPolicy  # noqa: E402


class TimedPolicy:
    """Delegates to the algorithm and records how long each ``act`` takes."""

    def __init__(self, inner):
        self.inner = inner
        self.kind = inner.kind
        self.steps_s: list[float] = []
        self.reset_s: list[float] = []

    def reset(self, context):
        started = time.perf_counter()
        self.inner.reset(context)
        self.reset_s.append(time.perf_counter() - started)

    def act(self, step):
        started = time.perf_counter_ns()
        action = self.inner.act(step)
        self.steps_s.append((time.perf_counter_ns() - started) * 1e-9)
        return action


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", default="reports/supermimic_canonical_eval.json")
    ap.add_argument("--timing-out", default="reports/supermimic_timing.json")
    ap.add_argument("--algorithm", default="SuperMimic")
    ap.add_argument("--training-commit", default="8bfc1158bd968f729fcc65d5e897a05b05ea3d36")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-name", default="cpu-only")
    ap.add_argument("--training-wall-s", type=float, default=0.0)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    setup_started = time.perf_counter()
    info = spec.build_model()
    frozen = manifest.load_manifest()
    inner = SuperMimicPolicy.from_checkpoint(args.checkpoint)
    policy = TimedPolicy(inner)
    checkpoint_sha = policy_mod.sha256_file(args.checkpoint)
    clips = args.clips or protocol_clip_ids()
    setup_s = time.perf_counter() - setup_started

    def progress(result):
        if args.quiet:
            return
        status = "FAILED" if result.failed else "ok"
        print(
            f"[{result.clip:24s}] ret={result.normalized_return:.4f} {status:7s} "
            f"alive={result.alive_steps}/{result.steps} "
            f"ref={result.reference_wall_s:4.1f}s sim={result.rollout_wall_s:5.1f}s",
            flush=True,
        )

    started = time.perf_counter()
    results = runner.evaluate_all(info, policy, clips, progress=progress, use_cache=False)
    rollout_s = time.perf_counter() - started

    record = runner.build_record(
        algorithm=args.algorithm,
        training_commit=args.training_commit,
        seed=args.seed,
        gpu_name=args.gpu_name,
        gpu_count=1,
        checkpoint_sha256=checkpoint_sha,
        training_wall_s=args.training_wall_s,
        setup_wall_s=setup_s,
        results=results,
        mode="checkpoint",
        extra={
            "policy_kind": getattr(policy, "kind", "unknown"),
            "clip_subset": None if len(clips) == 40 else list(clips),
            "frozen": {
                "mjlab_commit": frozen["mjlab"]["commit"],
                "robot_model_sha256": frozen["robot_model"]["sha256"],
                "dataset_manifest_sha256": frozen["dataset"]["manifest_sha256"],
                "compiled_fingerprint": frozen["compiled_fingerprint"],
            },
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    steps = policy.steps_s
    total_policy_s = float(sum(steps))
    timing = {
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "checkpoint_sha256": checkpoint_sha,
        "actor_parameters": inner.parameter_count(),
        "actor_architecture": inner.architecture(),
        "observation_dim": 99,
        "goal_dim": 730,
        "history": 10,
        "clips": len(results),
        "control_steps": len(steps),
        "policy_total_s": total_policy_s,
        "policy_mean_ms": 1e3 * total_policy_s / max(len(steps), 1),
        "policy_p50_ms": 1e3 * percentile(steps, 50),
        "policy_p95_ms": 1e3 * percentile(steps, 95),
        "policy_p99_ms": 1e3 * percentile(steps, 99),
        "policy_max_ms": 1e3 * max(steps) if steps else 0.0,
        "reset_p50_ms": 1e3 * percentile(policy.reset_s, 50),
        "rollout_wall_s": rollout_s,
        "policy_share_of_rollout": total_policy_s / rollout_s if rollout_s else 0.0,
        "evaluation_summary": {
            "mean_normalized_return": float(np.mean([r.normalized_return for r in results])),
            "failed_clips": int(sum(r.failed for r in results)),
            "validator_status": core.audit_record(record)["status"],
        },
        "per_clip_return": {r.clip: r.normalized_return for r in results},
    }
    timing_out = Path(args.timing_out)
    timing_out.parent.mkdir(parents=True, exist_ok=True)
    timing_out.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"clips                : {len(results)}")
    print(f"control steps        : {len(steps)}")
    print(f"mean normalized      : {timing['evaluation_summary']['mean_normalized_return']:.4f}")
    print(f"failed clips         : {timing['evaluation_summary']['failed_clips']}")
    print(f"validator status     : {timing['evaluation_summary']['validator_status']}")
    print(f"policy total         : {total_policy_s:.1f} s on cpu")
    print(
        f"policy latency       : mean {timing['policy_mean_ms']:.2f} ms  "
        f"p50 {timing['policy_p50_ms']:.2f}  p95 {timing['policy_p95_ms']:.2f}  "
        f"p99 {timing['policy_p99_ms']:.2f}"
    )
    print(f"rollout wall         : {rollout_s:.1f} s (policy {100 * timing['policy_share_of_rollout']:.1f}%)")
    print(f"actor parameters     : {timing['actor_parameters']:,}")
    print(f"wrote                : {out}  +  {timing_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
