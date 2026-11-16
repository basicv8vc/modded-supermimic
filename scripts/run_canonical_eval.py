#!/usr/bin/env python
"""Run the canonical G1 LAFAN1 evaluation.

The evaluator fixes the task, the physics and the score. It does not fix what a
policy observes, how it is trained or what its checkpoint looks like: an
algorithm plugs in through ``--policy-file`` and implements two methods::

    def reset(self, context): ...     # once per clip
    def act(self, step): return ...   # (29,) joint targets, 50 Hz

See ``benchmark/evaluator/policy_api.py`` for the exact contract.

Formal run with an algorithm implementation (recommended):

    python scripts/run_canonical_eval.py \
        --policy-file ../my_algo/infer.py:make_policy \
        --checkpoint ../my_algo/ckpt_30000.pt \
        --algorithm SuperMimic \
        --training-commit <40-hex SHA of the training code> \
        --seed 0 \
        --gpu-name "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
        --training-wall-s 5220 \
        --out reports/canonical_eval.json

The evaluator hashes ``--checkpoint`` for the record and hands the path to your
factory; it never opens or parses it. Without ``--policy-file`` it falls back to
the bundled mjlab-native loader (``.npz`` export or rsl_rl ``.pt``), which is
just one algorithm implementation:

    python scripts/run_canonical_eval.py --checkpoint checkpoints/actor.npz ...

Smoke test (no checkpoint, scripted policy, clearly marked non-submittable):

    python scripts/run_canonical_eval.py --smoke-policy replay --out reports/smoke.json

Exit code is 0 when the record was produced, 2 when the produced record would be
rejected by ``benchmark/core.py``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from benchmark import core  # noqa: E402
from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import manifest, policy as policy_mod, runner, spec  # noqa: E402
from benchmark.evaluator.protocol import protocol_clip_ids  # noqa: E402


class SmokePolicy:
    """Deterministic scripted policies used only to prove the physics runs.

    Not a submission: ``mode="smoke"`` records are labelled and the validator
    rejects a real checkpoint SHA for them elsewhere. It also doubles as the
    smallest possible example of the policy interface.
    """

    kind = "smoke_scripted"

    def __init__(self, mode: str):
        self._mode = mode
        self._context = None

    def reset(self, context) -> None:
        self._context = context

    def act(self, step):
        if self._mode == "hold":
            return np.zeros(C.ACTION_DIM)
        # "replay": PD the reference joint angles, ignore the rest of the state.
        context = self._context
        target = context.reference.joint_pos[step.reference_index]
        return (target - context.default_joint_pos) / context.action_scale


def load_policy_factory(spec_str: str):
    """Resolve ``path/to/file.py:make_policy`` or ``package.module:make_policy``."""
    target, _, factory_name = spec_str.partition(":")
    factory_name = factory_name or "make_policy"
    path = Path(target)

    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise SystemExit(f"policy module not found: {path}")
        module_name = f"_policy_plugin_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise SystemExit(f"cannot import policy module: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise SystemExit(
            f"{spec_str}: no callable {factory_name!r} in the module "
            f"(export a factory that returns an object with reset()/act())"
        )
    return factory


def build_policy(args, info):
    if args.smoke_policy:
        return SmokePolicy(args.smoke_policy), None, "smoke"

    if args.policy_file:
        factory = load_policy_factory(args.policy_file)
        if not args.checkpoint:
            print(
                "note: no --checkpoint given, so this record cannot be submitted "
                "(the protocol requires a checkpoint hash)",
                file=sys.stderr,
            )
            return factory(), None, "smoke"
        path = Path(args.checkpoint)
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        # The algorithm loads its own weights; the evaluator only hashes the file.
        return factory(path), policy_mod.sha256_file(path), "checkpoint"

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    pol = policy_mod.load_policy(path, info)
    return pol, policy_mod.sha256_file(path), "checkpoint"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--policy-file",
        default=None,
        metavar="PATH:FACTORY",
        help=(
            "algorithm entry point, e.g. 'my_algo/infer.py:make_policy'; the factory "
            "receives the checkpoint path when --checkpoint is given, else no argument"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "weights file: hashed for the record and passed to your factory. Without "
            "--policy-file it is loaded by the bundled mjlab-native loader"
        ),
    )
    parser.add_argument("--smoke-policy", choices=("hold", "replay"), default=None)
    parser.add_argument("--clips", nargs="*", default=None, help="subset of clip ids (default: all 40)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N clips")
    parser.add_argument("--out", default="reports/canonical_eval.json")
    parser.add_argument("--algorithm", default="unknown")
    parser.add_argument("--training-commit", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-name", default="cpu-only")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--training-wall-s", type=float, default=0.0)
    parser.add_argument("--setup-wall-s", type=float, default=0.0)
    parser.add_argument(
        "--cache-references",
        action="store_true",
        help="cache the 50 Hz reference arrays under data/lafan1_g1_npz (~1 GB for 40 clips)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.smoke_policy and not args.policy_file:
        parser.error(
            "provide --policy-file (your algorithm) or --checkpoint (bundled mjlab "
            "loader) or --smoke-policy (scripted smoke test)"
        )

    setup_started = time.perf_counter()
    info = spec.build_model()
    frozen = manifest.load_manifest()
    pol, checkpoint_sha, mode = build_policy(args, info)

    clips = args.clips or protocol_clip_ids()
    if args.limit:
        clips = clips[: args.limit]
    setup_s = args.setup_wall_s + (time.perf_counter() - setup_started)

    def progress(result):
        if args.quiet:
            return
        status = "FAILED" if result.failed else "ok"
        print(
            f"[{result.clip:24s}] ret={result.normalized_return:.4f} {status:7s} "
            f"alive={result.alive_steps}/{result.steps} "
            f"ref={result.reference_wall_s:4.1f}s sim={result.rollout_wall_s:4.1f}s",
            flush=True,
        )

    results = runner.evaluate_all(
        info, pol, clips, progress=progress, use_cache=args.cache_references
    )

    if mode == "smoke":
        checkpoint_sha = checkpoint_sha or ("0" * 64)
        algorithm = args.algorithm if args.algorithm != "unknown" else "smoke-scripted"
        training_commit = args.training_commit or "0" * 40
    else:
        algorithm = args.algorithm
        training_commit = args.training_commit or "0" * 40

    record = runner.build_record(
        algorithm=algorithm,
        training_commit=training_commit,
        seed=args.seed,
        gpu_name=args.gpu_name,
        gpu_count=args.gpu_count,
        checkpoint_sha256=checkpoint_sha,
        training_wall_s=args.training_wall_s,
        setup_wall_s=setup_s,
        results=results,
        mode=mode,
        extra={
            "policy_kind": getattr(pol, "kind", "unknown"),
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

    audit = core.audit_record(record)
    gate_errors = runner.submission_errors(record)
    mean_return = float(np.mean([r.normalized_return for r in results]))
    failed = int(sum(r.failed for r in results))
    if not args.quiet:
        print()
        print(f"clips            : {len(results)}")
        print(f"mean normalized  : {mean_return:.4f}  (gate {core.load_protocol()['target_normalized_return']})")
        print(f"failed clips     : {failed}")
        print(f"checkpoint       : {checkpoint_sha}")
        print(f"mode             : {mode}")
        print(f"wrote            : {out}")
        print(f"validator status : {audit['status']}")
        for error in audit["errors"]:
            print(f"  - {error}")
        for error in gate_errors:
            print(f"  - [submission gate] {error}")

    if mode == "smoke":
        return 0
    return 0 if audit["status"] != "INVALID" and not gate_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
