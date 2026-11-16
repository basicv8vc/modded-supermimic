#!/usr/bin/env python
"""Dump deterministic rollout traces for the v1-vs-mjlab alignment report.

Engine-agnostic: it only uses the evaluator's public surface
(``motion.get_reference`` / ``env.TrackingEnv`` / ``step_input`` / ``step``),
so the same script runs against the pre-refactor (mujoco-C) evaluator and the
post-refactor (official mjlab) evaluator. Run it once per engine, then compare
the dumps with ``scripts/alignment/compare_traces.py``.

    python scripts/alignment/dump_traces.py --out reports/alignment/traces_v1
    python scripts/alignment/dump_traces.py --out reports/alignment/traces_mjlab \\
        --actions-from reports/alignment/traces_v1

Action sequences are pure functions of the step index and the reference clip
(never of engine state), so both engines see identical inputs::

    zero        a[t] = 0
    feedforward a[t] = (ref.joint_pos[t] - default_joint_pos) / action_scale

``--cache reuse`` loads ``data/lafan1_g1_npz/<clip>.npz`` when present (and
writes it otherwise); ``--cache regen`` always reconverts from the CSV through
the evaluator's own converter.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import motion, spec  # noqa: E402

CLIPS = ("fallAndGetUp3_subject1", "walk4_subject1")
SEQUENCES = ("zero", "feedforward", "pd_track")


def load_reference(csv_path: Path, use_cache: bool):
    """Pre-refactor ``get_reference(csv, info, ...)`` vs post-refactor one."""
    if "info" in inspect.signature(motion.get_reference).parameters:
        return motion.get_reference(csv_path, spec.build_model(), use_cache=use_cache)
    return motion.get_reference(csv_path, use_cache=use_cache)


def build_env(ref, info):
    from benchmark.evaluator.env import TrackingEnv

    if "info" in inspect.signature(TrackingEnv.__init__).parameters:
        return TrackingEnv(info, ref)
    return TrackingEnv(ref)


def policy_action(name: str, si, ref, info) -> np.ndarray:
    """Deterministic action for one step.

    ``zero`` and ``feedforward`` are pure functions of the step and the
    reference clip; ``pd_track`` closes the loop through ``step_input`` and is
    therefore generated on the first engine and *replayed* on the second.
    """
    k = int(si.reference_index)
    k = min(k, ref.num_frames - 1)
    if name == "zero":
        return np.zeros(C.ACTION_DIM)
    if name == "feedforward":
        return (ref.joint_pos[k] - info.default_joint_pos) / info.action_scale
    if name == "pd_track":
        q = np.asarray(si.joint_pos, dtype=np.float64)
        target = q + 0.8 * (ref.joint_pos[k] - q)
        return (target - info.default_joint_pos) / info.action_scale
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--actions-from", default=None)
    ap.add_argument("--cache", choices=("reuse", "regen"), default="reuse")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-state", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_cache = args.cache == "reuse"
    info = spec.build_model()

    meta = {
        "clips": list(CLIPS),
        "sequences": list(SEQUENCES),
        "cache": args.cache,
        "max_steps": args.max_steps,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for clip in CLIPS:
        csv_path = C.DATA_DIR / f"{clip}.csv"
        ref = load_reference(csv_path, use_cache=use_cache)
        for seq in SEQUENCES:
            if args.actions_from:
                acts = np.load(Path(args.actions_from) / f"{clip}_{seq}_actions.npy")
            else:
                acts = None

            env = build_env(ref, info)

            n_steps = ref.num_frames - 1
            if args.max_steps is not None:
                n_steps = min(n_steps, args.max_steps)
            if acts is not None:
                n_steps = min(n_steps, len(acts))

            path = out / f"{clip}_{seq}.jsonl"
            with path.open("w") as fh:
                total = 0.0
                alive = True
                steps_run = 0
                generated = []
                for step in range(n_steps):
                    si = env.step_input()
                    if acts is not None:
                        action = np.asarray(acts[step], dtype=np.float64)
                    else:
                        action = policy_action(seq, si, ref, info)
                        generated.append(action)
                    result = env.step(action)
                    steps_run = step + 1
                    record = {
                        "step": step,
                        "ref_index": int(si.reference_index),
                        "reward": float(result.reward),
                        "terms": {k: float(v) for k, v in result.terms.items()},
                        "terminated": bool(result.terminated),
                        "terminated_terms": list(result.terminated_terms),
                        "input": {
                            k: np.asarray(getattr(si, k), dtype=np.float64).tolist()
                            for k in (
                                "joint_pos",
                                "joint_vel",
                                "base_lin_vel",
                                "base_ang_vel",
                                "anchor_pos_b",
                                "anchor_ori_b",
                            )
                        },
                    }
                    if not args.no_state:
                        record["state"] = {
                            k: np.asarray(v, dtype=np.float64).tolist()
                            for k, v in env.state_snapshot().items()
                        }
                    fh.write(json.dumps(record) + "\n")
                    total += result.reward
                    if result.terminated:
                        alive = False
                        break
                if acts is None:
                    acts_run = np.array(generated)
                    np.save(out / f"{clip}_{seq}_actions.npy", acts_run)
                else:
                    acts_run = np.asarray(acts[:steps_run])
                summary = {
                    "clip": clip,
                    "sequence": seq,
                    "num_frames": int(ref.num_frames),
                    "steps_run": steps_run,
                    "total_reward": float(total),
                    "failed": not alive,
                    "actions_sha256": hashlib.sha256(
                        np.ascontiguousarray(acts_run).tobytes()
                    ).hexdigest(),
                }
                fh.write(json.dumps({"summary": summary}) + "\n")
            print(f"[dump] {path} ({steps_run} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
