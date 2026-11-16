#!/usr/bin/env python3
"""P0: attribute the per-control-step wall clock to phases (real CUDA path).

Runs the same per-step call sequence as ``runner.evaluate_clip`` with
``profile=True`` instrumentation, for a bounded number of steps:

    step_input()        GPU->CPU observation extraction (multiple syncs)
    policy.act()        policy inference
    env.step()          mjlab step (CUDA graph replay + sync when profile=True)
    reward readback     RewardManager iterable terms -> CPU
    termination read    per-term get_term() -> CPU
    finite check        qpos/qvel -> CPU

Usage:
    python scripts/v2/profile_breakdown.py --smoke-policy replay \\
        --clips walk1_subject1 --max-steps 300 --out reports/v2/step_breakdown.json
    python scripts/v2/profile_breakdown.py --policy-file benchmark/policies/supermimic.py:make_policy \\
        --checkpoint /content/ckpt/ckpt_015000.pt --clips walk1_subject1 --max-steps 300 ...
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


class ReplayPolicy:
    kind = "v2_smoke_scripted"

    def __init__(self):
        self._context = None

    def reset(self, context) -> None:
        self._context = context

    def act(self, step):
        target = self._context.reference.joint_pos[step.reference_index]
        return (target - self._context.default_joint_pos) / self._context.action_scale


def _load_factory(spec_str: str):
    target, _, name = spec_str.partition(":")
    name = name or "make_policy"
    path = Path(target)
    if path.suffix == ".py" or path.exists():
        mspec = importlib.util.spec_from_file_location(f"_v2_policy_{path.stem}", path)
        module = importlib.util.module_from_spec(mspec)
        sys.modules[mspec.name] = module
        mspec.loader.exec_module(module)
    else:
        module = __import__(target, fromlist=[name])
    return getattr(module, name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-policy", choices=("hold", "replay"))
    ap.add_argument("--policy-file")
    ap.add_argument("--checkpoint")
    ap.add_argument("--clips", nargs="*", default=["walk1_subject1"])
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-references", action="store_true", default=True)
    ap.add_argument("--out", default="reports/v2/step_breakdown.json")
    args = ap.parse_args(argv)

    from benchmark.evaluator import constants as C  # noqa: PLC0415
    from benchmark.evaluator import policy_api  # noqa: PLC0415
    from benchmark.evaluator import spec as v1spec  # noqa: PLC0415
    from benchmark.evaluator.motion import get_reference  # noqa: PLC0415
    from benchmark.v2 import runner as v2runner  # noqa: PLC0415
    from benchmark.v2 import runtime as RT  # noqa: PLC0415

    info = v1spec.build_model()
    if args.smoke_policy:
        policy = ReplayPolicy() if args.smoke_policy == "replay" else ReplayPolicy()
        policy_kind = f"smoke_{args.smoke_policy}"
    else:
        policy = _load_factory(args.policy_file)(args.checkpoint)
        policy_kind = args.policy_file

    rows = []
    telemetry = RT.GpuTelemetry(interval=0.4).start()
    for clip in args.clips:
        ref = get_reference(C.DATA_DIR / f"{clip}.csv", use_cache=args.cache_references)
        env = v2runner.V2TrackingEnv(ref, device=args.device, profile=True)
        policy_api.require_interface(policy)
        policy.reset(policy_api.ClipContext(
            clip=clip,
            joint_order=C.JOINT_ORDER,
            default_joint_pos=np.array(info.default_joint_pos, dtype=np.float64),
            action_scale=np.array(info.action_scale, dtype=np.float64),
            control_dt=C.STEP_DT,
            reference=env.reference_view,
        ))
        n = min(args.max_steps, ref.num_frames - 1)
        loop_t0 = time.perf_counter()
        ran = 0
        for _ in range(n):
            step_in = env.step_input()
            action = policy_api.check_action(policy.act(step_in), clip=clip, step=0)
            result = env.step(action)
            ran += 1
            if result.terminated:
                break  # same stop semantics as runner.evaluate_clip
        loop_wall = time.perf_counter() - loop_t0
        perf = env.performance()
        phases = perf.get("phase_times", {})
        step_mean = perf["step_wall_s_mean"]
        accounted = (phases.get("step_input_s", 0.0) + phases.get("policy_s", 0.0)
                     + perf["total_step_wall_s"] + phases.get("reward_read_s", 0.0)
                     + phases.get("term_read_s", 0.0) + phases.get("finite_s", 0.0))
        rows.append({
            "clip": clip,
            "steps": ran,
            "policy_kind": policy_kind,
            "loop_wall_s": loop_wall,
            "loop_steps_per_s": ran / loop_wall if loop_wall > 0 else 0.0,
            "env_build_s": perf["cuda_env_build_s"],
            "reset_s": perf["reset_wall_s"],
            "first_step_wall_s": perf["first_step_wall_s"],
            "step_wall_s_mean": step_mean,
            "step_wall_s_max": perf["step_wall_s_max"],
            "steady_simulation_fps": perf["steady_simulation_fps"],
            "phases_total_s": phases,
            "phases_per_step_ms": {k: 1000.0 * v / max(ran, 1) for k, v in phases.items()},
            "phases_per_step_pct": {
                k: 100.0 * v / loop_wall if loop_wall > 0 else 0.0 for k, v in phases.items()
            },
            "env_step_per_step_ms": 1000.0 * perf["total_step_wall_s"] / max(ran, 1),
            "unattributed_s": loop_wall - accounted,
        })
        print(f"[{clip}] steps={ran} loop={loop_wall:.2f}s ({ran / loop_wall:.1f} steps/s)")
        for k, v in rows[-1]["phases_per_step_ms"].items():
            print(f"    {k:16s} {v:8.2f} ms/step  ({rows[-1]['phases_per_step_pct'][k]:5.1f}%)")
        print(f"    {'env_step':16s} {rows[-1]['env_step_per_step_ms']:8.2f} ms/step")
        print(f"    {'unattributed':16s} {1000.0 * rows[-1]['unattributed_s'] / max(ran, 1):8.2f} ms/step")
    telemetry_summary = telemetry.stop()

    report = {
        "device": args.device,
        "max_steps": args.max_steps,
        "rows": rows,
        "telemetry": telemetry_summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
