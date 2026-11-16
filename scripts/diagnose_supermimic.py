#!/usr/bin/env python
"""Why did each clip end? Diagnostic layer on top of the frozen evaluation.

The score stays exactly what ``scripts/run_canonical_eval.py`` produces. This
script re-runs the same rollouts and records extra evidence that the record
schema has no field for:

* the tracking quality *while the episode was alive* (sum of the positive reward
  terms per step, whose ceiling is ``MAX_STEP_REWARD = 0.1``), so "the policy
  never tracked" and "the policy tracked but tripped a gate" stay distinguishable;
* how much of the alive window was eaten by the ``self_collisions`` penalty,
  which SuperMimic's own environment cannot produce (it disables self-collisions
  and has no such term);
* at the terminating step, which of the four ``ee_body_pos`` bodies was out by
  more than the frozen 0.25 m threshold.

    python scripts/diagnose_supermimic.py --checkpoint <ckpt> --out reports/....json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import motion, policy as policy_mod, policy_api, spec  # noqa: E402
from benchmark.evaluator.env import TrackingEnv  # noqa: E402
from benchmark.evaluator.protocol import protocol_clip_ids  # noqa: E402
from benchmark.policies.supermimic import SuperMimicPolicy  # noqa: E402


def _ee_z_errors(env) -> np.ndarray:
    """The official ``ee_body_pos`` gate quantities (z-only, 0.25 m).

    Read straight from the live ``MotionCommand`` tensors -- the same values
    ``bad_motion_body_pos_z_only`` compares.
    """
    cmd = env._command
    names = list(cmd.cfg.body_names)
    idx = [names.index(n) for n in C.EE_BODIES]
    rel = cmd.body_pos_relative_w[0].detach().cpu().numpy()
    rob = cmd.robot_body_pos_w[0].detach().cpu().numpy()
    return np.abs(rel[idx, 2] - rob[idx, 2])


def run_clip(info, ref, policy, clip: str) -> dict:
    env = TrackingEnv(ref)
    policy.reset(
        policy_api.ClipContext(
            clip=clip,
            joint_order=C.JOINT_ORDER,
            default_joint_pos=np.array(info.default_joint_pos, dtype=np.float64),
            action_scale=np.array(info.action_scale, dtype=np.float64),
            control_dt=C.STEP_DT,
            reference=env.reference_view,
        )
    )
    positive = 0.0
    collision = 0.0
    reward = 0.0
    alive = 0
    worst_ee = np.zeros(len(C.EE_BODIES))
    fail_step = None
    fail_terms: tuple[str, ...] = ()
    fail_ee: list[str] = []

    for step in range(ref.num_frames - 1):
        result = env.step(policy.act(env.step_input()))
        err = _ee_z_errors(env)
        worst_ee = np.maximum(worst_ee, err)
        positive += sum(v for v in result.terms.values() if v > 0.0)
        collision += result.terms.get("self_collisions", 0.0)
        reward += result.reward
        alive += 1
        if result.terminated:
            fail_step = step + 1
            fail_terms = result.terminated_terms
            fail_ee = [n for n, e in zip(C.EE_BODIES, err) if e > C.TERM_EE_BODY_POS_THRESHOLD]
            break

    n_steps = ref.num_frames - 1
    return {
        "clip": clip,
        "steps": n_steps,
        "alive_steps": alive,
        "failed": fail_step is not None,
        "fall_step": fail_step,
        "fall_terms": list(fail_terms),
        "fall_ee_bodies": fail_ee,
        "normalized_return": float(min(max(reward / (n_steps * C.MAX_STEP_REWARD), 0.0), 1.0)),
        # per-step rates over the alive window (0.1 is the ceiling of the positive part)
        "tracking_rate_alive": positive / max(alive, 1),
        "self_collision_rate_alive": collision / max(alive, 1),
        "self_collision_cost_alive": -C.REWARDS["self_collisions"]["weight"]
        * C.STEP_DT
        * collision
        / max(alive, 1),
        "worst_ee_z_error": {n: float(v) for n, v in zip(C.EE_BODIES, worst_ee)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", default="reports/supermimic_diagnostics.json")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args(argv)

    info = spec.build_model()
    policy = SuperMimicPolicy.from_checkpoint(args.checkpoint)
    clips = args.clips or protocol_clip_ids()
    rows = []
    for clip in clips:
        ref = motion.get_reference(C.DATA_DIR / f"{clip}.csv")
        row = run_clip(info, ref, policy, clip)
        rows.append(row)
        print(
            f"[{clip:24s}] ret={row['normalized_return']:.4f} alive={row['alive_steps']:>5}"
            f"/{row['steps']:<5} track/step={row['tracking_rate_alive']:.4f} "
            f"coll/step={row['self_collision_rate_alive']:.2f} "
            f"{'fail:' + ','.join(row['fall_terms']) if row['failed'] else 'ok'}"
            f"{' (' + ','.join(row['fall_ee_bodies']) + ')' if row['fall_ee_bodies'] else ''}",
            flush=True,
        )

    payload = {
        "checkpoint_sha256": policy_mod.sha256_file(args.checkpoint),
        "max_step_reward": C.MAX_STEP_REWARD,
        "termination_thresholds": {
            "anchor_pos_m": C.TERM_ANCHOR_POS_THRESHOLD,
            "anchor_ori": C.TERM_ANCHOR_ORI_THRESHOLD,
            "ee_body_pos_m": C.TERM_EE_BODY_POS_THRESHOLD,
        },
        "clips": rows,
        "summary": {
            "failed": sum(r["failed"] for r in rows),
            "mean_return": float(np.mean([r["normalized_return"] for r in rows])),
            "mean_tracking_rate_alive": float(np.mean([r["tracking_rate_alive"] for r in rows])),
            "failures_by_term": {
                term: sum(term in r["fall_terms"] for r in rows)
                for term in ("anchor_pos", "anchor_ori", "ee_body_pos", "nonfinite")
            },
            "failures_by_ee_body": {
                body: sum(body in r["fall_ee_bodies"] for r in rows) for body in C.EE_BODIES
            },
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
