"""Per-step diagnostic for one clip: which termination term fires, and how fast.

Diagnostic only, not part of the benchmark. Gate quantities are read from the
official ``MotionCommand`` tensors the terminations compare.

    python scripts/diagnose_clip.py --clip dance1_subject1 --steps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.evaluator import constants as C
from benchmark.evaluator.env import TrackingEnv
from benchmark.evaluator.mathx import quat_apply_inverse
from benchmark.evaluator.motion import get_reference
from benchmark.evaluator.spec import build_model

GRAVITY_VEC_W = np.array([0.0, 0.0, -1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dance1_subject1")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--policy", choices=("hold", "replay", "zero_ref"), default="replay")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "lafan1_g1"))
    args = ap.parse_args()

    info = build_model()
    csv_path = Path(args.data_root) / f"{args.clip}.csv"
    ref = get_reference(csv_path)
    env = TrackingEnv(ref)
    anchor_slice = info.anchor_body_id - 1
    ee_slice = info.ee_body_ids - 1

    print(f"clip={args.clip} frames={ref.num_frames} steps={ref.num_frames - 1}")
    print(f"ref pelvis z t0={ref.body_pos_w[0, 0, 2]:.4f}")
    print(
        f"\n{'k':>5} {'pelvisz':>8} {'dposZ':>7} {'dgravZ':>7} "
        + " ".join(f"{'ee' + str(i):>8}" for i in range(len(C.EE_BODIES)))
        + f" {'reward':>9}  terms"
    )

    for k in range(args.steps):
        if args.policy == "hold":
            action = np.zeros(C.ACTION_DIM)
        elif args.policy == "zero_ref":
            action = (ref.joint_pos[0] - info.default_joint_pos) / info.action_scale
        else:
            action = (
                ref.joint_pos[min(k, ref.num_frames - 1)] - info.default_joint_pos
            ) / info.action_scale
        step = env.step(action)

        # official gate quantities (the terminations compare exactly these)
        cmd = env._command
        names = list(cmd.cfg.body_names)
        tracked_ee = [names.index(n) for n in C.EE_BODIES]
        rel = cmd.body_pos_relative_w[0].detach().cpu().numpy()
        rob = cmd.robot_body_pos_w[0].detach().cpu().numpy()
        eez = rel[tracked_ee, 2] - rob[tracked_ee, 2]

        rk = min(env.ref_index, ref.num_frames - 1)
        robot_anchor_z = float(cmd.robot_anchor_pos_w[0, 2])
        dpos = abs(ref.body_pos_w[rk, anchor_slice][2] - robot_anchor_z)
        mg = quat_apply_inverse(ref.body_quat_w[rk, anchor_slice], GRAVITY_VEC_W)
        rg = quat_apply_inverse(
            cmd.robot_anchor_quat_w[0].detach().cpu().numpy(), GRAVITY_VEC_W
        )
        dgrav = abs(mg[2] - rg[2])
        print(
            f"{env.ref_index:>5} {robot_anchor_z:>8.4f} {dpos:>7.4f} {dgrav:>7.4f} "
            + " ".join(f"{v:>+8.3f}" for v in eez)
            + f" {step.reward:>9.4f}  {','.join(step.terminated_terms) or '-'}"
        )
        if step.terminated:
            print(f"\nterminated at step {env.ref_index}: {step.terminated_terms}")
            break
    else:
        print(f"\nsurvived {args.steps} steps")

    print(
        f"thresholds: anchor_pos={C.TERM_ANCHOR_POS_THRESHOLD} "
        f"anchor_ori={C.TERM_ANCHOR_ORI_THRESHOLD} ee={C.TERM_EE_BODY_POS_THRESHOLD}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
