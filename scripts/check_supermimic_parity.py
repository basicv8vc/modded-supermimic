#!/usr/bin/env python
"""Compare the numpy SuperMimic inference against the original torch code.

This is the fidelity check behind ``benchmark/policies/supermimic.py``. It needs
the upstream sources and torch, i.e. an environment the evaluator itself does
not require:

    git clone https://github.com/zita-ch/supermimic_vibed /tmp/supermimic_vibed
    git -C /tmp/supermimic_vibed checkout 8bfc1158bd968f729fcc65d5e897a05b05ea3d36
    python scripts/check_supermimic_parity.py \
        --upstream /tmp/supermimic_vibed \
        --checkpoint ~/Downloads/.../ckpt_015000.pt \
        --clip fallAndGetUp3_subject1 --steps 300

It reports, in order:

1. the per-frame reference derivatives (``root_v_local``, ``root_w_local``,
   ``qd``) against ``MotionLib``'s own tensors;
2. the 730-dim goal against ``MotionLib.goal`` for many frames, including the
   clamped tail;
3. the 29-dim action against ``Policy.forward`` for the same inputs, in fp32 and
   under the bf16 autocast the original evaluation ran with;
4. a closed-loop rollout of *this* clip driven by each implementation in turn,
   so the comparison is not limited to one-step agreement.

``MotionLib`` needs forward kinematics, which needs IsaacLab; the stub below
returns the pelvis pose unchanged. Only ``goal()``/``ref_state()`` are compared,
and neither reads the FK output, so the stub does not weaken the check -- it
does mean body-pose arrays are not compared here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from benchmark.evaluator import constants as C  # noqa: E402
from benchmark.evaluator import motion, spec  # noqa: E402
from benchmark.evaluator.env import TrackingEnv  # noqa: E402
from benchmark.evaluator import runner  # noqa: E402
from benchmark.policies import supermimic as sm  # noqa: E402


def _require_torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "torch is required for the parity check; install it in a separate "
            "environment (the evaluator itself must stay torch-free)"
        )
    return torch


def build_upstream(upstream: Path, torch, device="cpu"):
    """Import ``supermimic.networks``/``motion`` with IsaacLab stubbed out."""
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    import types  # noqa: PLC0415

    kinematics = types.ModuleType("supermimic.kinematics")

    class G1Kinematics:  # noqa: D401 - stand-in
        """FK is not exercised here; only goal()/ref_state() are compared."""

        def forward(self, p, R, v, w, q, qd):
            zero = torch.zeros_like(p)
            return zero, R, zero, zero

    kinematics.G1Kinematics = G1Kinematics
    sys.modules["supermimic.kinematics"] = kinematics

    import supermimic.motion as up_motion  # noqa: PLC0415
    import supermimic.networks as up_networks  # noqa: PLC0415

    return up_motion, up_networks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="checkout of supermimic_vibed")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--clip", default="fallAndGetUp3_subject1")
    ap.add_argument("--steps", type=int, default=300, help="closed-loop steps to compare")
    aw = ap.add_argument
    aw("--data-dir", type=Path, default=C.DATA_DIR)
    args = ap.parse_args(argv)

    torch = _require_torch()
    up_motion, up_networks = build_upstream(args.upstream, torch)

    info = spec.build_model()
    ref = motion.get_reference(args.data_dir / f"{args.clip}.csv")

    # ---------------------------------------------------------------- reference
    filenames = sorted(str(p) for p in args.data_dir.glob("*.csv"))
    lib = up_motion.MotionLib(filenames, up_motion.G1Kinematics(), "cpu")
    idx = lib.names.index(args.clip)
    start, end = int(lib.clip_start[idx]), int(lib.clip_end[idx])
    frames = slice(start, end + 1)

    ours = sm.reference_derivatives(ref.joint_pos, ref.body_pos_w[:, 0], ref.body_quat_w[:, 0])
    print(f"clip           : {args.clip}  canonical T={ref.num_frames}  upstream T={end - start + 1}")
    print(f"frames         : canonical {ref.num_frames - 1} control steps, upstream {end - start} steps")

    pairs = {
        "p": (lib.root_p[frames], ref.body_pos_w[:, 0]),
        "R": (lib.root_R[frames], np.stack([sm.mathx.matrix_from_quat(q) for q in ref.body_quat_w[:, 0]])),
        "q": (lib.q[frames], ref.joint_pos),
        "qd": (lib.qd[frames], ours["qd"]),
        "root_v_local": (lib.root_v_local[frames], ours["root_v_local"]),
        "root_w_local": (lib.root_w_local[frames], ours["root_w_local"]),
    }
    print("\n[1] reference tensors (upstream MotionLib vs numpy rebuild)")
    for name, (a, b) in pairs.items():
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        b = np.asarray(b)
        n = min(len(a), len(b))
        print(f"  {name:14s} max|delta| = {np.abs(a[:n] - b[:n]).max():.3e}   (compared {n} frames)")

    # -------------------------------------------------------------------- goal
    policy = sm.SuperMimicPolicy.from_checkpoint(args.checkpoint)
    probe = sm.SuperMimicPolicy.__new__(sm.SuperMimicPolicy)
    probe.goal_frames = sm.GOAL_FRAMES
    probe._derived = ours
    probe._derived["yaw"] = np.stack([sm.yaw_matrix(R) for R in ours["R"]])

    K = sm.GOAL_FRAMES
    worst, worst_f = 0.0, None
    probe_frames = sorted({0, 1, 2, 5, 17, 50, ref.num_frames // 2, ref.num_frames - 20,
                           ref.num_frames - 11, ref.num_frames - 10, ref.num_frames - 3,
                           ref.num_frames - 2, ref.num_frames - 1})
    for f in probe_frames:
        if f < 0:
            continue
        up = lib.goal(torch.tensor([start + f]), K)[0].detach().numpy()
        mine = probe.goal(f)
        err = float(np.abs(up - mine).max())
        if err > worst:
            worst, worst_f = err, f
    print(f"\n[2] goal (730-d, {len(probe_frames)} frames incl. the clamped tail): max|delta| = {worst:.3e} at f={worst_f}")

    # ----------------------------------------------------------------- forward
    obs_hist, goals, actions = [], [], []
    env = TrackingEnv(ref)
    import benchmark.evaluator.policy_api as api  # noqa: PLC0415

    policy.reset(
        api.ClipContext(
            clip=args.clip,
            joint_order=C.JOINT_ORDER,
            default_joint_pos=np.array(info.default_joint_pos, dtype=np.float64),
            action_scale=np.array(info.action_scale, dtype=np.float64),
            control_dt=C.STEP_DT,
            reference=env.reference_view,
        )
    )
    for _ in range(min(args.steps, ref.num_frames - 1)):
        step = env.step_input()
        a = policy.act(step)
        obs_hist.append(policy._hist.copy())
        goals.append(policy.goal(step.reference_index - 1))
        actions.append(a)
        env.step(a)

    obs_hist = np.stack(obs_hist).astype(np.float32)
    goals = np.stack(goals).astype(np.float32)

    pi = up_networks.Policy(sm.OBS_DIM, sm.GOAL_PER_FRAME * K, hist=sm.HIST).to("cpu")
    pi.load_state_dict({k[len("pi."):]: torch.from_numpy(v) for k, v in policy.tensors.items() if k.startswith("pi.")})
    pi.eval()

    def upstream_actions(amp):
        up_networks.AMP_DTYPE = amp
        with torch.no_grad():
            out = pi(torch.from_numpy(obs_hist), torch.from_numpy(goals))
        return out.numpy().astype(np.float64)

    mine = np.stack(actions)
    for label, amp in (("fp32", None), ("bf16 autocast", torch.bfloat16)):
        up = upstream_actions(amp)
        delta = np.abs(up - mine)
        print(
            f"\n[3] action, {label:14s} ({len(mine)} steps): max|delta| = {delta.max():.3e}  "
            f"mean|delta| = {delta.mean():.3e}  max|action| = {np.abs(mine).max():.3f}"
        )

    # The original evaluation ran the matmuls under ``torch.autocast("cuda",
    # bfloat16)``. CPU autocast cannot stand in for that (torch disables it
    # without CUDA), so emulate the narrowest possible version: same math, same
    # layer boundaries, bf16 storage for every operand.
    pi_bf16 = up_networks.Policy(sm.OBS_DIM, sm.GOAL_PER_FRAME * K, hist=sm.HIST).to("cpu")
    pi_bf16.load_state_dict(pi.state_dict())
    pi_bf16.to(torch.bfloat16).eval()
    up_networks.AMP_DTYPE = None
    with torch.no_grad():
        out = pi_bf16(
            torch.from_numpy(obs_hist).to(torch.bfloat16),
            torch.from_numpy(goals).to(torch.bfloat16),
        )
    delta = np.abs(out.float().numpy() - mine)
    print(
        f"\n[3] action, bf16 weights   ({len(mine)} steps): max|delta| = {delta.max():.3e}  "
        f"mean|delta| = {delta.mean():.3e}   <- what the GPU run actually computed"
    )

    # --------------------------------------------------------------- closed loop
    print("\n[4] closed loop, same physics, driven by each implementation")

    class TorchPolicy:
        kind = "supermimic-native-pi-torch"

        def __init__(self, module, dtype=torch.float32):
            self.module, self.dtype = module, dtype

        def reset(self, context):
            self._ctx = context
            self._hist = None

        def act(self, step):
            obs = policy.observation(step)
            self._hist = (
                np.repeat(obs[None, :], sm.HIST, axis=0)
                if self._hist is None
                else np.concatenate([self._hist[1:], obs[None, :]], axis=0)
            )
            g = policy.goal(step.reference_index - 1)
            up_networks.AMP_DTYPE = None
            with torch.no_grad():
                out = self.module(
                    torch.from_numpy(self._hist.astype(np.float32))[None].to(self.dtype),
                    torch.from_numpy(g.astype(np.float32))[None].to(self.dtype),
                )
            return out[0].float().numpy().astype(np.float64)

    for label, pol in (
        ("numpy", policy),
        ("torch fp32", TorchPolicy(pi)),
        ("torch bf16", TorchPolicy(pi_bf16, torch.bfloat16)),
    ):
        result = runner.evaluate_clip(info, ref, pol, args.clip)
        print(
            f"  {label:10s} normalized_return={result.normalized_return:.6f} "
            f"failed={result.failed} alive={result.alive_steps}/{result.steps} "
            f"terms={result.fall_terms}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
