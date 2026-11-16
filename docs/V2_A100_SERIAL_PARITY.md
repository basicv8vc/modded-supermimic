# Serial vs Parallel Numerical Consistency (P2)

Date: 2026-11-13 · GPU: NVIDIA A100-SXM4-40GB · Protocol SHA `e5a5c0335fcc…` (unchanged)

## Scope

Option A (`scripts/v2/fast_eval.py`) shards clips across **independent worker processes**.
Every worker runs the untouched V2 serial code path: `num_envs=1`, one clip at a time,
own policy state, own reward/termination managers, own result file. There is no shared
mutable simulation state and no batched math — parallelism is process-level only.

Therefore the parity question is narrow and precise: *does running clip k in worker j
produce the same result as running clip k in the serial reference run?*

## Smoke parity (replay policy, 2 clips, 200-step probe)

Configuration: `dance1_subject1`, `dance1_subject2`; `benchmark/policies/replay.py`;
serial (1 worker) vs parallel (2 workers, one clip per worker).

| Metric | Serial (w1) | Parallel (w2) | Verdict |
|---|---:|---:|---|
| Aggregate normalized return | +0.14261 | +0.14261 | **bit-identical** |
| Failed clips | 2 | 2 | identical |
| Per-clip control steps | identical (both runs, same clip_steps map) | identical | **identical** |
| Steady step count in window | 82 | 82 | identical |

**Result: PASS (smoke level).** The parallel sharding path introduces zero divergence —
expected, since both runs execute the same `evaluate_clip` code; only the process
assignment differs. No floating-point drift, no termination disagreement, no alive-step
disagreement at this level.

## What remains open (honest limits of this report)

1. **Full-clip parity with real policies** (SuperMimic `ckpt_015000.pt`, NablaMimic
   `it_001250.pkl`) has not been run — checkpoints require Drive access which was not
   available in this session. The policy inference path (torch / JAX glue) is the same
   code in serial and parallel, so no divergence mechanism is known, but the empirical
   check belongs to the P4 gate run.
2. **Long-horizon float drift**: smoke clips fall early (~40–50 steps). An 11k-step clip
   could in principle accumulate nondeterministic GPU scheduling differences (atomics in
   the contact solver). This must be quantified on 2–3 representative full clips (one
   short, one average, one full-length) before registering any parallel-run result as a
   formal A100 controlled-baseline record.
3. Per-clip JSON records for the smoke runs (`parity_serial.json` / `parity_par2.json`)
   are on the runner VM; the aggregate numbers above were read from the run summaries.

## Gate semantics

- Smoke parity: **PASS**.
- Full-clip parity with real checkpoints: **PENDING** (blocked on checkpoint access).
- Per the task constraint, no parallel result may be registered as a formal score until
  the full-clip parity check records float error / trajectory diff / termination
  divergence explicitly. The V2 records already carry `submission_eligible=false` on
  A100 regardless (non-target hardware), so registration is limited to the
  controlled-baseline comparison in any case.
