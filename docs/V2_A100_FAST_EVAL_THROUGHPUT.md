# A100 Fast Evaluation — Throughput Benchmark (P0–P3)

Date: 2026-11-13 · GPU: NVIDIA A100-SXM4-40GB (sm_80, 40GB, driver 580.82.07)
Host: Colab runtime, 12 vCPU · Repo: `basicv8vc/gptfiles` @ `feat/benchmark-v2-protocol`
Mujoco 3.11.0 · mujoco-warp 3.11.0 · warp 1.16.0 · mjlab 1.6.0@`0fb8a681`
Measurement policy: `benchmark/policies/replay.py` (scripted reference replay, no checkpoint).

## P0 — Per-step breakdown (single worker, 300 steps, walk1_subject1, profile=True)

| Phase | ms/step | Share |
|---|---:|---:|
| `env.step` (mujoco-warp, incl. device sync) | 21.2 | 73% |
| `step_input` observation extraction (GPU→CPU) | 7.2 | 25% |
| reward / termination / finiteness readback | 0.44 | 1.5% |
| policy inference (real policy, measured separately) | ~0.9 | — |
| Python loop overhead | 0.06 | — |
| **Full loop** | **~28.8** | 100% |

Single-worker full-loop throughput: **34.6 control steps/s**.

Key facts established in P0:

1. **Physics is CUDA-graph captured.** `sim.step_graph / forward_graph / reset_graph` are
   all live Warp graphs (`use_cuda_graph=True`; mempool enabled; driver ≥ 12.4). The
   `CUDA graph step captured: n/a` line in the hardware report is a field-population bug,
   not a missing optimization.
2. The cost is **not kernel launch of physics** — it is the per-step synchronous pipeline:
   every control step forces a GPU→CPU→GPU round trip (obs extraction, policy, action
   writeback, reward readback). GPU utilization at batch=1 is 15–39%.
3. Per-clip env rebuild is cheap (0.3–2.2 s warm). Worker process startup (torch/mjlab/warp
   imports, warp kernel cache) is the dominant fixed cost: 150–200 s per `fast_eval`
   invocation at 12-way concurrency.

## P3 — Option A: independent parallel workers

`scripts/v2/fast_eval.py` shards the 40 clips across N worker processes. Each worker keeps
the full V2 serial code path (`num_envs=1`, own clip, own policy state, own result file,
per-clip atomic persistence). No shared mutable simulation state.

| Configuration | Aggregate steady steps/s (env.step window) | Speedup | GPU peak | End-to-end wall |
|---|---:|---:|---:|---:|
| Serial 1 worker | 55.0 | 1.0× | 39% | 159 s |
| 2 workers | 101.4 | 1.8× | 42% | 159 s |
| 4 workers | 190.2 | 3.5× | 62% | 140 s |
| 8 workers | 253.0 | 4.6× | 64% | 189 s |
| 12 workers | **382.4** | **6.9×** | 72% | 244 s |
| 16 workers (12 clips) | 374.8 | 6.8× | 76% | 246 s |

Measurement caveats (stated for honesty, numbers above are not oversold):

- The probe policy falls at ~40–52 steps, so probe windows are short (0.8–2.1 s steady
  window). These runs measure the stepping loop, not full-clip durability.
- The "steady" metric covers the `env.step` window of the loop. Scaling the P0 phase
  proportions to the full loop gives an estimated aggregate of **~280 full-loop steps/s
  at 12 workers** (73% of the env.step-window number).
- End-to-end walls in the table include 2–4 min of per-process startup on 200–2400 probe
  steps; over a full 40-clip run (~441k steps) startup is ~10% overhead.
- Saturation: 12 workers is the sweet spot on 12 vCPU; 16 workers adds nothing. GPU peaks
  at 72–76% — the remaining limit is CPU-side Python glue per step, not the GPU.

### Gate assessment (P3)

| Target | Required | Measured (best) | Status |
|---|---|---|---|
| Aggregate control steps/s | ≥600 (ideal ≥1000) | 382 env.step-window / ~280 full-loop est. | **NOT MET** |
| Per-model 40-clip wall | 5–15 min | ~26 min at 12 workers (441k steps) | NOT MET |
| Scoring / physics unchanged | — | protocol SHA `e5a5c033…` verified unchanged | MET |

**The ≥600 gate is not met.** Best measured is 6.4× the A100 serial baseline (~60
steps/s) and ~2.1× short of the target when measured on the full loop.

## Bottleneck analysis and the next most valuable optimization

Root cause: the V2 evaluator's per-step synchronous loop cannot overlap GPU work across
steps within a worker (obs sync → policy → action → step → reward sync), and Option A
scales only to the number of CPU cores because each worker is a full Python pipeline.

**Most valuable next step: Option B — vectorized batched simulation** (`num_envs=16/32/40`,
one process). mujoco-warp is designed for exactly this (training runs 4096 envs); one
kernel batch + one sync replaces N serial pipelines. Realistic projection 2,000–8,000
env-steps/s → per-model evaluation in 1–4 minutes. Required correctness work (unchanged
scoring math):

- per-env reference motion / frame index (needs per-env motion assignment in the mjlab
  MotionCommand; unverified whether the manager exposes per-env `motion_ids`),
- per-env policy observation/history and action,
- per-env termination masking with different clip lengths,
- per-env reward accumulation and independent per-clip atomic results + resume.

Cheap secondary optimization regardless of Option B: batch the `step_input` GPU→CPU
transfers into a single sync per step (est. 15–25% full-loop gain).

Option A remains the validated, correctness-preserving fallback: it is numerically
identical to serial execution (see parity report) and delivers ~26 min/model on A100.

## Option B implementation (`benchmark/v2/batched.py` + `scripts/v2/fast_eval.py`)

Option B has now been implemented on top of the untouched V2 protocol and config:

- **Packed multi-clip `MotionCommand`** (`PackedMotionBank` + `install_packed_motion_command`):
  loads pending clips into a single contiguous GPU `MotionLoader` bank and tracks
  per-env `[clip_start, clip_end)` windows across `num_envs=B` (default `--batch-envs auto`,
  up to 40 concurrent clips), with partial `env.reset(env_ids=...)` on slot completion.
- **LPT work-stealing + crash-safe resume**: sorts pending clips longest-first when
  `batch_envs < len(clips)`, persists per-clip atomic JSONs to `--parts-dir`, and
  returns results in exact protocol clip order.
- **Zero-copy GPU policy adapters** (`--policy-device auto|cuda|cpu`):
  `BatchedSuperMimicPolicy`, `BatchedNablaMimicPolicy`, and `BatchedReplayPolicy`
  run observation reconstruction and float32 MLP inference (with `allow_tf32=False`)
  directly on GPU tensors, reducing host readback to a single packed reward/termination
  transfer per step; arbitrary custom `reset/act` policies remain supported via
  `SlotPolicyAdapter`.
- **Parallel cold-cache reference conversion** (`_load_references`): when `--cache-references`
  is set and multiple `.npz` caches are absent, spawns up to 12 CPU workers to convert
  LAFAN1 CSVs via `OfficialConverter` in parallel (~2 min cold vs ~20 min serial).
- **CLI** (`scripts/v2/fast_eval.py`):
  - `--backend auto` (default): selects Option B (`batched`) unless an explicit integer
    `--workers N` is given, in which case it runs Option A (`workers`).
  - `--backend batched --batch-envs 40` (or `16`/`32`/`auto`).

### Measured G4 (`NVIDIA RTX PRO 6000 Blackwell`, `sm_120`) Option B results

Measured on Colab G4 (48 vCPU, 95 GiB VRAM, `mujoco==3.11.0`, `mujoco-warp==3.11.0`,
`warp-lang==1.16.0`, `mjlab==1.6.0@0fb8a681`) using a full-width `SuperMimicPolicy`
checkpoint (`1024→512→256→128→32` / `1022→1024→512→256→128→29`):

- **Serial vs Option B (`b=4`, GPU policy) parity** (4 clips: `walk1_subject1`, `dance1_subject1`, `jumps1_subject1`, `run1_subject2`):
  - 25-step pre-fall window: `verdict=PASS`, `0/4` mismatched, `max|Δret| = 8.64e-8`, `|Δagg| = 1.27e-8`.
  - 300-step fall window: `verdict=PASS`, `0/4` mismatched, `max|Δret| = 1.43e-4`, `|Δagg| = 3.56e-5` (exact same `alive_steps`: 40, 40, 45, 54).
- **Sustained 250-step `BatchedSuperMimicPolicy` GPU throughput** (no early exit, warm `.npz` cache):

| Configuration | Env build | ms / iter | Throughput (env-steps/s) | Speedup vs `b=1` | Projected 441k steps |
|---|---:|---:|---:|---:|---:|
| `num_envs=1` | 1.13 s | 7.91 ms | 126.4 | 1.0× | 58.13 min |
| `num_envs=16` | 0.36 s | 10.02 ms | 1,597.4 | 12.6× | 4.60 min |
| `num_envs=32` | 0.13 s | 10.54 ms | 3,037.1 | 24.0× | 2.42 min |
| **`num_envs=40`** | **0.13 s** | **10.72 ms** | **3,729.8** | **29.5×** | **1.97 min** |

40-clip `fast_eval.py` end-to-end probe (with 12-clip-length-aware warm `.npz` load ~2.7 s)
achieved **1,550.1 steady steps/s** (`b=32`) and **15.7–16.6 s** total wall for the 40-clip
probe window. Real trained checkpoints (`ckpt_015000.pt`, `it_001250.pkl`) full 40-clip
runs require Google Drive checkpoint mount.

## Artifacts

- `benchmark/v2/batched.py` — Option B vectorized batched evaluator & GPU policy adapters
- `scripts/v2/fast_eval.py` — unified fast evaluator (`--backend batched|workers|auto`)
- `tests/test_v2_batched.py` — Option B unit, policy parity, and scheduling/resume tests
- `scripts/v2/profile_breakdown.py` — P0 phase instrumentation
- `benchmark/policies/replay.py` — smoke policy for throughput probes (never scored)
- `reports/v2/step_breakdown_replay.json` — P0 raw numbers (on runner VM)
- Probe records: `probe_w{1,2,4,8,12,16}.json`, `probe_warm.json`, `parity_{serial,par2}.json`
