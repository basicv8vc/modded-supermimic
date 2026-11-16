# Benchmark V2 — G4 Baseline & CTO Diagnostic Closure Report

- Protocol: `modded-supermimic-g1-lafan1-v2` ([`benchmark/v2/protocol_v2.json`](../benchmark/v2/protocol_v2.json))
- Protocol SHA256: `e5a5c0335fcc239edae4a085eb66b9ac5b93d4c44917c7168d21761c12675ba0` (frozen, unchanged)
- Target Hardware: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`, CUDA 13.0, driver `580.82.07`, Colab G4)
- Hardware Acceptance: [`reports/v2/gpu_runtime_status.json`](../reports/v2/gpu_runtime_status.json) → **`PASS`** (`checked_at_utc: 2026-11-13T15:07:24Z`)

---

## 1. Official G4 40-Clip V2 Results

Evaluated using Option B batched simulation (`benchmark/v2/batched.py`, `num_envs=40`, GPU zero-copy float32 MLP inference with `allow_tf32=False`, full V2 `FULL_COLLISION` physics and untouched V2 scoring):

| Algorithm | Checkpoint SHA256 | Native Eval | V2 G4 Return | V2 Failed Clips | Rollout Wall | Steady Throughput | Record |
|---|---|---|---:|---:|---:|---:|---|
| **SuperMimic (Official, `GenCoord` `proprio` `h=10`)** | `83b9417dfa10…` (`g4_official_gencoord_proprio_h10`, train `3419.35 s`) | — *(trained on G4 from scratch)* | **0.84631** | **1 / 40** (`jumps1_subject2`) | 140.6 s (`149.3 s` total) | 3,141.4 steps/s | [`reports/v2/g4_official_supermimic_seed0.json`](../reports/v2/g4_official_supermimic_seed0.json) |
| **NablaMimic** | `4086e7a9bd38…` (`ea_s0_20261106T142202Z`) | `0.8345` (`0/40`) | **0.82817** | **1 / 40** (`fallAndGetUp2_subject3`) | 139.3 s (`165.9 s` total) | 3,150.5 steps/s | [`reports/v2/g4_nablamimic_seed0.json`](../reports/v2/g4_nablamimic_seed0.json) |

> [!NOTE]
> `SuperMimic-vibed` (`8bfc1158`, `816a02a4`, `g4_supermimic_seed0.json` / `a100_supermimic_seed0.json`) has been retired from the official V2 baseline table in favor of the physics-matched official `HybridRobotics/supermimic` baseline (`0.84631`). Sections 2–4 below are preserved purely as historical root-cause diagnostic records.

### Option B vs Serial 2-Clip Parity Notes on G4
- [`reports/v2/g4_parity_official_supermimic_2clips.json`](../reports/v2/g4_parity_official_supermimic_2clips.json): **`PASS` (`0 / 2` mismatched)**
  - `walk1_subject1`: `match=true` (`0.88297` vs `0.88357`, `|Δret|=0.000595`, `max|Δterm|=3.16e-5`, both 13,064 alive steps)
  - `dance1_subject1`: `match=true` (`0.85088` vs `0.85112`, `|Δret|=0.000233`, `max|Δterm|=1.58e-5`, both 6,573 alive steps)
- [`reports/v2/g4_parity_nablamimic_2clips.json`](../reports/v2/g4_parity_nablamimic_2clips.json):
  - `walk1_subject1`: `match=true` (`0.88715` vs `0.89056`, `|Δret|=0.00341`, both 13,064 alive steps)
  - `fallAndGetUp2_subject3`: `match=false` only because the exact fall step shifted by 18 steps (`4916` vs `4934`, both failed on `root_height`, `|Δret|=0.00246 < 0.01 tol`)

---

## 2. CTO Root-Cause Diagnostics (`816a02a4` & `0.52869` Drop)

Following the CTO audit, three minimal verification experiments were executed without altering [`protocol_v2.json`](../benchmark/v2/protocol_v2.json) or overwriting official V2 records.

| Priority | Experiment | Purpose | Outcome | Artifact |
|---|---|---|---|---|
| **P0-1** | Verify `2026-11-04` (`816a02a4`) Drive training logs & native eval | Rule out a broken/collapsed checkpoint | **Checkpoint valid**: native `0.81913` (`1/40` failed) | Google Drive `lafan1_main_20261104T061942Z` |
| **P0-2** | V2 40-clip diagnostic ablation with `FULL_COLLISION_WITHOUT_SELF` | Quantify causal impact of self-collision physics mismatch | **`0.78908` (`1/40` failed)**; rescues **23 / 24** clips | [`reports/v2/diag_g4_supermimic_no_self_collision.json`](../reports/v2/diag_g4_supermimic_no_self_collision.json) |
| **P1** | Compare FP32 vs `torch.autocast("cuda", bfloat16)` in open & closed loop | Rule out inference-precision mismatch | **`0.58543` (`19/40` failed)**; shifts edge clips only | [`reports/v2/diag_g4_supermimic_bf16_full_collision.json`](../reports/v2/diag_g4_supermimic_bf16_full_collision.json) |

### P0-1: Checkpoint Provenance (`2026-11-03` vs `2026-11-04`)

| Item | Local Baseline (`2026-11-03`) | G4 Evaluated (`2026-11-04`) |
|---|---|---|
| Checkpoint file | `ckpt_015000.pt` | `ckpt_015000.pt` |
| SHA256 prefix | `93bc5c43` | `816a02a4` |
| `args.json` (`seed`, `deterministic`) | `0`, `false` (identical hyperparameters) | `0`, `false` (identical hyperparameters) |
| Training wall time | `1.4509 h` | `1.448 h` (`3.36 it/s`) |
| Native `ckpt_015000_eval.json` | `0.84165` (`norm_return_framew = 0.84421`), `0/40` failed | `0.81913` (`norm_return_framew = 0.81922`), `1/40` failed (`jumps1_subject5: 0.1971`; other 39 clips `0.7613–0.9168`) |
| Training progression (`eval.csv`) | `0.5480@2k -> 0.8205@10k -> 0.8289@12k -> 0.8310@14k (2 failed) -> 0.8417@15k (0 failed)` | `0.5263 -> 0.8245@10k -> 0.8327@12k -> 0.8438@14k (0/40 failed) -> 0.8191@15k (1/40 failed)` |

Conclusion: `816a02a4` is a healthy, normally converged 15,000-iteration checkpoint; the SHA mismatch with `93bc5c43` is expected given `"deterministic": false` across separate training runs, and its native performance (`~0.82`, `0.8438` at 14k) rules out a defective checkpoint.

### P0-2: Self-Collision Physics Ablation (Diagnostic Only, `submission_eligible=false`)

- Standard V2 keeps `mjlab`'s `FULL_COLLISION` enabled (`self_collision=true`).
- Upstream `zita-ch/supermimic_vibed` trained with `enabled_self_collisions=False` in `supermimic/robot.py`, whereas `NablaMimic` trained with self-collisions enabled.
- Running the exact same `816a02a4` policy (`BatchedSuperMimicPolicy`, FP32, untouched V2 reward/termination/normalization) under `mjlab`'s `FULL_COLLISION_WITHOUT_SELF` (`contype=0, conaffinity=1` on the 33 robot collision geoms while keeping terrain collision and foot friction intact):
  - Aggregate return jumps from **`0.52869` → `0.78908`** (`+0.26039`).
  - Failed clips drop from **`24 / 40` → `1 / 40`** (only `fallAndGetUp1_subject1` failed at step `4965 / 8409` on `root_height`, `return=0.42122`), rescuing **23 of 24** failed clips with **0** regressions.
  - The remaining `~0.03` gap versus native `0.81913` aligns with secondary physics/contact/friction (`0.6` vs `1.0`) and reward-anchor (`torso_link` vs `pelvis`) differences.

### P1: BF16 vs FP32 Inference Precision (Diagnostic Only, `submission_eligible=false`)

- Note that official G4 Option B evaluation (`BatchedSuperMimicPolicy` in `benchmark/v2/batched.py`) already ran **PyTorch CUDA FP32** (`allow_tf32=False`), not NumPy.
- Open-loop check across 200 realistic inputs (`torch.autocast("cuda", dtype=torch.bfloat16)` vs FP32 `allow_tf32=False`): `max_abs_action_diff = 0.03194`, `mean = 0.00490`, `p99 = 0.02040`.
- Closed-loop 40-clip V2 check under standard `FULL_COLLISION` with `autocast(bfloat16)`:
  - Aggregate return **`0.58543`**, **`19 / 40` failed** (7 borderline clips survived: `dance2_subject1`, `fight1_subject2`, `fight1_subject5`, `run1_subject2`, `run2_subject4`, `walk1_subject2`, `walk3_subject3`; 2 newly failed: `dance2_subject4`, `run2_subject1`).
  - Trajectory divergence confirms numerical sensitivity on borderline self-collision episodes, but BF16 does not resolve the `FULL_COLLISION` collapse.

---

## 3. Conclusions & Next Step

1. **V2 Evaluator (`e5a5c033...`) & G4 Acceptance**: Verified and consistent. No protocol change is warranted.
2. **Primary Cause of SuperMimic V2 Drop**: Proven causally to be the **training-vs-evaluation self-collision mismatch** (`enabled_self_collisions=False` during SuperMimic training vs `FULL_COLLISION` in V2), **not** a broken checkpoint and **not** FP32 vs BF16 precision.
3. **Resolution (Superseded by Official `HybridRobotics/supermimic`)**: Rather than investing further G4 compute in the unofficial `supermimic_vibed` fork, we replaced it as an official V2 baseline with official `HybridRobotics/supermimic` (`SuperMimic-GenCoord-G1` trained on G4 with `FULL_COLLISION` and `proprio`, `history=10` — see Section 5). Note that Sections 2–4 below summarize historical `supermimic_vibed` diagnostics.

---

## 4. SuperMimic Policy Adapter Off-by-One Fix & Post-Fix A100 Re-Evaluation (2026-11-14)

### 4.1 Root Cause in the Evaluator Adapter (Not Protocol, Not NablaMimic)

`mjlab`'s `MotionCommand` resets the robot to reference frame `0`, then `ManagerBasedRlEnv.reset()` invokes `command_manager.compute(dt=0.0)` which executes `MotionCommand._update_command()` and increments `time_steps` from `0` to `1` even at `dt=0`. Consequently, on the first control step (`robot` physically at frame `0`), `StepInput.reference_index` is `1`.

- **NablaMimic (`g4_nablamimic_seed0.json`, `0.82817`, `1/40`) was NOT affected**: both serial `NablaMimicPolicy` and `BatchedNablaMimicPolicy` already subtracted `1` (`frame = max(global_frames - 1, _slot_starts)`) and read `anchor_ori_b_prev`.
- **SuperMimic-vibed WAS affected before this fix**: both serial `SuperMimicPolicy` ([`benchmark/policies/supermimic.py`](../benchmark/policies/supermimic.py)) and batched `BatchedSuperMimicPolicy` ([`benchmark/v2/batched.py`](../benchmark/v2/batched.py)) read `reference_index` (`f + 1`) and `anchor_ori_b` (`f + 1`) directly, shifting the reference orientation, projected gravity, and future goal window (`f+2..f+11` instead of `f+1..f+10`) ahead by 1 frame on every step. Therefore, the earlier G4 records (`0.52869`, `0.78908`, `0.58543`) are archived as **pre-fix** measurements.

### 4.2 Targeted Adapter Fix & Independent Verification

Without modifying [`protocol_v2.json`](../benchmark/v2/protocol_v2.json), V2 physics (`FULL_COLLISION`), `supermimic_vibed` training code, or `816a02a4`:

1. **Serial & Batched SuperMimic Adapters fixed** (`benchmark/policies/supermimic.py`, `benchmark/v2/batched.py`):
   - Observation & `q_ref` / `g_ref`: use `frame = reference_index - 1` (serial) / `torch.maximum(global_frames - 1, _slot_starts)` (batched) and `anchor_ori_b_prev` (fail-closed if missing).
   - Goal horizon: computes `f+1 .. f+10` relative to `yaw(R_ref[f])` and `p_ref[f]`.
2. **Independent Upstream & Adapter Verification on Colab A100 (`816a02a4`)**:
   - [`tests/test_supermimic_temporal_alignment.py`](../tests/test_supermimic_temporal_alignment.py) + [`tests/test_supermimic_policy.py`](../tests/test_supermimic_policy.py) with live `816a02a4` & upstream `supermimic_vibed`: **`20 passed, 0 skipped`**.
   - [`scripts/check_supermimic_parity.py`](../scripts/check_supermimic_parity.py) against upstream `MotionLib.goal` + `MDP.obs` + `Policy.forward` (`fallAndGetUp3_subject1`, 300 steps, FP32): **`max|delta| = 3.099e-06`**, `mean|delta| = 3.195e-07`.
   - 200-step lockstep parity (`walk1_subject1`, same `StepInput`/GPU state) between `BatchedSuperMimicPolicy` (`supermimic_gpu`) and serial `SuperMimicPolicy`: **`max|delta| = 1.550e-06`**.
   - 1-clip GPU Smoke (`walk1_subject1` serial): **`+0.86704`**, `0` failed (`13064/13064` alive). (In single-env `batch_envs=1` smoke, `supermimic_gpu` diverged late at step `11477/13064`, `+0.76383`; in the 40-env `batch_envs=40` evaluation below, `walk1_subject1` survived `13064/13064` at **`+0.86706`**.)

### 4.3 Post-Fix 40-Clip V2 Re-Evaluation on A100 (`FULL_COLLISION`)

To conserve Colab G4 compute units as requested, the post-fix 40-clip V2 evaluation of `816a02a4` under canonical `FULL_COLLISION` ran on **`NVIDIA A100-SXM4-40GB` (`sm_80`)**:

| Condition | Hardware / Scope | Checkpoint | Aggregate Return | Failed Clips | Notes | Artifact |
|---|---|---|---:|---:|---|---|
| **Pre-fix** G4 (`FULL_COLLISION`) | G4 `sm_120` (superseded) | `816a02a4` | `0.52869` | `24 / 40` | 1-frame shifted adapter | [`g4_supermimic_seed0.json`](../reports/v2/g4_supermimic_seed0.json) |
| **Post-fix** A100 (`FULL_COLLISION`) | **A100 `sm_80`** (`controlled_baseline_non_target_hardware`, `submission_eligible=false`) | `816a02a4` | **`0.56999`** | **`20 / 40`** | Aligned adapter (`841.2` steady steps/s) | [`a100_supermimic_seed0.json`](../reports/v2/a100_supermimic_seed0.json) |
| G4 NablaMimic (`FULL_COLLISION`, unaffected) | G4 `sm_120` (`acceptance`) | `4086e7a9` | `0.82817` | `1 / 40` | Already frame-aligned | [`g4_nablamimic_seed0.json`](../reports/v2/g4_nablamimic_seed0.json) |

**Key Observations (Pre-fix G4 `0.52869` → Post-fix A100 `0.56999`)**:
- **Net improvement**: **`+0.04130`** return and **`4` fewer failed clips** (`24/40` → `20/40`): **5 clips rescued** (`dance2_subject2: 0.30236 -> 0.78713`, `fight1_subject3: 0.40097 -> 0.80798`, `fight1_subject5: 0.24636 -> 0.81326`, `run1_subject2: 0.23007 -> 0.81013`, `walk1_subject2: 0.52997 -> 0.85656`), **1 borderline late-step regression** (`run2_subject1: 0.77398 (12239/12239) -> 0.76764 (11885/12239)`).
- **Consistent gain on stable clips**: across the 15 clips passing in both runs, **14/15** improved with mean delta **`+0.00638`** (highest on dynamic clips: `run1_subject5 +0.01927`, `dance2_subject4 +0.01725`, `dance2_subject5 +0.01245`, `dance1_subject3 +0.01168`).
- **Two effects cleanly separated**: fixing the 1-frame adapter bug materially improves tracking fidelity and rescues 5 borderline clips, **and** even with exact frame alignment (`max|delta| = 3.1e-6`), `816a02a4` still fails `20/40` clips (`0.56999` vs native `0.81913` and pre-fix no-self-collision `0.78908`), confirming that the remaining `~0.22–0.25` gap under canonical V2 `FULL_COLLISION` is dominated by the training-vs-evaluation physics/self-collision mismatch (`enabled_self_collisions=False` during `supermimic_vibed` training).

> [!NOTE]
> [`reports/v2/a100_supermimic_seed0.json`](../reports/v2/a100_supermimic_seed0.json) (`target_hardware_met=false`, `submission_eligible=false`) and [`reports/v2/g4_supermimic_seed0.json`](../reports/v2/g4_supermimic_seed0.json) are now archived in `retired_baselines` inside [`reports/v2/registry.json`](../reports/v2/registry.json); the active official SuperMimic V2 baseline on G4 is Section 5 below.

---

## 5. Official `HybridRobotics/supermimic` Deployable G4 Baseline (2026-11-15)

To establish a direct, physics-matched (`FULL_COLLISION`), sensor-only baseline from the official upstream implementation without privileged teacher state or DAgger distillation:

- **Upstream commits**: `HybridRobotics/supermimic` @ `abc124c1bd5976341034668af3d12c087f2f1f55`, `HybridRobotics/mjlab_research` @ `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`
- **Training configuration (Colab G4, `RTX PRO 6000 Blackwell`, `sm_120`)**:
  - Task: `SuperMimic-GenCoord-G1` on the canonical 40 LAFAN1 clips (`--runner.actor-obs proprio --runner.history-length 10`, default `ActorCriticPolicyLearnerCfg`, `FULL_COLLISION` physics).
  - Iterations / Wall time: `25,001` iterations in **`3419.35 s` (`0.950 h`)**, producing `actor.pt` (SHA256 `83b9417dfa10aba86fd82c4923f0c8afae7e7317e6bc76a36010bed717cfe9b5`).
- **Adapter & Live Numerical Parity**:
  - Policy adapter [`benchmark/policies/official_supermimic.py`](../benchmark/policies/official_supermimic.py) (`OfficialSuperMimicPolicy`, `93`-dim proprio $\times 10$ history + `640`-dim strided goal reconstructed strictly from public `StepInput`) + GPU batched path [`benchmark/v2/batched.py`](../benchmark/v2/batched.py) (`BatchedOfficialSuperMimicPolicy`, FP32 `allow_tf32=False`).
  - Unit + reconstruction suite [`tests/test_official_supermimic_policy.py`](../tests/test_official_supermimic_policy.py) + [`tests/test_v2_batched.py`](../tests/test_v2_batched.py) on G4: **`17 passed`**.
  - 25-step closed-loop live parity against upstream `HybridRobotics/supermimic` `MultiEncoderAE` (zero `encoder_bias`): `max|obs| = 3.181e-07`, `max|goal| = 4.923e-07`, `max|act| = 4.768e-07`.
  - 2-clip Option B (`num_envs=2`) vs serial (`workers=2`) closed-loop V2 parity ([`reports/v2/g4_parity_official_supermimic_2clips.json`](../reports/v2/g4_parity_official_supermimic_2clips.json)): **`PASS` (`0/2` mismatched, `max|Δret|=0.000595`, `max|Δterm|=3.16e-5`)**.
- **G4 40-Clip V2 `FULL_COLLISION` Acceptance Result** ([`reports/v2/g4_official_supermimic_seed0.json`](../reports/v2/g4_official_supermimic_seed0.json)):
  - **Aggregate Normalized Return**: **`0.84631`** (`submission_eligible=true`, `target_hardware_met=true`, `report_scope="acceptance"`, `protocol_sha256="e5a5c033..."`).
  - **Failed Clips**: **`1 / 40`** (`jumps1_subject2`: `0.77013`, terminated at step `11614/12221`); remaining 39 clips passed full duration (`0.78884` on `dance1_subject2` to `0.91424` on `walk4_subject1`).
  - **Eval Speed**: `140.6 s` rollout (`149.3 s` total), **`3,141.4` steady steps/s**.

### Follow-Up 35k Sweep & Earliest `0/40` Bisection (`official_35k_20261115T120918Z`)

A follow-up `35,001`-iteration G4 run using the identical upstream commits (`abc124c1...` / `dd51a9fc...`) and default hyperparameters while saving snapshots every `500` iterations (`4632.13 s` total; full curve and bisect artifacts archived in [`../baselines/supermimic_mjlab/RESULTS.md`](../baselines/supermimic_mjlab/RESULTS.md) and [`../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/bisect_report.json`](../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/bisect_report.json)) resolved the single `25,001`-iter failure and mapped the time-to-`0/40` threshold (`return >= 0.82`):

- **Earliest `0/40` passing (`return >= 0.82`)**: **`iter = 15,000`** (`2210.83 s` / `0.614 h`), **`0.82641`**, **`0 / 40` failed** (with single-clip edge oscillation at `14500`: `1/40`, `15500`: `1/40`, `16000`: `0/40`, `16500`: `1/40`).
- **Stable `0/40` onset**: **`iter = 17,000`** (`2453.09 s` / `0.681 h`, **`0.83635`**, **`0 / 40` failed**); every evaluated snapshot from `17,000` through `35,000` (`17000`, `17500`, `18500`, `24500..35000`) recorded `0 / 40` failures.
- **Peak (`iter = 34,500`, `4568.23 s`)**: **`0.85751`**, **`0 / 40` failed** (2-clip Option B vs serial parity `PASS` in [`parity_best_2clips.json`](../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/parity_best_2clips.json): `0/2` mismatched, `max|Δret|=0.00583`, `max|Δterm|=6.13e-4`); `final` at `iter = 35,000` (`4632.13 s`) scored **`0.85264`**, `0 / 40` failed.
- **Registry status note**: [`reports/v2/registry.json`](../reports/v2/registry.json) retains the audited `25,001`-iter per-clip file (`g4_official_supermimic_seed0.json`) in `records[]` until the full per-clip `34,500`/`15,000` JSONs are imported from Google Drive (`MyDrive/supermimic_v2/runs/official_35k_20261115T120918Z/evals/`).
