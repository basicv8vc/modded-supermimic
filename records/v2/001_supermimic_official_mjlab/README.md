# Record 001 — Official SuperMimic (`HybridRobotics/supermimic`, `GenCoord` `proprio`, `h=10`)

- **Algorithm**: `SuperMimic-official-gencoord-proprio` (`--runner.actor-obs proprio --runner.history-length 10`, default `ActorCriticPolicyLearnerCfg`, trained from scratch)
- **Trainer / Simulator**: PyTorch + `mjlab` (`HybridRobotics/supermimic` @ `abc124c1bd5976341034668af3d12c087f2f1f55`, `HybridRobotics/mjlab_research` @ `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`)
- **Hardware**: `1 × NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`, Colab G4)
- **Seed**: `0`

## Key Milestones (V2 `FULL_COLLISION`)

| Checkpoint / Run | Iter | Training Wall (s) | Aggregate Return | Failed Clips | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Audited 25k baseline (`g4_official_gencoord_proprio_h10`) | `25,001` | `3,419.35 s` (`56m 59s`) | `0.84631` | `1 / 40` (`jumps1_subject2`) | [`g4_official_supermimic_seed0.json`](../../../reports/v2/g4_official_supermimic_seed0.json) |
| **35k sweep: Earliest `0/40` & `>=0.82`** (`official_35k_20261115T120918Z`) | **`15,000`** | **`2,210.83 s` (`36m 51s`)** | **`0.82641`** | **`0 / 40`** | [`bisect_report.json`](../../../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/bisect_report.json) |
| **35k sweep: Stable `0/40` onset** | **`17,000`** | **`2,453.09 s` (`40m 53s`)** | **`0.83635`** | **`0 / 40`** | [`RESULTS.md`](../../../baselines/supermimic_mjlab/RESULTS.md) |
| 35k sweep: Peak passing | `34,500` | `4,568.23 s` (`76m 08s`) | `0.85751` | `0 / 40` | [`parity_best_2clips.json`](../../../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/parity_best_2clips.json) |
| 35k sweep: Final | `35,000` | `4,632.13 s` (`77m 12s`) | `0.85264` | `0 / 40` | [`sweep_summary.json`](../../../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/sweep_summary.json) |

> [!NOTE]
> Per-snapshot 40-clip JSONs from `official_35k_20261115T120918Z` remain in Google Drive (`MyDrive/supermimic_v2/runs/official_35k_20261115T120918Z/evals/`), while the audited 25k full 40-clip JSON is checked into [`reports/v2/g4_official_supermimic_seed0.json`](../../../reports/v2/g4_official_supermimic_seed0.json).

## Parity & Reproduction

- 2-clip Option B vs serial V2 parity (`25k` record): [`g4_parity_official_supermimic_2clips.json`](../../../reports/v2/g4_parity_official_supermimic_2clips.json) (`PASS`, `0/2` mismatched, `max|Δreturn|=0.000595`).
- 2-clip Option B vs serial V2 parity (`34.5k` peak): [`parity_best_2clips.json`](../../../baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/parity_best_2clips.json) (`PASS`, `0/2` mismatched, `max|Δreturn|=0.00583`).
- Reproduce end-to-end on Colab G4: [`baselines/supermimic_mjlab/run.sh`](../../../baselines/supermimic_mjlab/run.sh) (`ITERS=35001 SAVE_INTERVAL=500 ./run.sh all`).
