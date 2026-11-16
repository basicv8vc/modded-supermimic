# Record 002 — NablaMimic (`configs/ours.args`, `NablaEngine 0.1.0`)

- **Algorithm**: `NablaMimic` (`sailab-ai/NablaMimic` @ `bfbb2a6773b4776c17402a34a93ec7589d9c7c18`, `configs/ours.args`)
- **Trainer / Simulator**: JAX + `NablaEngine 0.1.0` @ `c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c`
- **Hardware**: `1 × NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`, Colab G4)
- **Seed**: `0`
- **Run ID**: `v2_25m_s0_20261116T021256Z` (`--hours 0.41667 --seed 0`, `3,936` iters in `1,498.46 s`)

## Key Milestones (V2 `FULL_COLLISION`)

| Checkpoint / Run | Iter | Training Wall (s) | Aggregate Return | Failed Clips | Full 40-Clip V2 JSON |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior 8-min baseline (`ea_s0_20261106T142202Z`) | — | ~`480 s` (`8m`) | `0.82817` | `1 / 40` (`fallAndGetUp2_subject3`) | [`g4_nablamimic_seed0.json`](../../../reports/v2/g4_nablamimic_seed0.json) |
| Last failing checkpoint in 25m bisect | `1,250` | `477.20 s` (`7m 57s`) | `0.82498` | `2 / 40` (`fallAndGetUp1_subject5`, `fallAndGetUp3_subject1`) | [`v2_it_001250.json`](../../../reports/v2/nablamimic_25m_bisect/v2_it_001250.json) |
| **Earliest & Stable `0/40` + `>=0.82`** | **`1,500`** | **`572.18 s` (`9m 32s`)** | **`0.84108`** | **`0 / 40`** | [`v2_it_001500.json`](../../../reports/v2/nablamimic_25m_bisect/v2_it_001500.json) |
| Final / Peak checkpoint (`25m` endpoint) | `3,936` | `1,498.46 s` (`24m 58s`) | `0.86896` | `0 / 40` | [`v2_it_003936.json`](../../../reports/v2/nablamimic_25m_bisect/v2_it_003936.json) |

Every evaluated checkpoint from `it_001500` through `it_003936` (11 checkpoints at 250-iter intervals + final `3,936`) achieved **`0 / 40` failures** and monotonically rising normalized returns (`0.84108` → `0.86896`). See [`bisect_summary.json`](../../../reports/v2/nablamimic_25m_bisect/bisect_summary.json).

## Training & Evaluation Artifacts

- Training run archive: [`baselines/nablamimic/out/v2_25m_s0_20261116T021256Z/`](../../../baselines/nablamimic/out/v2_25m_s0_20261116T021256Z/summary.json) (`config.json`, `curve.csv`, `train.log`, `provenance.json`, `manifest.json`)
- Reproduce end-to-end on Colab G4: [`baselines/nablamimic/run.sh`](../../../baselines/nablamimic/run.sh) (`RUN=ours BUDGET_MINUTES=25 DRIVE=1 ./run.sh all`)
