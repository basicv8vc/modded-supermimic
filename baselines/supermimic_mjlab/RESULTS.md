# Official `HybridRobotics/supermimic` 35k Sweep & Earliest `0/40` Analysis

- **Run ID**: `official_35k_20261115T120918Z`
- **Upstream commits**:
  - `HybridRobotics/supermimic`: `abc124c1bd5976341034668af3d12c087f2f1f55`
  - `HybridRobotics/mjlab_research`: `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`
- **Training configuration**:
  - Task: `SuperMimic-GenCoord-G1` on the canonical 40 LAFAN1 clips (`--runner.actor-obs proprio --runner.history-length 10`, default hyperparameters)
  - Iterations: `35,001` (`--runner.max-iterations 35001`, `--runner.save-interval 500`)
  - Hardware: Colab G4 (`NVIDIA RTX PRO 6000 Blackwell Server Edition` / `sm_120`)
- **Evaluator**: Benchmark V2 (`modded-supermimic-g1-lafan1-v2`, `protocol_sha256 = e5a5c0335fcc239edae4a085eb66b9ac5b93d4c44917c7168d21761c12675ba0`, Option B batched `num_envs=40`, `FULL_COLLISION` enabled, `return_threshold = 0.82` for earliest-passing gate)

## Key Conclusions

1. **Earliest `0/40` failed checkpoint (with `aggregate_return >= 0.82`)**:
   - **`actor_snap031_iter15000_2210.8s`** (`est_iter = 15000`, `wall_s = 2210.83 s` / `0.614 h`): **`0.82641`**, **`0 / 40` failed**.
2. **Boundary transition (`14500..17000` iters)**:
   - `14500..16500` exhibits single-clip edge oscillation (`14500`: `1/40`, `15000`: `0/40`, `15500`: `1/40`, `16000`: `0/40`, `16500`: `1/40`).
   - From **`iter = 17000`** (`2453.09 s`, `0.83635`) onward through **`35000`**, every evaluated checkpoint (`17000`, `17500`, `18500`, `24500..35000`) achieves **`0 / 40` failed**.
3. **Peak checkpoint (`best_passing`)**:
   - **`actor_snap070_iter34500_4568.2s`** (`est_iter = 34500`, `wall_s = 4568.23 s` / `1.269 h`): **`0.85751`**, **`0 / 40` failed**.
   - 2-clip Option B vs serial V2 parity (`out/official_35k_20261115T120918Z/parity_best_2clips.json`): **`PASS`** (`0 / 2` mismatched, `max|Δreturn| = 0.005828`, `max|Δterm| = 0.000613`; `walk1_subject1`: `0.88770` vs `0.89353`, `dance1_subject1`: `0.84125` vs `0.84084`).
4. **Final checkpoint (`final`)**:
   - `iter = 35000` (`4632.13 s` / `1.287 h`): **`0.85264`**, **`0 / 40` failed**.

## Evaluated Checkpoints (`bisect_report.json` / `sweep_summary.json`)

| Snapshot | Iter | Train Wall (s) | Aggregate Return | Failed Clips | Notes |
|---|---:|---:|---:|---:|---|
| `actor_snap025_iter12000_1847.2s` | 12000 | 1847.20 | 0.78958 | 4 | bisect probe |
| `actor_snap029_iter14000_2090.7s` | 14000 | 2090.71 | 0.80302 | 2 | edge check |
| `actor_snap030_iter14500_2150.8s` | 14500 | 2150.77 | 0.83096 | 1 | edge check |
| **`actor_snap031_iter15000_2210.8s`** | **15000** | **2210.83** | **0.82641** | **0** | **earliest `0/40` & `>=0.82`** |
| `actor_snap032_iter15500_2270.9s` | 15500 | 2270.89 | 0.80982 | 1 | edge jitter |
| `actor_snap033_iter16000_2330.9s` | 16000 | 2330.95 | 0.83896 | 0 | edge check |
| `actor_snap034_iter16500_2393.0s` | 16500 | 2393.01 | 0.82806 | 1 | edge jitter |
| **`actor_snap035_iter17000_2453.1s`** | **17000** | **2453.09** | **0.83635** | **0** | **stable `0/40` onset** |
| `actor_snap036_iter17500_2513.6s` | 17500 | 2513.57 | 0.83841 | 0 | edge check above |
| `actor_snap038_iter18500_2634.1s` | 18500 | 2634.12 | 0.83464 | 0 | bisect probe |
| `actor_snap050_iter24500_3360.6s` | 24500 | 3360.60 | 0.84167 | 0 | sweep |
| `actor_snap051_iter25000_3420.7s` | 25000 | 3420.66 | 0.84654 | 0 | sweep |
| `actor_snap053_iter26000_3541.1s` | 26000 | 3541.06 | 0.84286 | 0 | sweep |
| `actor_snap055_iter27000_3661.2s` | 27000 | 3661.19 | 0.84976 | 0 | sweep |
| `actor_snap057_iter28000_3783.3s` | 28000 | 3783.31 | 0.84077 | 0 | sweep |
| `actor_snap059_iter29000_3903.7s` | 29000 | 3903.74 | 0.84593 | 0 | sweep |
| `actor_snap061_iter30000_4024.7s` | 30000 | 4024.70 | 0.84840 | 0 | sweep |
| `actor_snap063_iter31000_4145.5s` | 31000 | 4145.53 | 0.84916 | 0 | sweep |
| `actor_snap065_iter32000_4266.5s` | 32000 | 4266.49 | 0.84658 | 0 | sweep |
| `actor_snap067_iter33000_4387.2s` | 33000 | 4387.20 | 0.85081 | 0 | sweep |
| `actor_snap068_iter33500_4447.7s` | 33500 | 4447.68 | 0.84900 | 0 | dense sweep |
| `actor_snap069_iter34000_4507.7s` | 34000 | 4507.74 | 0.85352 | 0 | sweep |
| **`actor_snap070_iter34500_4568.2s`** | **34500** | **4568.23** | **0.85751** | **0** | **best passing** |
| `final` | 35000 | 4632.13 | 0.85264 | 0 | final checkpoint |

## Local & Drive Artifacts

- Local summary artifacts (`out/official_35k_20261115T120918Z/`):
  - `bisect_report.json`, `selected.json`, `sweep_summary.json`, `parity_best_2clips.json`, `pipeline.log`
- Full checkpoints (`actor_snap*.pt`) and per-snapshot 40-clip V2 reports (`*_v2.json`) remain persisted in Google Drive under `MyDrive/supermimic_v2/runs/official_35k_20261115T120918Z/`.
