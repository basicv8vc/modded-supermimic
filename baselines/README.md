# Baseline Training Workflows

Training in `modded-supermimic` is intentionally decoupled from the frozen V2 referee (`benchmark/`): any RL algorithm, policy architecture, or training simulator can compete as long as it trains from scratch on the 40 canonical LAFAN1 clips and clears the V2 gate (`>= 0.82` normalized return, `0/40` failed clips) on 1× Colab G4.

| Baseline | Framework & Training Sim | Reproducible Entry | Key V2 Result on G4 |
| --- | --- | --- | --- |
| [`instanttrack/`](instanttrack/README.md) | PyTorch + `mjlab` / MuJoCo Warp (`Huy-Quang-Dao/InstantTrack` `v1.4.0` + `BlinkMimic`) | [`baselines/instanttrack/run.sh`](instanttrack/run.sh) | Earliest `0/40`: **`6m 45s`** (`405.23 s` raw; ~`3m 24s` / `204.16 s` net train, iter `2,300`, `0.82169`); `15m` end: `0.84308` @ `7,920` iters |
| [`nablamimic/`](nablamimic/README.md) | JAX + `NablaEngine 0.1.0` (`sailab-ai/NablaMimic`) | [`baselines/nablamimic/run.sh`](nablamimic/run.sh) | Earliest & stable `0/40`: **`9m 32s`** (`572.18 s`, iter `1,500`, `0.84108`); `25m` end: `0.86896` @ `3,936` iters |
| [`supermimic_mjlab/`](supermimic_mjlab/README.md) | PyTorch + `mjlab` (`HybridRobotics/supermimic`) | [`baselines/supermimic_mjlab/run.sh`](supermimic_mjlab/run.sh) | Earliest `0/40`: **`36m 51s`** (`2,210.83 s`, iter `15,000`, `0.82641`); stable `0/40`: **`40m 53s`** (`2,453.09 s`, iter `17,000`, `0.83635`); peak `0.85751` @ `34,500` iters |
