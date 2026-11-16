# V2 Speedrun Records

Each folder under `records/v2/` documents a verifiable `modded-supermimic-g1-lafan1-v2` record on **1× Colab G4 (`RTX PRO 6000 Blackwell`, `sm_120`)**, linked to its machine-audited 40-clip V2 evaluation JSON and reproducible training workflow.

| # | Record | Trainer / Sim | Earliest `0/40` & `>=0.82` | Stable `0/40` | Best / Endpoint | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 001 | [`001_supermimic_official_mjlab`](v2/001_supermimic_official_mjlab/README.md) | PyTorch + `mjlab` | `2,210.83 s` (`36m 51s`, iter `15,000`, `0.82641`) | `2,453.09 s` (`40m 53s`, iter `17,000`, `0.83635`) | `0.85751` @ `34,500` iters *(25k `1/40` audited in `records[]`)* | `0/40` in `35k` sweep |
| 002 | [`002_nablamimic_nablaengine`](v2/002_nablamimic_nablaengine/README.md) | JAX + `NablaEngine` | `572.18 s` (`9m 32s`, iter `1,500`, `0.84108`) | `572.18 s` (`9m 32s`, iter `1,500`, `0.84108`) | `0.86896` @ `3,936` iters (`1,498.46 s`, `0/40`) | Audited 40-clip JSONs committed |

All registered records are verified offline by:

```bash
python3 scripts/v2/audit_registry.py
```
