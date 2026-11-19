# Record 003 — InstantTrack v1.4 (`FULL_COLLISION`, `seed=0`)

- **Algorithm**: `InstantTrack-v1.4-FULL_COLLISION` (`Huy-Quang-Dao/InstantTrack` @ `b9512a96d4e82e6ddea9a6de3b03276888bbcb2c`, `v1.4.0`, incorporating [BlinkMimic](https://arxiv.org/abs/2611.16422) `fuyichen0725/blinkmimic` @ `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`)
- **Trainer / Simulator**: PyTorch + `mjlab` / MuJoCo Warp (world-model BPTT + Muon + `--pi_ref_skip` + `--wm_structured` + `--eprobe_active 372` + CUDA graphs)
- **Hardware**: `1 × NVIDIA RTX PRO 6000 Blackwell Server Edition` (`sm_120`, Colab G4)
- **Seed**: `0`
- **Run ID**: `instanttrack_v14_s0_20261119T123338Z` (`PRESET[instanttrack]`, `BUDGET_MINUTES=15`, `7,920` iters in `900.10 s` / `15.00 min`)
- **V2 Physics & Data Alignment**: Upstream `g1_constants.py` was patched via [`baselines/instanttrack/vm/patch_full_collision.py`](../../../baselines/instanttrack/vm/patch_full_collision.py) to use frozen [`assets/g1/g1.xml`](../../../assets/g1/g1.xml) with `FULL_COLLISION` enabled (`njmax=1024`, `nconmax=256`) and the pinned 40 LAFAN1 CSVs from [`scripts/fetch_lafan1.py`](../../../scripts/fetch_lafan1.py); algorithm hyperparameters were kept at upstream `PRESET[instanttrack]` defaults.

## Key Milestones (V2 `FULL_COLLISION`)

`wall_time` reported by `instanttrack.train` starts before iteration 1's one-time `torch.compile` / CUDA-graph warmup (`it 1` recorded at `201.07 s` in `eval.csv`). Both raw wall-clock and compile-excluded net training time (`raw_wall - 201.07 s`) are shown below:

| Checkpoint | Iter | Raw Wall (s) | Net Train Wall (~s) | Aggregate Return | Failed Clips | Full 40-Clip V2 JSON |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior checkpoint (`ckpt_002200`) | `2,200` | `396.48 s` (`6m 36s`) | ~`195.41 s` (`3m 15s`) | `0.81271` | `1 / 40` | [`ckpt_002200_v2.json`](../../../reports/v2/instanttrack_v14_s0/ckpt_002200_v2.json) |
| **Earliest `0/40` + `>=0.82` (`ckpt_002300`)** | **`2,300`** | **`405.23 s` (`6m 45s`)** | **~`204.16 s` (`3m 24s`)** | **`0.82169`** | **`0 / 40`** | [`ckpt_002300_v2.json`](../../../reports/v2/instanttrack_v14_s0/ckpt_002300_v2.json) |
| Edge oscillation (`ckpt_002400`) | `2,400` | `413.99 s` (`6m 54s`) | ~`212.92 s` (`3m 33s`) | `0.80696` | `1 / 40` | [`ckpt_002400_v2.json`](../../../reports/v2/instanttrack_v14_s0/ckpt_002400_v2.json) |
| Adjacent passing (`ckpt_002466`) | `2,466` | `420.22 s` (`7m 00s`) | ~`219.15 s` (`3m 39s`) | `0.82461` | `0 / 40` | [`ckpt_002466_v2.json`](../../../reports/v2/instanttrack_v14_s0/ckpt_002466_v2.json) |
| **Final / Peak V2 evaluated (`15m` endpoint)** | **`7,920`** | **`900.10 s` (`15m 00s`)** | **~`699.03 s` (`11m 39s`)** | **`0.84308`** | **`0 / 40`** | [`ckpt_007920_v2.json`](../../../reports/v2/instanttrack_v14_s0/ckpt_007920_v2.json) |

## Parity & Reproduction

- **2-clip Option B vs serial V2 parity (`ckpt_002300`)**: [`parity_ckpt_002300_2clips.json`](../../../reports/v2/instanttrack_v14_s0/parity_ckpt_002300_2clips.json) — **`PASS`** (`0 / 2` mismatched, `max|Δreturn| = 2.689e-03`, `max|Δterm| = 2.246e-04`; `dance1_subject1`: `0.80566` vs `0.80592`, `walk1_subject1`: `0.87644` vs `0.87375`).
- **Policy adapter**: [`baselines/instanttrack/policy.py`](../../../baselines/instanttrack/policy.py) (`reset(context)` / `act(step)`).
- **Training logs & screen curve**: [`baselines/instanttrack/out/instanttrack_v14_s0_20261119T123338Z/`](../../../baselines/instanttrack/out/instanttrack_v14_s0_20261119T123338Z/stdout.log) (`args.json`, `eval.csv`, `launcher_meta.json`, `stdout.log`).
- **Reproduce end-to-end on Colab G4**: [`baselines/instanttrack/run.sh`](../../../baselines/instanttrack/run.sh) (`./run.sh all`).
