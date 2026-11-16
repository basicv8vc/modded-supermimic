# NablaMimic run report — v2_25m_s0_20261116T021256Z

| 字段 | 内容 |
|---|---|
| experiment_id | v2_25m_s0_20261116T021256Z |
| algorithm | NablaMimic |
| git_sha | bfbb2a6773b4776c17402a34a93ec7589d9c7c18 |
| engine | NablaEngine (nablaengine[jax] @ c61a6fe7f5ec47c7f4f9b0a4c055d60cbe40807c) |
| seed | 0 |
| gpu | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| model | G1 29-DoF (Menagerie unitree_g1/scene_mjx.xml @ f3b6013, 31 bodies / 502 contact pairs) |
| training_time_min | 24.97 |
| updates | 3936 |
| normalized_return | None |
| failed_clips | None |
| drive_path | /content/drive/MyDrive/supermimic/nablamimic/runs/v2_25m_s0_20261116T021256Z |
| status | INCOMPLETE |

## Checkpoint curve

| iter | wall_min | full_norm_return | failed_frac | rew_mean |
|---:|---:|---:|---:|---:|
| 0 | 0.02 |  |  | 0.3744380474090576 |
| 250 | 1.62 |  |  | 0.6045236587524414 |
| 500 | 3.2 |  |  | 0.6949520111083984 |
| 750 | 4.79 |  |  | 0.7222703099250793 |
| 1000 | 6.37 |  |  | 0.7340661287307739 |
| 1250 | 7.95 |  |  | 0.7563154697418213 |
| 1500 | 9.54 |  |  | 0.7720813751220703 |
| 1750 | 11.13 |  |  | 0.7776584625244141 |
| 2000 | 12.71 |  |  | 0.7827082872390747 |
| 2250 | 14.3 |  |  | 0.7865767478942871 |
| 2500 | 15.88 |  |  | 0.7944167256355286 |
| 2750 | 17.47 |  |  | 0.7988910675048828 |
| 3000 | 19.06 |  |  | 0.7997531294822693 |
| 3250 | 20.65 |  |  | 0.8052129745483398 |
| 3500 | 22.24 |  |  | 0.8045280575752258 |
| 3750 | 23.83 |  |  | 0.8032241463661194 |
| 3936 | 0.0 |  |  |  |

> 评估未运行或未产出 eval_nominal.json — 状态 INCOMPLETE。

## Integrity problems

- no evaluated checkpoint

> 这些问题使该 run 不能标为 PASS，也不会进入多 seed 汇总。

## Notes

- Effective training budget is set by `--hours` only; the algorithm, reward, model and optimizer parameters are the official `configs/ours.args` values.
- Wall-clock in `log.jsonl` (`wall_s`) counts training only, excluding JIT compilation and evaluation (matches the paper's timing).
- Evaluation scores the Polyak-averaged (EMA) actor, decay 0.999 with warmup.
