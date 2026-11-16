# Alignment report: v1 evaluator vs official mjlab implementation

Scope: Phase 3 numerical verification of the
"Official mjlab Reuse & Minimal Glue Refactor". The pre-refactor evaluator
(`canonical-mjlab-g1-v1`, mujoco-C float64, everything reimplemented locally)
is compared against the refactored evaluator (mjlab v1.6.0 @
`0fb8a681136be94ffc636a3dd423cabb97d91f10` running `ManagerBasedRlEnv`,
warp float32 backend, CPU). Machine-readable artifacts:

* `reports/alignment_v1_vs_mjlab_l1.json` -- per-field first-divergent-step
  comparison of deterministic action-sequence replays (Layer 1);
* `reports/alignment/fk_paths.json` -- three-way reference-construction
  comparison: v1 converter vs official converter vs on-disk cache (Layer 2);
* `reports/alignment/traces_v1/`, `reports/alignment/traces_mjlab_l1/` --
  raw per-step traces (state, inputs, reward terms, terminations).

Historical baseline is preserved untouched: `reports/supermimic_canonical_eval*.json`
(aggregate normalized return **0.2221**, 40 clips). No reward weight, physics
parameter, termination threshold or checkpoint was changed to make numbers
match. `benchmark/core.py`, `protocol.json`, `manifest.json` are byte-identical
to the frozen protocol.

---

## 1. The 20/40-clip one-frame difference: RESOLVED (data processing, not physics)

Verified on all 40 LAFAN1 CSVs:

| rule | frame count `n` |
|---|---|
| official mjlab `csv_to_npz.MotionLoader` == canonical v1 | `len(np.arange(0, (N_in-1)/30, 0.02))` |
| SuperMimic native (`supermimic/robot.py`) | `int(floor((N_in-1)/30/0.02)) + 1` |

The two differ by exactly one frame **iff `(N_in-1) % 3 == 0`**, which holds
for exactly 20 of the 40 clips (e.g. `walk4_subject1`: 8195 vs 8196).
Official and canonical v1 agree on every clip; the difference is the
**endpoint-inclusion rule of the 30 Hz -> 50 Hz resampling**, i.e. data
processing. It is not a physics, time-boundary or rollout-length bug.

Consequence: frame counts in this refactor follow the official rule (identical
to v1). SuperMimic native inference sees its own frame count; its policy
consumes `step.reference_index` and is self-consistent under either rule.

## 2. Frame-index pairing: constant +1 offset (official semantics)

`ManagerBasedRlEnv.reset()` ends with `command_manager.compute(dt=0.0)`, which
advances `MotionCommand.time_steps` to **1**. Therefore under the official
implementation the control step `i` scores reference frame `i+1`; frame 0 is
only the initial condition the robot is teleported to. The v1 evaluator scored
frame `i` at step `i` (frames 0..T-2).

Empirically confirmed on all 6 replayed sequences: `ref_index offset: 1,
first mismatch step: 0` -- a constant offset, not drift. Both run `T-1` steps
with identical termination steps.

This is the one intentional semantic change of the refactor (raw official
semantics are shipped; the CTO-facing note flags it). Its score effect is
quantified in section 3: on zero-action sequences the total-reward delta is
~0.03 over 7 steps (the same state scored against the next frame).

## 3. Layer 1 -- deterministic action replay (2 clips x 3 sequences)

Actions (zero / feedforward / closed-loop PD) generated on the v1 engine and
**replayed byte-identically** on the official engine (`--actions-from`), so
inputs are exactly equal. Comparison after disabling RSI joint randomization
(see section 5): state at reset now matches to float32/float64 noise.

| clip / sequence | steps | total reward v1 -> official | termination | first state diff > 1e-5 |
|---|---|---|---|---|
| fallAndGetUp3 zero | 7 = 7 | 0.3348 -> 0.3642 (+0.0294) | step 6 `ee_body_pos` = | none (state) |
| walk4 zero | 7 = 7 | 0.3488 -> 0.3806 (+0.0318) | step 6 `ee_body_pos` = | none (state) |
| fallAndGetUp3 feedforward | 34 = 34 | 2.3573 -> 2.4364 (+0.0791) | matched | ~step 24-28 |
| walk4 feedforward | 39 = 39 | 2.6796 -> 2.8229 (+0.1434) | matched | ~step 25 |
| fallAndGetUp3 pd_track | 17 = 17 | 1.2141 -> 1.2661 (+0.0520) | matched | ~step 13 |
| walk4 pd_track | 30 = 30 | 1.9806 -> 2.1287 (+0.1481) | matched | ~step 14 |

Findings:

* **Zero-action sequences**: robot state (joint_pos, root pose) agrees to
  <= 1e-6 (max 7.7e-7) at every step until both terminate at the *same* step
  with the *same* term (`ee_body_pos`). Residuals are pure float32-vs-float64
  integration noise. The reward delta is dominated by the pairing offset
  (same state scored against frame k vs k+1), e.g. reward first > 1e-4 at
  step 0, max 1.05e-2 -- exactly the one-frame reference advance.
* **Feedforward / PD sequences** (identical action tapes): state agrees to
  ~1e-7 early, then sensitive dependence amplifies the float32 warp vs
  float64 C backend difference -- first > 1e-5 around step 13-25, growing to
  O(0.01-1) by the fall. Termination steps still match on all 4 sequences.
  This is backend precision drift, not a logic difference; free-fall probes
  put the raw per-step integrator difference at ~1e-8.
* **Reward term breakdown**: recorded per step per term in the traces;
  differences follow the state drift plus the pairing offset. No term changes
  sign or disappears; the 9 official terms are exactly the v1 term set with
  the same weights.
* **Terminations**: identical term sets on the terminating step in all six
  pairs. (On the separate synthetic 600-frame tipping test the *order* of
  fall gates flips under float32 drift: v1 fires `anchor_ori`, the official
  backend fires `ee_body_pos` first -- both are "fall detected".)

## 4. Layer 2 -- reference construction (FK data path)

Three-way comparison (`scripts/alignment/compare_fk_paths.py`), 2 clips:

* **on-disk npz cache == v1 converter output, bitwise (max|d| = 0.0) on every
  array** -- the cache that served the historical runs is v1's numpy/mujoco-C
  FK path (`mj_forward` + `xpos`/`cvel` + `subtree_com` correction).
* v1 converter vs official converter (`csv_to_npz.MotionLoader` resampling +
  `Scene`/`Simulation` teleport FK):

| array | max abs delta (fallAndGetUp3 / walk4) |
|---|---|
| joint_pos | 8.6e-5 / 2.5e-4 |
| joint_vel | 2.5e-3 / 8.2e-3 |
| body_pos_w | 1.4e-3 / 1.4e-3 |
| body_quat_w | 1.1e-3 / 1.2e-3 |
| body_lin_vel_w | **7.36 / 4.84** |
| body_ang_vel_w | 5.1e-2 / 5.0e-2 |

Positions/quaternions agree to ~1e-3 (float32 FK + resampler rounding).
**`body_lin_vel_w` is a convention difference, not noise**: the root body
agrees to 3.6e-4 while all 29 other bodies differ by up to 1-7.7 m/s with
*different vector norms* (so it is not a frame rotation). v1's formula
(`cvel` linear part minus `omega x (subtree_com - xpos)`) does not reproduce
mjlab's own `robot.data.body_link_lin_vel_w` for non-root bodies.

Consequence and decision: the official runtime's `motion_body_lin_vel` reward
term compares the npz arrays against `robot.data.body_link_vel_w`-family
tensors, so the npz **must** be produced by the official converter for the
score to be self-consistent. The npz cache is therefore regenerated through
`OfficialConverter` for all 40 clips (v1-convention cache retired). The
Layer-1 traces above were recorded against the v1-convention cache read by
*both* engines (same input); the velocity-term caveat is noted there.

## 5. Root cause found and fixed: RSI joint randomization

The official `MotionCommandCfg.joint_position_range` defaults to
`(-0.52, 0.52)` rad; `play=True` only zeroes `pose_range`/`velocity_range`.
The reset therefore added random joint offsets the frozen protocol forbids
(observable as a 0.087 rad step-0 state offset on all 29 joints). The glue now
sets `joint_position_range = (0.0, 0.0)`; after the fix the reset state equals
reference frame 0 exactly and the Layer-1 state comparisons collapse to
backend noise. This was a glue-level configuration issue, not an mjlab bug.

## 6. Precision and contract notes (technical limitations)

* **float32 warp backend** (official) vs float64 mujoco-C (v1): per-step
  integrator difference ~1e-8 in free fall; chaotic amplification in contact
  rich tracking as quantified in section 3. This is inherent to the official
  stack and is *not* compensated for.
* **Action-mapping tensor quantization**: the official action term stores
  offset/scale as float32 (they differ from the frozen float64 spec by up to
  3.0e-8 / 1.7e-8). `ClipContext` continues to carry the frozen float64 values
  because the SuperMimic contract asserts 1e-8/1e-12 agreement; the residual
  effective-target error inside the env is ~3e-8 rad.
  `evaluate_clip` asserts live-vs-frozen agreement to 1e-6 (conformance gate).
* **`auto_reset=False`** keeps terminal state and the official env refuses to
  step past termination (the rollout driver already stops there).
* **`nonfinite` guard** kept as glue (v1 convention): a non-finite state marks
  the clip failed even though it is not an official termination term.
* **TorchZip reader**: `torch.load` is tried first with fallback to the
  pure-python reader so legacy/minimal `.pt` zips still load now that torch is
  a transitive dependency of mjlab.

## 7. Score impact statement

* Historical result **0.2221** (40 clips, 32 failed) remains the recorded
  baseline; nothing in this refactor rewrites it.
* The refactored evaluator scores raw official semantics. Expected deltas vs
  the baseline when the same SuperMimic checkpoint is re-run: pairing offset
  (+1 frame), official-FK reference arrays (incl. the body_lin_vel_w
  convention change), float32 backend drift. Each mechanism is quantified
  above; none is compensated by weight/physics tuning.
* The frozen gates (`target_normalized_return=0.82`, `target_failed_clips=0`,
  40 clips, pinned dependency versions, implementation id
  `canonical-mjlab-g1-v1`) are untouched in `benchmark/core.py`/`protocol.json`.

## SuperMimic native inference parity（2026-11-11）

上游 `zita-ch/supermimic_vibed @ 8bfc1158` + checkpoint `ckpt_015000.pt`
（SHA256 `93bc5c43…3955e7`），clip `fallAndGetUp3_subject1`，300 个控制步：

| 对比项 | max\|delta\| | 说明 |
|---|---|---|
| reference p / R / q / qd | 1.9e-5 / 2.0e-3 / 1.2e-4 / 3.3e-3 | numpy 重建 vs 上游 MotionLib（float32 级） |
| root_v_local / root_w_local | 9.2e-4 / 5.0e-2 | 同上（角速度为姿态参数化敏感项） |
| goal（730 维，13 帧含尾部 clamp） | 3.1e-2 @f=50 | 尾部 clamp 边界帧 |
| action fp32（300 步） | 6.7e-6 | numpy 推理 vs torch forward，单步逐位一致量级 |
| action bf16 autocast | 6.7e-6 | 同上 |
| action bf16 weights | 5.3e-2 | GPU 实际历史运行采用 bf16 权重；numpy 前向不复刻权重量化（如实记录） |

闭环对比（同一物理，分别由两种实现驱动）：

| 驱动方 | normalized_return | 终止 |
|---|---|---|
| numpy（本仓库推理） | 0.0875 | ee_body_pos @1102/5108 |
| torch fp32 | 0.0879 | ee_body_pos @1232/5108 |
| torch bf16 | 0.0997 | ee_body_pos @1243/5108 |

结论：单步推理等价（fp32 6.7e-6）；闭环差异由混沌放大与终止时刻敏感性产生
（alive 步数差 ~130 步即分值差 ~1e-2 量级），numpy 推理保真度足以代表
SuperMimic 策略本身，历史 0.2221 的复现差异应主要归因于
evaluator 配对语义 + warp float32 后端，而非策略推理移植。

## 40-clip 闭环回归（2026-11-11，SuperMimic 原生策略）

同一 checkpoint `ckpt_015000.pt`（SHA256 `93bc5c43…3955e7`，与历史一致），
`reports/supermimic_canonical_eval_mjlab_reuse.json`（评估墙钟 5856s）vs
历史基线 `reports/supermimic_canonical_eval_cli.json`（630s，v1 引擎，**不动**）：

| 指标 | v1（历史） | mjlab 复用 | 差异 |
|---|---|---|---|
| aggregate normalized return | 0.2221 | **0.2489** | +0.0268 |
| failed clips | 32 | **32** | **失败集合逐 clip 一致** |
| 存活 clip | 8 个 | 同 8 个 | 无一 clip 由 ok→failed 或反之 |

- 40 个 clip 中无一变差（delta ∈ [0, +0.115]，多数 +0.01~+0.04）；
  最大 delta：`walk4_subject1` +0.1148（历史该 clip 全程存活却得 0.0000）。
- 差异全部可归因于已定性来源：官方配对语义（+1 ref_index 偏移使奖励/终止
  评估帧移一位）、warp float32 后端、官方 `body_link_*` 速度约定、
  npz 重采样端点规则。**未调任何奖励权重/物理参数**掩盖差异。
- 结论：官方复用重构在保持失败模式完全一致的前提下分数略升，
  Evaluator 语义与 v1 实质等价，可交付。

## 附：轨迹复现（原始 trace 不随 PR 分发）

上文 L1 对比所用的逐帧 trace（`reports/alignment/traces_v1/`、
`traces_mjlab_l1/`，jsonl+replay npy，约 1 MB）不随 PR 分发以保持瘦身；
本表数据已全部汇总在 `alignment_v1_vs_mjlab_l1.json` 与本文件。重新生成：

    # v1 侧（pre-refactor 引擎，git 快照 ac97315）
    python scripts/alignment/dump_traces.py --out reports/alignment/traces_v1 ...
    # mjlab 侧（--actions-from 复放相同动作序列）
    python scripts/alignment/dump_traces.py --out reports/alignment/traces_mjlab_l1 \
        --actions-from reports/alignment/traces_v1 --cache reuse
    python scripts/alignment/compare_traces.py --left ... --right ...

## 附：结果身份与已知回归（CTO 审阅项）

**结果身份。** `0.2221`（v1）与 `0.2489`（mjlab 复用）**不构成同一评估实现下的
“提升”**：二者属于不同评估实现（帧配对 i vs i+1、参考速度约定不同、
float64 vs float32 后端）。记录自本修复起携带
`evaluator.source_git_sha`（源码 SHA）与 `evaluator.motion_converter_version`
（轨迹转换管线标识）以永久区分实现身份；`implementation` 标签与正式协议
版本是否升级由 CTO 决定，Agent 未修改冻结协议。历史记录
`supermimic_canonical_eval_cli.json` 不动。

**缓存版本。** npz 缓存现内嵌 `converter_version`（当前
`official-csv-to-npz-mjlab1.6.0-1`）；缺标识/标识不匹配的旧缓存自动重建，
并有 `CacheVersionTest` 回归守护。当前 40 个缓存因缺标识已全部按新管线重建。

**耗时回归。** 官方 mjlab 评估墙钟 5856s vs v1 的 630s（≈9.3×，CPU warp
float32 后端 + 每 clip 环境构建）。已记录，本轮不做性能优化。

**测试计数说明。** 上轮 89/89 为全部 4 个测试模块（含 `test_benchmark.py`
21 项）；本轮汇报的 70 项仅覆盖 3 个模块（属选定子集，非完整回归——已纠正）。
本轮完整 `unittest discover`：**95/95**（89 项原有全部保留 + 净增 6 项：
2 项官方配对语义测试 + 4 项缓存失效测试）。
