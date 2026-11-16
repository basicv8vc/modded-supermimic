# Benchmark V2 评估协议说明

`modded-supermimic-g1-lafan1-v2`

- Protocol JSON：`benchmark/v2/protocol_v2.json`
- Protocol SHA256：`e5a5c0335fcc239edae4a085eb66b9ac5b93d4c44917c7168d21761c12675ba0`
- Evaluator implementation：`canonical-mjlab-g1-v2-original-aligned`
- Evaluator source tree：`research/modded-supermimic/benchmark/v2`
- 状态：`G4_BASELINE_EVALUATION_COMPLETE` —— 目标硬件 RTX PRO 6000 Blackwell（`sm_120`）验收已通过
  （[`reports/v2/gpu_runtime_status.json`](../reports/v2/gpu_runtime_status.json) `acceptance_status=PASS`），
  官方 V2 正式 G4 基线现已确立为 **官方 `HybridRobotics/supermimic`**（`SuperMimic-GenCoord-G1` 纯传感器 `proprio`, `history=10`, **`0.84631`**, `1/40` 失败，[`reports/v2/g4_official_supermimic_seed0.json`](../reports/v2/g4_official_supermimic_seed0.json)）与 **NablaMimic**（`0.82817`, `1/40` 失败，[`reports/v2/g4_nablamimic_seed0.json`](../reports/v2/g4_nablamimic_seed0.json)）；早期的非官方分支 `SuperMimic-vibed`（修复前 `0.52869` / 修复后 A100 `0.56999`）已从 V2 正式榜单中移出并仅作历史诊断归档；
  详见第 §10 节及 [`docs/V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md`](./V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md)。
  下方 §§1–9 保留为协议设计、L4/A100 预检与验收机制的历史原始记录。
- 证据附件：`docs/V2_SOURCE_COMPARISON.md`（逐项对照表，含源码路径与行号）、
  `reports/v2/gpu_runtime_status.json`（G4 目标硬件验收，`PASS`）、
  `docs/V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md`（G4 正式 40-clip 基线与 P0/P1 诊断归因）、
  `reports/v2/preflight_l4_sm89_2026-11-12.json`（L4 预检，**非**验收产物）、
  `reports/v2/gpu_performance.json`、
  `reports/v2/smoke_gpu*.json`（均 `submission_eligible=false`，
  `report_scope=preflight_non_target_hardware`）。

对照使用的固定源码：

| 实现 | Commit |
|---|---|
| `zita-ch/supermimic_vibed` | `8bfc1158bd968f729fcc65d5e897a05b05ea3d36` |
| `sailab-ai/NablaMimic` | `bfbb2a6773b4776c17402a34a93ec7589d9c7c18` |
| `mujocolab/mjlab` 1.6.0 | `0fb8a681136be94ffc636a3dd423cabb97d91f10` |

本分支基于 PR #13 的 HEAD（`66a6e52664195fee3c446924980ae586bc55c25d`）建立。
PR #12（HEAD `f5bf4e0d881837b84ecb0654c88d7d0d3bfc361e`）与 PR #13 均为 OPEN Draft，
且 #13 的 base 就是 #12 的分支 `research/mjlab-reuse-refactor`。本 PR **没有**修改、
合并或 rebase 这两个 PR。**V1 完整保留**：`benchmark/protocol.json`（SHA256
`ddec7124dbee4c999986f995a8c2aa88097a494e1a0f4df00f2967366c15935b`）、
`benchmark/evaluator/*`、`benchmark/core.py`、`reports/*.json` 一个字节都没改。

---

## 1. 为什么 V1 与原始 SuperMimic / NablaMimic 评估不同？

V1 把 **mjlab 官方 tracking 任务的默认训练配置**直接当作评估 rubric，而
SuperMimic / NablaMimic 的评估有各自的、更窄的定义。源码核实后，V1 的偏差有 5 处：

| # | 差异 | V1 | 两份原始实现 | 证据 |
|---|---|---|---|---|
| 1 | 自碰撞奖励惩罚 | `self_collisions: -10.0`（第 9 个 reward 项） | **都没有**自碰撞 reward 项 | `benchmark/evaluator/constants.py:288`；SM `mdp.py:125-135`；NM `env.py:58-59, 418-422` |
| 2 | joint-limit 权重 | `-10.0` | 两者都是 `-2.0` | `constants.py:287`；SM `mdp.py:48`；NM `env.py:59` |
| 3 | `ee_body_pos` 终止 | 4 个踝/腕 body 的 z 误差 > 0.25 m 即失败 | **两者都没有** | `constants.py:301` + `mjlab/.../terminations.py:73-86`；SM `mdp.py:147-152`；NM `env.py:428-433` |
| 4 | 逐 clip `[0,1]` 截断 | `min(max(raw,0),1)`，且 `core.audit_record` 直接拒绝越界记录 | 两者都允许负值 / >1 | `benchmark/evaluator/runner.py:164`、`benchmark/core.py:169-170`；SM `evaluate.py:84`、NM `evaluate.py:90, 114` |
| 5 | 终止项的 anchor 身体 | mjlab 的 `anchor_pos` / `anchor_ori` 读 **torso_link** | 两者的终止都读 **pelvis** free joint | `mjlab/.../terminations.py:27-53`；SM `mdp.py:149-151`；NM `env.py:428-433` |

第 5 条是上一轮没被明确列出的偏差。V1 的 **reward** anchor（torso）与 NM 一致，
但 **termination** anchor 与两份原始实现都不一致。

同时必须说清楚：**V1 本来就已经对齐的部分**，V2 不动它 ——
6 个 tracking 权重/std、`dt=0.02`、`MAX_STEP_REWARD=0.1`、从首帧起步、
`n_steps = T-1` 的完整 clip、full-clip 分母、失败 clip 定义、
帧索引（PR #13 的时间索引修复只影响策略观测，不影响打分帧）。

一句话：V1 的分数度量的是「某个未经对齐的默认 mjlab 任务配置」下的表现，
而不是两份原始论文/代码所定义的 Motion Tracking 评估。

---

## 2. V2 采用了哪些共同语义？

V2 采用两个原始实现**共同**的评估语义（若两者不同，见 §3，不假装相同）：

### Reward（8 项，`scale_rewards_by_dt`）

| 项 | weight | std | 数学定义 |
|---|---|---|---|
| `motion_global_root_pos` | 0.5 | 0.3 | `exp(-‖p_ref_anchor − p_robot_anchor‖² / std²)` |
| `motion_global_root_ori` | 0.5 | 0.4 | `exp(-angle(q_ref,q_rob)² / std²)` |
| `motion_body_pos` | 1.0 | 0.3 | `exp(-mean_b ‖p_ref_reanchored − p_robot‖² / std²)` |
| `motion_body_ori` | 1.0 | 0.4 | `exp(-mean_b angle² / std²)` |
| `motion_body_lin_vel` | 1.0 | 1.0 | `exp(-mean_b ‖v_ref − v_robot‖² / std²)`，世界系 |
| `motion_body_ang_vel` | 1.0 | 3.14 | `exp(-mean_b ‖w_ref − w_robot‖² / std²)`，世界系 |
| `action_rate_l2` | −0.1 | — | `Σ(a_t − a_{t−1})²` |
| `joint_limit` | −2.0 | — | `Σ[relu(soft_lo − q) + relu(q − soft_hi)]`，软限位 `mid ± 0.5*range*0.9` |

每步 `r_t = dt * Σ weight_i * value_i`，`dt = 0.02`，理论上限 `0.1`。
误差坐标系、姿态误差（测地角平方）、软限位定义、re-anchoring 形式
（`d_pos = [robot_anchor.xy, ref_anchor.z]` + `yaw(robot_anchor ⊗ ref_anchor⁻¹)`）
都与原始实现一致；逐项源码行号见 `docs/V2_SOURCE_COMPARISON.md` §4。

### Termination

- 仅 `|z_pelvis(robot) − z_pelvis(ref)| > 0.25` 与
  `|g_b_z(robot_pelvis) − g_b_z(ref_pelvis)| > 0.8`，边界严格 `>`；
  其中 `g_b_z(q) = [R(q)ᵀ(0,0,−1)]_z = −R[2,2]`。
- `nonfinite`（qpos/qvel 出现 NaN/Inf）判失败；这是 NM `env.py:432-433` 的显式条款
  （SM 没有显式 guard，V1 用 glue 补过），作为数值安全下限保留。
- **不启用** `ee_body_pos`；腕/踝偏差再大也不直接判失败。
- 正常到达 clip 末尾是 **success**，不计入 failed clips。

### Self-collision

严格区分两件事：

1. **物理**：canonical G1 的 `FULL_COLLISION` 接触集与 `self_collision` ContactSensor
   全部保留（`cfg.scene` 一个字段都没改），身体之间仍会发生真实接触。
2. **评分**：V2 **不添加** self-collision reward penalty —— 两份原始评估都没有。

### Normalization

```
normalized_return_clip = Σ_{t=0}^{T-2} r_t * alive_t / ((T-1) * 0.1)
```

- 分母恒为**完整 clip** 的理论上限，不因提前终止而缩短；
- 终止步的奖励**计入**（三套实现都在 `alive` 更新前掩码），其后各步严格为 0；
- 负值、>1 的值**原样保留**，不做 `[0,1]` clip，也不引入其他评分修正；
- 总分为 40 条 clip 的非加权平均；`failed_clips` 为提前终止的 clip 数。

### 版本身份

V2 有独立 Protocol ID / JSON / SHA256、独立 evaluator implementation 字符串、
独立 `reports/v2/registry.json`（随 G4 实测完成后登记正式记录）与独立文档。
`benchmark/v2/` 之外的 benchmark 架构没有重构。

---

## 3. 哪些语义仍然不同？

**不能声称 V2 同时复现了两套评估的全部数值。** 已核实的实质差异：

| 差异 | SM | NM | V2 | 处理 |
|---|---|---|---|---|
| Reward 的 root/anchor 身体 | **pelvis** | **torso_link** | **torso_link**（≡ mjlab 官方 term） | 选 NM≡mjlab，明确记录 SM 变体无法同时复现 |
| 自碰撞物理 | 关闭（`robot.py:141`） | 开启（`robot.py:90-99`） | 开启（canonical G1） | 保留；物理差异不可消除 |
| 物理后端 | Isaac Lab 3 / Newton + MJWarpSolver | MJX / patched MuJoCo Warp / NablaEngine | mjlab 1.6.0 + MuJoCo Warp 3.11.0 | 不复制仿真器，不声称数值等价 |
| 求解器/接触/足底几何 | 4 子步、`nconmax=128`、pyramidal、implicitfast | 7 段脚底胶囊、10 次 Newton 迭代 | mjlab 默认（10 次迭代、脚摩擦 0.6 / 地面 1.0） | 都属于统一物理模型带来的残差 |
| floor alignment | 不做 | 5% 分位脚底归零 | mjlab converter 不做 | 40 clip 实测偏移 ≤ 7.2e-08 m，可忽略但非零 |
| 重采样帧数 | `floor(dur/dt)+1` | `arange(0, dur, dt)` | = NM / mjlab | 真实 CSV 实测 0/40 差帧，float32 时间网格残差 ≤ 6.2e-04 rad |
| 动作空间 | 绝对动作 | 绝对 / 残差两套配方，评估在 setpoint 空间算 action-rate | 绝对动作契约 | PR #13 adapter 已把残差动作改写为等价绝对动作 |
| nonfinite guard | 无显式条款 | 有 | 有 | 采用 NM，属于「两者严格交集之外」的一项 |
| 姿态误差数值写法 | `acos(clamp)` | `2*atan2` | mjlab `quat_error_magnitude` | 三者都是测地角平方，仅在近 π 处有极小数值差 |

**anchor 分歧是最重要的、无法调和的一条**：SM 的 4 个 pose/root 项绕 pelvis，
NM 与 mjlab 绕 torso_link。三者的 re-anchoring 数学形式完全相同，但身体不同，
所以在机器人偏离参考时数值不同。这不是浮点误差。
`tests/test_benchmark_v2.py` 专门构造了 `coincident_anchor`（让 torso 与 pelvis 重合）
来证明：当两个 anchor 是同一 body 时，V2 与 SM **逐项完全一致**；
在一般 `drift` 状态下差异约 `4.05e-4`（单步 reward），并被测试断言为
「真实存在且已记录」，而不是被掩盖。

V2 选择 torso_link 的理由：
1. NM 与 mjlab 官方 term 已经一致，无需任何 glue 即可复用官方实现；
2. 任务书要求「优先复用 mjlab 官方 Reward Term」「不要复制或重新实现仿真器」，
   pelvis anchor 会要求重写 6 个官方 reward term，违反最小实现原则；
3. V2 的协议 JSON 与本文档都把这个选择标注为**未解决的差异**，而非「已对齐」。

其余已核实但被判定为**等价**的点：权重/std/dt/软限位、误差坐标系、
姿态误差的三种写法、速度误差的世界系约定、首帧起步与打分帧、
full-clip 分母、失败 clip 语义。

---

## 4. V2 是否忠实实现了共同的 Reward 和 Termination？

**Reward：对 NM 是逐项忠实；对 SM 除 anchor 身体外忠实。**

- V2 直接复用 mjlab 1.6.0 的 8 个官方 reward term（函数名、weight、std 都由
  `tests/test_benchmark_v2.py::V2ConfigTests` 静态断言），只做两处配置级修改：
  删除 `self_collisions`、`joint_limit` 权重由 −10 改回 −2。
- 用**真实源码**作为 oracle 的逐项 parity（`scripts/v2/verify_v2.py`）：

  | 用例 | SuperMimic | NablaMimic | V2 | 最大逐项 \|NM−V2\| |
  |---|---|---|---|---|
  | `perfect` | 0.059057691 | 0.059058066 | 0.059058066 | 1.23e-09 |
  | `coincident_anchor` | 0.041988020 | 0.041988026 | 0.041988020 | 1.23e-09 |
  | `drift` | 0.041972816 | 0.042377915 | 0.042377919 | 2.19e-09 |

  NM 用 float32、V2 用 float64，误差量级即 float32 舍入；`drift` 下 SM 与 V2 的
  0.000405 差值来自 anchor 身体不同，不是实现 bug。
- `action_rate_l2` 与 `joint_limit` 在**所有**三个用例下与 SM 的真实
  `mdp.reward()` 输出逐项一致（10 位小数）。
- 额外做了数学一致性测试：四元数测地角平方 vs `acos(trace)` 形式（200 组随机）、
  `yaw_quat` vs 投影 x 轴、`g_b_z` vs `−R[2,2]`、软限位 `mid ± 0.5*range*0.9`、
  re-anchoring 的 xy 平移 / 纯 yaw 不变性、完美跟踪恰好等于 `0.1` 上限。

**Termination：忠实，且是用最小 glue 实现的。**

mjlab 官方 `bad_anchor_pos_z_only` / `bad_anchor_ori` 读 torso_link，
**无法表达**两份原始实现的 pelvis 语义，因此 V2 增加了 3 个最小 term
（`bad_root_height` / `bad_root_tilt` / `bad_nonfinite`，`benchmark/v2/config.py`），
只改 termination，不动 physics。`GlueMathConsistencyTests` 断言 glue 与
`benchmark/v2/spec.py` 的可执行定义在阈值两侧完全一致。

真实源码 termination parity：

| 用例 | SM | NM | V2 |
|---|---|---|---|
| 正常 / 高度 0.249 / 高度正好 0.250 / tilt 0.6 / 腕踝偏差 0.9 m | False | False | False |
| 高度 0.251 / −0.251 / tilt 1.7 | True | True | True |
| nonfinite | False（SM 无 guard） | True | True |
| clip 正常结束 | 不算失败 | 不算失败 | 不算失败 |

V2 没有 `ee_body_pos`；腕/踝大偏差但 pelvis 正常时不判失败（V1 会判失败）。

**Normalization parity**：固定合成 reward 序列覆盖了 full-clip 分母、
上限恰好为 1、提前终止后剩余归零、终止步奖励计入、负值 −0.5 保留、
>1 的 1.5 保留、V1 clamp 不被应用、非加权平均、T−1 步、以及 50 组随机序列
对照 SM `evaluate.py:84` / NM `evaluate.py:90` 的原始公式。全部通过。

---

## 5. 哪些差异来自统一 canonical G1 物理模型？

任务书要求继续使用统一的 canonical G1 29-DoF 模型并保留 mjlab 官方环境、
PD/Actuator、MuJoCo Warp 与完整自碰撞。因此以下差异**不是评估协议的偏差，
而是统一物理模型的必然结果**，V2 不也无法消除：

1. **自碰撞物理**：SM 关闭、NM 开启；V2 保留 canonical G1 的全自碰撞。
   注意：删除的是 *reward penalty*，不是 *contact*。二者必须严格区分。
2. **求解器 / 接触模型 / 积分器 / 锥 / 迭代次数**：三套后端各不相同。
3. **脚底几何**：NM 用 7 段胶囊 + 显式自碰撞 pair；canonical G1 用 mjlab
   `FULL_COLLISION` 与脚摩擦 0.6 / 地面 1.0。
4. **body-origin 世界线速度的换算路径**：SM 自建 torch FK 递推；NM 用
   `mj_objectVelocity` + `cvel` 换算；mjlab 用 `body_link_lin_vel_w`。
   误差坐标系一致（世界系、不旋转），但数值路径不同。PR #12 的
   `reports/alignment_v1_vs_mjlab.md` 曾测得重构前自建 FK 与官方 converter 的
   非根 body 线速度差 1–7.7 m/s。V2 沿用官方 converter，不引入新 FK。
5. **PD / Actuator / action scale / joint order / contact configuration**：
   全部沿用 PR #12 的 `benchmark/evaluator/`，未修改。
6. **Reference re-anchoring / motion resampling / floor alignment**：
   V2 沿用 mjlab 官方 `csv_to_npz` converter；NM 额外做 floor alignment，
   实测 40 clip 偏移 ≤ 7.2e-08 m（数据本身已对齐）。
7. **帧索引**：`reset()` 后 `time_steps=1`，首帧仅作初值，第一个打分帧为 1；
   V2 `n_steps = num_frames − 1`，与两份原始实现一致。PR #13 的时间索引修复
   只作用于策略观测的 `anchor_*_prev`，V2 完整继承。

**结论**：V2 不声称能同时复现两套不同物理后端的全部数值。它统一的是
*评估定义*（Reward / Termination / Normalization），不是 *物理后端*。

---

## 6. 为什么 V1 的历史分数不能直接与 V2 比较？

1. **评分函数不同**：V1 多一个 `−10.0 × self_collisions` 惩罚项，
   `joint_limit` 权重是 −10 而非 −2，且每条 clip 的 normalized return 被强制截断到
   `[0,1]`。V2 允许负值 / >1，同一个 rollout 在两套 rubric 下会得出不同数字。
2. **失败判定不同**：V1 的 `ee_body_pos` 会在腕/踝偏差 >0.25 m 时提前终止，
   并且终止 anchor 是 torso 而非 pelvis；V2 不会。这直接改变 `alive` 掩码长度，
   进而改变分子。
3. **择优/截断效应**：V1 的 clamp 把负分抬到 0、把 >1 压到 1，
   会系统性地抬高低分 clip 并压缩高分 clip，属于单调但非线性的变换，
   无法用常数换算回 V2。
4. **身份不同**：V1 `modded-supermimic-g1-lafan1-v1` /
   `canonical-mjlab-g1-v1` / protocol SHA `ddec7124…`；
   V2 `modded-supermimic-g1-lafan1-v2` /
   `canonical-mjlab-g1-v2-original-aligned` / protocol SHA `e5a5c033…`。
   `audit_v2_record` 会显式拒绝 V1 记录（`protocol_id mismatch`），反之亦然。
5. **质量门槛不可迁移**：V1 的 `0.82` / `0-of-40 failed` 门槛是绑定 V1 rubric
   校准的。V2 的 `quality_gate.status = NOT_SET`，在真实 GPU 40-clip 运行之前
   不设门槛。

因此：`reports/*.json` 里的全部 V1 成绩只属于 V1，`reports/v2/registry.json`
当前 `records: []`。两者不得混排在同一排行榜，也不得做差值/比例比较。

---

## 7. V2 是否适合成为后续算法训练效率排行榜的统一标准？

**协议层面：适合，且明显优于 V1。评估执行层面：目前还不适合，需要补一步实测。**

支持理由：

- 评估定义来自两份已验证开源实现的共同语义，而非某个框架的默认任务配置；
- 8 个 reward term、2+1 个终止条件、归一化公式都有源码行号和可执行定义
  （`benchmark/v2/spec.py`）双重固定；
- 有独立的 protocol ID / SHA256 / evaluator identity / 记录校验器，
  记录不可与 V1 混淆；
- 真实源码 oracle 的逐项 parity 与冻结快照（`tests/v2_oracle_snapshot.json`）
  让后来者可以在纯 CPU 环境复现数学一致性；
- 40 条 clip、`dt=0.02`、首帧起步、full-clip 分母、失败 clip 定义都固定，
  物理/机器人/数据集沿用 PR #12，无自由度。

尚缺的前提：

- 没有重新训练，没有重跑 40 clips，**没有任何 V2 分数**；
- `quality_gate` 尚未校准，不能据此给出「达标 / 不达标」；
- SM 的 pelvis anchor 与 V2 的 torso anchor 差异未消除（也未被掩盖）；
- 两份原始实现的物理后端差异不可消除；
- **目标 GPU 未验证**：Colab G4 / RTX PRO 6000 / sm_120 分配连续 4 次返回
  `HTTP 503 Service Unavailable`，冻结依赖栈在 Blackwell 上的兼容性因此
  **无法确认**。已在 NVIDIA L4 / sm_89 上完成全部 CUDA 验收（§9），但不得
  把它当作目标硬件验收。

已补齐的前提（本轮新增，§9）：

- V2 正式 evaluator 已改为 **CUDA-only**，无静默 CPU fallback；
- 真实 mjlab GPU RewardManager / TerminationManager 与 V2 oracle 的
  runtime parity 全部 PASS（逐项 8 个 reward term）；
- `walk1_subject1` 与 `dance1_subject1` 两条 CUDA smoke 通过，
  均 `SMOKE_VALID` 且 `submission_eligible=false`；
- 单环境稳态 FPS、CUDA init / JIT / warmup / rollout 时间分解、
  GPU util / memory / CPU util 已记录。

建议的落地顺序：

1. 拿到 RTX PRO 6000 / sm_120（或确认冻结栈在该架构上可用的替代 Blackwell
   主机），重跑 `scripts/v2/gpu_runtime_check.py`，关闭 sm_120 BLOCKED 项；
2. 用两个既有 checkpoint 各跑一次完整 40 clips，得到 V2 基线；
3. 基于真实分布校准 `quality_gate`，并把记录路径登记到
   `reports/v2/registry.json`；
4. 之后才把 V2 作为训练效率排行榜的统一评估标准，并要求每条记录带
   `protocol_sha256` 与 `training_wall_s` / `setup_wall_s` / `eval_wall_s`。

在完成 1–3 之前，**不推荐**把 V2 当作正式 Benchmark 发布，也不推荐用它给
现有 checkpoint 重新排名。协议定义已经就绪；缺的是目标硬件确认与正式基线。

---

## 8. 本轮验证状态与复现命令

### GPU-free 部分（本机可执行）

```
# 完整 V2 测试（含真实源码 oracle / mjlab 静态 cfg）
python -m pytest tests/test_benchmark_v2.py -q
# → 78 passed

# GPU runtime 集成测试的 GPU-free 契约层
python -m pytest tests/test_v2_gpu_runtime.py -q
# → 13 passed, 12 skipped (skip reason 均以 BLOCKED: 开头，绝不冒充 PASS)

# 纯 numpy 环境（无 torch / jax / mjlab / CUDA）
/usr/bin/python3 -m unittest discover -s tests -p 'test_benchmark_v2.py'
# → Ran 78 tests, OK (skipped=19)

# 真实源码 parity 报告 + 冻结快照
python scripts/v2/verify_v2.py --write-snapshot

# V2 身份（不需要 CUDA）
python scripts/v2/run_eval.py --dry-run

# 从已提交的运行时产物重建性能报告（无需 GPU）
python scripts/v2/gpu_performance_report.py
```

### CUDA 部分（已执行，见 §9）

```
python scripts/v2/gpu_runtime_check.py --clip walk1_subject1 --steps 6
# → status PASS（6/6 检查），exit 0

python scripts/v2/run_eval.py --smoke-policy replay --clips walk1_subject1 \
    --cache-references --out reports/v2/smoke_gpu.json
# → SMOKE_VALID / SMOKE_PASS，exit 0

python scripts/v2/run_eval.py --smoke-policy replay --clips dance1_subject1 \
    --cache-references --out reports/v2/smoke_gpu_dance1.json
# → SMOKE_VALID / SMOKE_PASS，exit 0

python -m pytest tests/test_v2_gpu_runtime.py -q -rs
# → 25 passed（真实 GPU 上 Tier 2 全部 PASS，未跳过）
```

`tests/test_benchmark_v2.py` 覆盖 10 组：V1 保留、V2 身份/协议、可执行数学、
官方 mjlab cfg 静态校验、glue 数学一致性、reward parity（冻结快照 + 真实源码）、
termination parity、normalization parity、V2 rollout driver 契约、V2 记录校验。

完整仓库测试套件（本机，无 CUDA）：`169 passed, 53 skipped, 3 failed`。
3 个失败都是**本 PR 之前就存在**的环境问题，与 V2 无关（这三个测试文件
甚至不 import `benchmark.v2`）：

- `tests/test_evaluator.py::ManifestTest::test_dataset_has_40_clips`
- `tests/test_evaluator.py::ManifestTest::test_manifest_hashes_are_current`

  两者都因 `data/lafan1_g1.sha256.json` 不存在（数据集目录被 gitignore，未下载）。
- `tests/test_nablamimic_policy.py::CheckpointContractTest::test_actor_shapes_match_the_recipe`

  因该环境缺少 `optax`（反序列化 `.pkl` checkpoint 需要）。

### 未执行 / 未验证

- **RTX PRO 6000 Blackwell / sm_120 上的 CUDA 验收**（BLOCKED，Colab G4
  连续 4 次 HTTP 503，见 §9.1）；
- 40-clip V2 评估、任何 V2 分数、quality gate 校准（本任务明确禁止）；
- 两套原始物理后端与 canonical G1 之间的数值等价（按设计不可等价）。

### 文件清单（本 PR 新增 / 更新，均为 additive；V1 文件一个字节未改）

```
benchmark/v2/__init__.py            V2 包与身份常量
benchmark/v2/protocol_v2.json       独立协议 + SHA256（本轮未修改）
benchmark/v2/spec.py                GPU-free 可执行 Reward/Term/Norm 定义 + V2 记录校验
                                    （本轮新增 smoke 校验器 + 可选相对参考入口）
benchmark/v2/config.py              官方 mjlab cfg 覆盖 + 最小 pelvis termination glue
benchmark/v2/runner.py              CUDA-only V2 rollout / runtime state / provenance
benchmark/v2/runtime.py             新增：CUDA 强校验、硬件 provenance、GPU/CPU telemetry
benchmark/v2/parity.py              新增：真实 GPU Reward/Termination/Frame/Re-anchor parity
                                    + 稳态性能测量
docs/V2_SOURCE_COMPARISON.md        源码级逐项对照（路径+行号）
docs/BENCHMARK_V2_EVALUATION.md     本文档
tests/test_benchmark_v2.py          78 个 GPU-free V2 测试（本轮未改主体）
tests/test_v2_gpu_runtime.py        新增：GPU runtime 集成测试（契约层 + 真实 CUDA 层）
tests/v2_support.py                 真实源码 oracle + 合成 fixture
tests/v2_oracle_snapshot.json       冻结 oracle 快照
scripts/v2/fetch_upstream.py        拉取固定 Commit 上游源码
scripts/v2/verify_v2.py             parity 报告 / 快照生成
scripts/v2/run_eval.py              V2 评估 CLI（smoke / benchmark 双模式）
scripts/v2/gpu_runtime_check.py     新增：独立 CUDA runtime 验收（PASS/FAIL/BLOCKED
                                    + acceptance_status 三态 + 源码指纹）
scripts/v2/gpu_performance_report.py 新增：从产物重建性能报告（无需 GPU）
scripts/v2/colab_g4_runbook.sh      新增：G4 目标硬件一条命令验收（sm_120 硬门禁）
scripts/v2/mark_preflight.py        新增：给非目标硬件的旧产物补 report_scope 元数据
reports/v2/registry.json            V2 记录登记表（records 仍为空，另列 smoke 产物）
reports/v2/preflight_l4_sm89_2026-11-12.json  L4 预检报告（BLOCKED_TARGET_HARDWARE）
reports/v2/gpu_performance.json     聚合性能 / provenance 报告
reports/v2/smoke_gpu*.json         脱敏 smoke 诊断产物（submission_eligible=false）
```

---

## 9. CUDA GPU 评估实现与实机验收

本节对应第二轮任务：把 V2 的**实际物理仿真**迁到 CUDA，完成最小真实环境验收。
所有数字来自 `reports/v2/gpu_runtime_status.json`、`reports/v2/gpu_performance.json`
与 `reports/v2/smoke_gpu*.json`，这些产物由脚本直接写出，硬件信息全部来自
**运行时检测**（`torch` / `warp` / `nvidia-smi` / `mjlab.Simulation`），
不是 `--gpu-name` 之类的用户输入字符串。

### 9.1 目标硬件：BLOCKED

| 项 | 目标 | 实际 (预检 1) | 实际 (预检 2) |
|---|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell（96GB） | **NVIDIA L4** | **NVIDIA A100-SXM4-40GB** |
| Compute Capability | sm_120 | **sm_89 (`(8, 9)`)** | **sm_80 (`(8, 0)`)** |
| VRAM | 96 GB | 峰值 422 MiB | 峰值 422 MiB |
| CUDA / Driver | 支持 Blackwell 的版本 | CUDA 13.0 / driver `580.82.07` | CUDA 13.0 / driver `580.82.07` |

Colab G4 分配**连续多次**返回 `HTTP 503 Service Unavailable`（云端 Blackwell 容量不足）：

```text
ColabRequestError: Failed to issue request POST
https://colab.research.google.com/tun/m/assign?...&variant=GPU&accelerator=G4:
Service Unavailable
```

响应体是 Google 的通用 503 HTML。在 G4 缺货期间，已按照指示切换分配 **A100-SXM4-40GB (sm_80)**，并连同之前的 **L4 (sm_89)** 完成了双硬件真实 GPU 校验：
- **冻结依赖栈在 sm_120 上的兼容性未被验证**，仍如实标 `BLOCKED_TARGET_HARDWARE`，不伪造 PASS；
- 冻结依赖栈（`mujoco==3.11.0` / `mujoco-warp==3.11.0` / `warp-lang==1.16.0` / `mjlab@0fb8a68`）在 **sm_80 (A100)** 和 **sm_89 (L4)** 上均确认 100% 可用且 `version_drift = {}`；
- A100 实机产物归档为 `reports/v2/preflight_a100_sm80.json` 及 `reports/v2/smoke_gpu_a100_*.json`；
- 修复了 `measure_performance()` 在 `zero` action 物理步进时缺少 `auto_reset=False` 下的手动 reset 判定 bug。
- DoD 中「GPU = RTX PRO 6000 Blackwell」这一项**未完成**，如实标 `BLOCKED`。

### 9.2 CUDA-only evaluator（消除 CPU 评估）

`benchmark/v2/runner.py` 中原先的 `ManagerBasedRlEnv(cfg=cfg, device="cpu")`
已删除，改为：

```python
self.probe = RT.probe_cuda(device)              # 无 CUDA 直接抛异常
self.device = RT.require_cuda_device(device)    # 默认 cuda:0，无 fallback
self._env = ManagerBasedRlEnv(cfg=cfg, device=self.device)
self.runtime = RT.inspect_runtime(self._env)    # 构造后复核真实设备
```

`benchmark/v2/runtime.py` 是唯一的设备决策点，任何一项不满足即抛
`CudaRuntimeUnavailable`（评估直接失败，exit 3 / 标记 BLOCKED），**没有静默 CPU
回退**：`torch.cuda.is_available()`、Warp CUDA device、构造后 `env.device` /
`sim.device` / `sim.wp_device.is_cuda` / `sim.wp_data.qpos.device.is_cuda` /
`mujoco_warp.Data` 类型 / 捕获的 CUDA step graph。

物理、接触、PD/Actuator、Reward、Termination 全部留在 GPU；只有 policy I/O
与 provenance 通过 `.cpu().numpy()` 回到 host。V1 的
`benchmark/evaluator/env.py` 仍保持 `device="cpu"`，并由测试显式断言未被改动。

### 9.3 硬件与 kernel 执行证据

`gpu_runtime_check.py` 打印并写入报告的运行时检测行：

```text
torch.cuda.is_available              : True
torch.cuda.get_device_name(0)        : NVIDIA L4
torch.cuda.get_device_capability(0)  : (8, 9) (sm_89)
warp CUDA devices                    : ['cuda:0']
mjlab runtime device                 : cuda:0
MuJoCo Warp simulation backend       : mujoco_warp
warp device / arch                   : cuda:0 / 89
CUDA runtime / driver                : 13.0 / 580.82.07
CUDA graph step captured             : True
physics state array device           : cuda:0
target GPU RTX PRO 6000 / sm_120     : False / False
```

不只是「显存被占用」：`verify_cuda_kernel_execution()` 同步两个 CUDA runtime，
读取 `sim.wp_data.qpos` 前后值，要求有限且发生非零变化：

```json
{"kernel_execution_wall_s": 0.286, "qpos_max_abs_delta": 0.1104,
 "qpos_finite_after_step": true, "state_advanced_by_gpu_kernel": true,
 "torch_memory_allocated_bytes": 33053184}
```

`runtime` 详情：`simulation_class = mjlab.sim.sim.Simulation`、
`mujoco_warp_data_class = mujoco_warp._src.types.Data`、
`use_cuda_graph = true`、`step_graph_captured = true`、
`wp_data_qpos_device = cuda:0`、`no_cpu_fallback = true`。

### 9.4 Runtime Reward Parity（8 项逐项）

**根因（不是公式 bug，也不是靠放宽 tolerance 解决）**：mjlab
`ManagerBasedRlEnv.step()` 在第 4 步计算 reward，而 `sim.forward()` 与
`command_manager.compute()` 在第 7–8 步才执行（`manager_based_rl_env.py:395-410`
明确说明 reward/termination 看到的是「落后一个物理子步」的派生量）。
官方 `motion_relative_body_*` reward term 读的是**上一步存下的**
`MotionCommand.body_pos_relative_w` / `body_quat_relative_w`，而原 oracle 用
「reward 时刻的实时 robot anchor」重新做 re-anchoring —— 两边其实是**不同状态**，
于是 `motion_body_ori` / `motion_body_pos` 出现 3.31e-05 / 9.32e-06 的残差。

离线用 mjlab 自己的 `quat_mul` / `yaw_quat` / `quat_apply` 在**同一状态**上
对照 `spec.reference_reanchor()`，5 组随机数据最大差 `3.1e-07`（float32 舍入），
证明 re-anchoring 公式本身正确。

修法：`spec.py` 增加可选 `rel_pos` / `rel_quat` 入口
（`reward_terms_from_relative_reference()`），parity harness 在
`reward_manager.compute` 被 hook 的**同一瞬间**从 GPU 读取 MotionCommand 的
两个相对参考缓冲交给 oracle。它们是**参考状态张量、不是 RewardManager 输出**；
oracle 仍独立计算全部误差、`exp`、weight、dt。同时新增
`check_reference_reanchor_parity()` 在 GPU 上单独校验 re-anchoring 公式与
root/anchor 语义（误差 `1.17e-07`，容差 `5e-6` 仅覆盖 float32 往返，
**reward 的 `atol=2e-5 / rtol=3e-4` 没有放宽**）。

结果（`reports/v2/gpu_runtime_status.json`，`status = PASS`）：

| Reward term | max abs error |
|---|---|
| `motion_global_root_pos` | 5.99e-10 |
| `motion_global_root_ori` | 1.26e-09 |
| `motion_body_pos` | 9.49e-10 |
| `motion_body_ori` | 2.30e-09 |
| `motion_body_lin_vel` | 4.33e-10 |
| `motion_body_ang_vel` | 1.00e-09 |
| `action_rate_l2` | 1.25e-11 |
| `joint_limit` | 0.0 |

整体 `max|err| = 1.72e-09`（修复前 3.31e-05），`max rel = 9.59e-07`。
同时验证并全部为 True：8 项齐全、runtime/ oracle 权重一致
（0.5/0.5/1/1/1/1/−0.1/−2）、dt scaling 0.02、body indexing 14 个 body 顺序、
anchor 为 `torso_link` 且 index 7、软限位 factor 0.9、action history 逐步对齐、
joint soft limits shape `[1,29,2]`、`reward_manager._scale_by_dt=True`、
frame alignment。

### 9.5 Runtime Termination Parity

边界在 CUDA tensor shim 上**精确构造**（不再依赖未知 baseline 偏差）：

- 高度：显式令 `robot_root_z = ref_root_z + dh`，`dh` 取 `0.249 / 0.25 / 0.251 / −0.251`，
  严格 `> 0.25` 的边界行为被真实验证；
- tilt：令 `ref_quat = identity (g_b_z = −1)`、`robot_quat = [√(1−d/2), √(d/2),0,0]`
  使 `|Δg_b_z| = d` 精确成立，`d` 取 `0.79 / 0.80 / 0.81`；
- nonfinite（qpos 注入 NaN）、wrist/ankle 0.9 m 大偏差但 pelvis 正常、
  clip 正常结束（`clip_end_success` 不算 failure）。

全部 case 与 `spec.termination()` 逐项 match，`status = PASS`，无 mismatch。
结构检查全部为 True：`ee_body_pos` 不存在、`anchor_pos/anchor_ori` 不存在、
3 个 pelvis glue 项存在、`self_collisions` reward penalty 不存在、
**self-collision ContactSensor 仍在（完整自碰撞物理保留）**、阈值与 spec 一致、
终止读取的 `body_pos_w` / `body_quat_w` 数值上等于当前 `time_steps` 帧
（不是陈旧/未来帧）、`TERMINATION_ROOT = pelvis` 而 reward anchor 为
`torso_link`（未混淆）、`time_out` 为 success。

### 9.6 Frame alignment（PR #13 语义）

`env.reset()` 以 `command_manager.compute(dt=0.0)` 结束，
`MotionCommand.time_steps` 被推进到 **1**：frame 0 只是初始条件，
**第一个打分帧是 1**（V1 `benchmark/evaluator/env.py:25-30` 已明确记录，
是 PR #13 修复的既定行为）。原先 parity 检查断言 frame 0 才是错的。

实测 `command_frame = 1, 2, 3`，且 `policy_reference_index`、
`env.ref_index`、`time_s = step × 0.02` 全部与之一致；
`motion.time_step_total == ref.num_frames`；`control_dt` 与 oracle 一致。
`check_reward_parity` 内部也改为使用**reward 时刻捕获的帧**（而非 step 返回后
已被 +1 的计数器），消除 off-by-one。`checks` 8 项全 True。

### 9.7 smoke 与 benchmark 严格分离

| | `smoke` | `benchmark` |
|---|---|---|
| clips | 1–2 条 | 固定完整 40 条，`--limit` 禁止 |
| policy | 必须 `--smoke-policy` | 必须 checkpoint / policy-file |
| 校验器 | `audit_v2_smoke()` | 原封不动的 `audit_v2_record()` |
| `submission_eligible` | `false`（top-level + evaluation 双重标记） | `true` |
| 进入 `records[]` | 否 | 是 |
| exit code | 成功 0 / 失败 2 / BLOCKED 3 | 同上 |

`audit_v2_record()` 的 40-clip Integrity Gate **一个字节都没放宽**；
`is_submission_eligible()` 还会拒绝「padding 到 40 行的 smoke」。
smoke 记录必须携带运行时检测出的 `actual_torch_device=cuda*`、
`actual_simulation_backend`、`cuda_graph_step_captured=true`，
CLI 输入的 `--gpu-name` 只作为 `declared_gpu_name` 保存，绝不作为证据。

### 9.8 性能测量（编译开销 vs 真实 rollout）

`reports/v2/gpu_performance.json` 可由 `scripts/v2/gpu_performance_report.py`
从已提交产物重建。以下为 `gpu_runtime_check.py` 的一次完整运行（单环境）：

| 阶段 | 墙钟 (s) | 说明 |
|---|---:|---|
| CUDA init probe | 0.033 | `torch` / `warp.init()` / `get_cuda_devices()` |
| Reference loading | 0.154 | CSV→npz 缓存命中 |
| CUDA env build | 10.853 | mjlab 环境构建、模型编译、graph capture |
| 首个 CUDA kernel（JIT） | 0.287 | 同步后的冷启动 step |
| Warmup 20 步 | 0.305 | 计入下面的稳态前段，**不计入 FPS** |
| 稳态 rollout 180 步 | 2.761 | **真实 physics/contact/PD/reward/termination** |
| 整个 runtime check | 15.101 | 含以上全部一次性开销 |

**稳态 `65.19 FPS`**（180 steady steps，step 均值 15.33 ms、最大 18.38 ms；
期间 4 次 manual reset 共 0.065 s）。这个数字**排除了** CUDA init、
env build、reference load 与 kernel JIT，不是「总耗时除以步数」的虚高值。

单条 clip smoke 的实测（脚本化 replay policy，会因 `root_height` 提前终止，
低 reward / 提前终止按任务要求**不算失败**）：

| clip | 完整 clip 步数 | alive | 归一化回报 | clip 墙钟 | 稳态 FPS |
|---|---:|---:|---:|---:|---:|
| `walk1_subject1` | 13064 | 52 | +0.002747 | ~1.32 s | 68.4 |
| `dance1_subject1` | 6573 | 46 | +0.004700 | ~1.22 s | 58.0 |
| 两条合并运行 | — | 52 / 46 | +0.00372 (avg) | 3.11 s (两条合计) | 68.5 / 67.9 |

遥测（0.4 s 采样）：runtime check 33 个样本，GPU util 均值 6.8% / 峰值 29%、
GPU memory 均值 147 MiB / 峰值 378 MiB、CPU util 均值 12.7% / 峰值 33%；
smoke 中 GPU memory 峰值 422 MiB、GPU util 峰值 29%、CPU util 约 9%。
**单环境 50 Hz 下 GPU 利用率低是预期行为**（kernel 之间 GPU 大部分空闲），
这是单环境正确性/延迟测量，不是批量吞吐宣称；按任务要求未实现并行多环境。

上表是 `V2TrackingEnv.step` 的口径，其中包含 numpy↔torch 的策略 I/O 往返。
任务书 §8 要求在 GPU 单环境仍然慢时先找瓶颈，因此
`measure_performance()` 现在额外用**预分配 CUDA action** 直接调原始
`env.step`，同时报告 `sim_step_fps`（纯 physics/contact/PD/reward/termination）
与 `policy_glue_overhead_s_per_step`，并给出 `bottleneck`
（`physics` / `policy_glue`）。上表的 L4 数据早于该改动，故只含 V2 口径；
G4 复跑会同时写入两个口径，届时「慢在仿真还是慢在胶水」是直接可读的。

### 9.9 Provenance

每条 smoke 记录的 `evaluation.provenance` 均由运行时检测填充：
GPU name / compute capability / `actual_torch_device` / `actual_warp_device` /
`actual_simulation_backend` / simulation state device / CUDA graph captured /
CUDA runtime 13.0 / driver 580.82.07 / MuJoCo 3.11.0 / MuJoCo Warp 3.11.0 /
Warp 1.16.0 / mjlab 1.6.0 + commit / `evaluator_git_sha` /
`v2_protocol_sha256 = e5a5c033…` / robot model SHA256 /
dataset manifest SHA256 / motion converter version /
**实测** simulation FPS（并附 `nominal_control_fps = 50` 以示区分）/
evaluation wall time、telemetry、performance。V2 Protocol JSON 的 SHA256
在本轮前后**未变化**，V1 protocol SHA256 同样未变化。

### 9.10 测试

| 环境 | 命令 | 结果 |
|---|---|---|
| 本机（无 CUDA） | `pytest tests/test_benchmark_v2.py` | 78 passed |
| 本机（无 torch / mujoco） | `python -m unittest tests.test_v2_gpu_runtime` | 29 tests，15 passed / 14 skipped，**0 error**（skip reason 以 `BLOCKED:` 开头） |
| 本机（无 CUDA） | `pytest tests/` | 169 passed, 53 skipped, 3 failed（3 个为既有环境缺失：`data/lafan1_g1.sha256.json`、`optax`） |
| Colab L4 CUDA | `pytest tests/test_v2_gpu_runtime.py -q -rs` | **25 passed**（Tier 2 真实 GPU 测试全部执行，未跳过） |
| Colab L4 CUDA | `scripts/v2/gpu_runtime_check.py` | **PASS**，6/6 检查，exit 0 |
| Colab L4 CUDA | 3 次 `run_eval.py` smoke | **SMOKE_PASS** ×3，exit 0 |
| Colab A100 CUDA | `pytest tests/test_v2_gpu_runtime.py -q -rs` | **29 passed**（全套 29 项测试全部执行通过） |
| Colab A100 CUDA | `scripts/v2/gpu_runtime_check.py` | **PASS**，6/6 检查，稳态 59.8 FPS，exit 4（`BLOCKED_TARGET_HARDWARE`） |
| Colab A100 CUDA | 3 次 `run_eval.py` smoke | **SMOKE_PASS** ×3（`walk1`、`dance1`、双 clip），exit 0 |

Tier 1 里的 `test_cpu_fallback_is_rejected_not_silently_used` 与
`test_provenance_reports_measured_fps_and_runtime_detection` 实际分别需要
`torch` 与 `mujoco`（后者经 `manifest → evaluator.spec` 间接 import）。
它们此前在干净机器上是 **ERROR 而不是 skip**，与模块文档「GPU-free tier 恒定
可跑」的承诺矛盾，现已加 `skipUnless` 守卫。新增的契约测试覆盖：
缺少 CUDA graph 不被误判为 CPU fallback、真实 CPU 状态仍被拒绝、
`acceptance_status()` 的三态语义、`source_fingerprint()` 覆盖全部实现文件。

覆盖：CUDA runtime selection、CPU fallback rejection、CUDA kernel 真实执行、
真实 mjlab reward parity、termination parity、frame alignment、re-anchoring
公式、V2 记录身份、V1 回归保护、单 clip smoke CLI exit 0、FPS/墙钟记录。
无 CUDA 时 `gpu_runtime_check.py` 明确写 `BLOCKED` 并 exit 3，绝不标 PASS。

### 9.11 验收状态机：parity PASS ≠ 验收 PASS

这是本轮最容易出错的地方，因此做成了**机制**，不是文档约定：

| 判据 | 字段 | 含义 |
|---|---|---|
| V2 数学与真实 GPU runtime 是否一致 | `status` | `PASS` / `FAIL` |
| 是否跑在任务指定硬件上 | `target_hardware_met` | `target_gpu_match AND sm_120_confirmed` |
| 能否算作验收 | `acceptance_status` | `PASS` / `BLOCKED_TARGET_HARDWARE` / `FAIL` |

`scripts/v2/gpu_runtime_check.py` 的退出码相应为 `0`（目标硬件 PASS）、
`4`（parity PASS 但硬件不对）、`2`（FAIL）、`3`（BLOCKED）。
`run_eval.py` 写出的每条记录也会带 `report_scope`：
`acceptance` 或 `preflight_non_target_hardware`。

因此本轮 L4 产物的正确读法是：**parity PASS、验收 BLOCKED**。为避免文件名
本身造成误读，L4 的 runtime 报告已归档为
`reports/v2/preflight_l4_sm89_2026-11-12.json`（`status` 改为
`BLOCKED_TARGET_HARDWARE`，附 `_note`），`reports/v2/smoke_gpu*.json` 由
`scripts/v2/mark_preflight.py` 打上 `report_scope=preflight_non_target_hardware`
—— 该脚本只加元数据，不改分数、不改 provenance、不改硬件字段，且拒绝处理
已是目标硬件的记录。`reports/v2/gpu_runtime_status.json` 在目标 GPU 跑过之前
**故意不存在**，其缺失本身即代表验收未完成。

另一处本轮修正：`inspect_runtime()` 原先要求 `sim.step_graph is not None`，
否则判定为 CPU fallback。但 mjlab 在 Warp mempool 关闭或驱动 < 12.4 时会
**主动**禁用 CUDA graph（`sim.py:_should_use_cuda_graph`），此时物理仍在 GPU
上，硬判会把正常的 GPU 运行误杀。现在 graph 捕获状态单独记为
`graph_capture_status`（`captured` / `disabled_by_runtime` /
`enabled_but_uncaptured`）并输出 warning，CPU fallback 的判定只依赖真实设备与
状态数组。真正的「kernel 跑了」证据仍来自 `verify_cuda_kernel_execution()`。

### 9.12 复跑 G4 验收：一条命令

`scripts/v2/colab_g4_runbook.sh` 把整条链路封装好（逐步打印、任一步失败即退出、
结束时打包 `/content/v2_g4_artifacts.zip`）。

仓库是 **private**，所以没有匿名的 raw-URL 一行命令。在 Colab 选 **G4** 运行时后：

```python
import os
os.environ["GH_TOKEN"] = "github_pat_xxx"        # 只读权限即可
!bash /content/gptfiles/research/modded-supermimic/scripts/v2/colab_g4_runbook.sh
```

（脚本尚未在 `/content` 时，先用同一 token clone 到 `/content/gptfiles`；
脚本检测到 `.git` 会复用并重新 checkout `BRANCH`，不会重复克隆。）

脚本会：检查 `sm_120`（不是 G4 直接 FATAL 退出，不会退回 L4 硬跑）→ 仅在
镜像 torch 不支持 Blackwell 时安装 cu130 版 torch（torch 不属于冻结基准栈，
该步骤会打印记录）→ 安装 `mujoco==3.11.0` / `mujoco-warp==3.11.0` /
`warp-lang==1.16.0` / `mjlab@0fb8a68` 并**校验无 drift**→ 克隆分支并记录
`V2_EVALUATOR_GIT_SHA` → `fetch_lafan1.py` 下载数据并核对 manifest SHA256 →
`gpu_runtime_check.py --scope acceptance`（要求 exit 0）→ 两条 clip smoke
（要求 `SMOKE_VALID` 且 `report_scope=acceptance`）→ 汇总 FPS/瓶颈。

本轮从当前会话**无法**执行该脚本：沙箱对 `colab.research.google.com`、
`oauth2.googleapis.com`、`accounts.google.com` 均不可达（curl 返回 `000`），
`colab` CLI 的 access token 于 15:11 过期且刷新被代理拦为 `502`；此外 Hy4 侧
G4 分配此前已连续 4 次返回 `HTTP 503`。因此目标硬件的 BLOCKED 状态在本轮
**没有被解除**，需在可访问 Google 的环境执行上面的命令。

### 9.13 结论：是否具备正式 40-clip 条件？

**在已验证的 L4 / sm_89 CUDA 栈上：代码与协议层面具备。** CUDA-only evaluator、
8 项 reward runtime parity、termination boundary parity、frame alignment、
smoke/benchmark 分离、provenance 与性能记录都已通过；正式 40-clip 校验器
未被放宽，V1/V2 协议与物理/机器人/数据集均未改动。

**但在任务指定的目标硬件上：不具备，仍 BLOCKED。** RTX PRO 6000 / sm_120
未能分配，冻结依赖栈在该架构上的兼容性未验证。正式 40-clip 评估必须在
目标 GPU 上重跑 `gpu_runtime_check.py` 并确认 `sm_120_confirmed=true` 之后开展。
此外 `quality_gate` 未校准、两个 checkpoint 的 V2 基线尚未建立，
因此 `reports/v2/registry.json` 的 `records` 仍为空 —— 本 PR 没有产生、
也不会伪造任何 V2 分数。

---

**最终原则**：V2 的目标是让分数反映算法的 Motion Tracking 能力，
并且与 SuperMimic / NablaMimic 已验证的评估定义一致、统一、可复现。
本 PR 不追求让两份 checkpoint 的分数变高，也没有伪造任何 V2 成绩；
目标 GPU 未验证的事实同样如实标注，绝不以 L4 结果冒充实机验收。

---

## 10. 后续进展：G4（`sm_120`）验收通过与正式 40-clip 基线（2026-11-13 ～ 11-14）

上述 §9 记录的 `BLOCKED` 状态已于后续 Colab G4（`NVIDIA RTX PRO 6000 Blackwell Server Edition / sm_120`）实机回合中解除：

- **目标硬件验收**：[`reports/v2/gpu_runtime_status.json`](../reports/v2/gpu_runtime_status.json) 已生成，结果为 **`acceptance_status = PASS`**。
- **Option B 向量化评估**：[`benchmark/v2/batched.py`](../benchmark/v2/batched.py) 与 [`scripts/v2/fast_eval.py`](../scripts/v2/fast_eval.py) 在冻结 `protocol_v2.json`（`e5a5c033...`）不变的前提下完成 G4 40-clip 评估：
  - **SuperMimic (Official `HybridRobotics/supermimic`, `GenCoord` `proprio` `h=10`)**（`83b9417d...`，G4 从零训练 25,001 iter / `3419.35 s`）：**`0.84631`**，`1 / 40` 失败（`jumps1_subject2`，[`reports/v2/g4_official_supermimic_seed0.json`](../reports/v2/g4_official_supermimic_seed0.json)，2-clip 闭环串行-并行一致性 `PASS` 见 [`reports/v2/g4_parity_official_supermimic_2clips.json`](../reports/v2/g4_parity_official_supermimic_2clips.json)）；后续 `35,001`-iter 加密追踪 run（`official_35k_20261115T120918Z`）在 `15,000` iter / `2210.83 s` 最早达 `0/40` 失败（`0.82641`），`17,000` iter（`2453.09 s`, `0.83635`）起稳定 `0/40`，`34,500` iter（`4568.23 s`）峰值达 **`0.85751`（`0/40` 失败）**（汇总见 [`../../supermimic/colab_official_workflow/RESULTS.md`](../../supermimic/colab_official_workflow/RESULTS.md)）。
  - **NablaMimic**（`4086e7a9...`）：`0.82817`，`1 / 40` 失败（[`reports/v2/g4_nablamimic_seed0.json`](../reports/v2/g4_nablamimic_seed0.json)，适配器已使用 `frame = max(global_frames - 1, _slot_starts)` 与 `anchor_ori_b_prev`，不受帧偏移影响）
  - **SuperMimic-vibed（已移出正式榜单，仅作历史诊断归档）**：修复前 G4 为 `0.52869`（`24 / 40` 失败，[`reports/v2/g4_supermimic_seed0.json`](../reports/v2/g4_supermimic_seed0.json)），适配器 1-帧偏移修复并对齐上游源码（`max|delta| = 3.099e-06`）后在 A100 上复测为 `0.56999`（`20 / 40` 失败，[`reports/v2/a100_supermimic_seed0.json`](../reports/v2/a100_supermimic_seed0.json)）。由于非官方 fork 训练关闭自碰撞，现已在 [`reports/v2/registry.json`](../reports/v2/registry.json) 中移入 `retired_baselines`，由上述官方 `HybridRobotics/supermimic` 基线取代。详见 [`docs/V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md`](./V2_G4_BASELINE_AND_DIAGNOSTIC_REPORT.md) 与 [`reports/v2/registry.json`](../reports/v2/registry.json)。
