# Benchmark V2 — 源码级对照（三项实现，固定 Commit）

本文件是 `docs/BENCHMARK_V2_EVALUATION.md` 的证据附件。所有行号针对**指定 Commit 的
原始文件**，不是当前 `main`，也不是任何本地修改过的副本。

| 代号 | 来源 | 固定 Commit | 本地校验位置 |
|---|---|---|---|
| SM | `zita-ch/supermimic_vibed` | `8bfc1158bd968f729fcc65d5e897a05b05ea3d36` | `git fetch --depth=1 origin <sha>`（本轮拉取） |
| NM | `sailab-ai/NablaMimic` | `bfbb2a6773b4776c17402a34a93ec7589d9c7c18` | `~/Downloads/NablaMimic`，`git status -- src/nablamimic/` 干净 |
| V1 | `basicv8vc/gptfiles` PR #12 + #13 | `66a6e52664195fee3c446924980ae586bc55c25d`（PR #13 HEAD，堆叠于 #12 `f5bf4e0d…`） | 本仓库 `research/modded-supermimic/` |
| mjlab | `mujocolab/mjlab` 1.6.0 | `0fb8a681136be94ffc636a3dd423cabb97d91f10` | PyPI wheel 源码，V1 `constants.py:22-25` 已冻结该 pin |

V1 的官方组件行号同时给出 mjlab 1.6.0 wheel 中的文件（前缀 `mjlab/`）。

---

## 1. 逐项对照表

| 项目 | SuperMimic (SM) | NablaMimic (NM) | 当前 Benchmark V1 |
|---|---|---|---|
| **Reward terms** | 8 项：6 个 tracking + `action_rate` + `joint_limit`。`supermimic/mdp.py:108-145`（`r_track` 的 6 项在 `:125-132`，两个惩罚在 `:133-135`）。**无自碰撞项** | 8 项，同名同义：`src/nablamimic/env.py:404-423`（`err` 6 项 `:410-417`，`terms` `:418`，两惩罚 `:420-422`）；权重表 `env.py:58-59` 恰好 8 个 key。**无自碰撞项** | **9 项**：`benchmark/evaluator/constants.py:279-289`；多出 `self_collisions: -10.0`（`:288`）。配置源自 `mjlab/tasks/tracking/tracking_env_cfg.py:209-251`（`:246-250` 是自碰撞项） |
| **Reward weights / std** | `supermimic/mdp.py:40-51`（`RewardCfg`）：`body_pos 0.3 / body_ori 0.4 / body_linvel 1.0 / body_angvel 3.14 / root_pos 0.3 / root_ori 0.4`；6 项权重 `1,1,1,1,0.5,0.5`（`:125-132`）；`w_action_rate=-0.1`、`w_joint_limit=-2.0`（`:47-48`）；`soft_limit_factor=0.9`（`:49`，配合 `robot.py:144`）。每项乘 `CONTROL_DT`（`:135`，`motion.py:14` = 0.02） | `src/nablamimic/env.py:57`（`STD`，数值与 SM 完全相同）、`:58-59`（`W`，8 个权重与 SM 完全相同，`joint_limit=-2.0`）；每项乘 `DT`（`:418-422`，`:48` = 0.02）；软限位 `env.py:164-166`（`mid ± 0.5*range*0.9`） | std/6 个正权重与两者一致，但 **`joint_limit=-10.0`**（`constants.py:287`，源自 `mjlab/tracking_env_cfg.py:241-245`），且多 `-10.0` 自碰撞。`MAX_STEP_REWARD` 定义 `constants.py:293`（= `5*0.02=0.1`，与 SM `mdp.py:36`、NM `env.py:425-426` 相同） |
| **Termination** | `supermimic/mdp.py:147-152`：`\|z_ref - z_robot\| > 0.25` 或 `\|rR[2,2] - R[2,2]\| > 0.8`，**只有这两条**；阈值 `mdp.py:50-51`。触发后记账 `env.py:205-209, 220-222`（`failed_eval \|= term & ~done_eval`） | `src/nablamimic/env.py:428-433`：同样的 z / tilt 两条 **加上** `bad = ~isfinite(qpos) \| ~isfinite(qvel)`（`:432-433`）。阈值 `env.py:71-72`。失败原因编码 `evaluate.py:67-69, 94`（height / tilt / both / nonfinite） | **4 项 + glue**：`constants.py:299-301`；`benchmark/evaluator/env.py:174-187` 收集除 `time_out` 外所有项并追加 `nonfinite`（`:180-181`）。官方项为 `mjlab/tasks/tracking/tracking_env_cfg.py:257-279`：`anchor_pos`(z, torso)、`anchor_ori`(torso)、**`ee_body_pos`**（4 个踝/腕，阈值 0.25）——实现 `mjlab/tasks/tracking/mdp/terminations.py:27-53, 73-86`。V1 的 `ee_body_pos` 是两份原始实现都没有的 |
| **Normalized return** | `supermimic/evaluate.py:84`：`norm = ret / ((clip_len-1) * MAX_STEP_REWARD)`；`:81` 用 `alive` 掩码，`:83` 让已结束 clip 继续 stepping 但不再计分；`:86-87` 报告 `norm.mean()` 与帧加权版，**无 `[0,1]` 截断** | `src/nablamimic/evaluate.py:84, 90`：`full_len = lengths-1`，`full_norm_return = (tsum.sum(1) / (full_len * max_reward)).mean()`；`:60` `where(am, tv, 0)`；`:70` `clip_ended` 只关 `alive`；`:114` 每 clip `return` 同式，**无 `[0,1]` 截断** | 分母同式：`benchmark/evaluator/runner.py:126, 157-158`；但 **`normalized_return = min(max(raw,0),1)`**（`:164`）逐 clip 截断；`benchmark/core.py:169-170` 进一步强制 `[0,1]` 才接受记录。截断是原始评估不存在的额外评分修正 |
| **Failed clips** | `supermimic/evaluate.py:88-89` 读 `env.failed_eval`；`env.py:221` 只在 `term & ~done_eval` 时置位，**clip 自然结束不算失败**（`env.py:208-209` 的 `timeout` 仅进 `done_eval`） | `src/nablamimic/evaluate.py:67-70, 92-94`：`newfall = alive & term`，`failed_frac=(fall_t>0).mean()`；`clip_ended`（`env.py:462-463`）只关 `alive` | `runner.py:149-155, 165`（`failed = fall_step is not None`），`env.py:174-182` 排除 `time_out`。**语义一致**，但触发项集合不同（多了 `ee_body_pos`） |
| **Reference motion** | `supermimic/motion.py:18-87`：30→50 Hz 线性/slerp 重采样，`:31` `n = floor(dur/0.02)+1`；`:46-66` 用自建 torch FK 生成 14 个 body 的 pos/rot/lin/ang vel；body 清单 `robot.py:52-58`，**注释明确 anchor/root = pelvis**；`evaluate.py:28, 47` 从 `data/lafan1_g1/*.csv` 读全部 clip。**不做 floor alignment** | `src/nablamimic/motion.py:77-87, 101-141`：`:79` `np.arange(0, (n-1)/fps_in, 1/fps_out)`；`:111-117` **floor alignment**（脚底最低点 5% 分位归零）；`:123` `mj_differentiatePos` 求 qvel；`:138-141` `mj_objectVelocity` 取 body-origin 世界速度；`env.py:41-46` 同为 14 bodies | `benchmark/evaluator/motion.py:110-201`：复用 mjlab `scripts/csv_to_npz.MotionLoader`（`mjlab/scripts/csv_to_npz.py:71-73` 的 `torch.arange(0, duration, output_dt)`，与 NM 同式）+ Scene/Simulation 遥测 FK；`constants.py:29-31` 冻结数据集 revision。**不做 floor alignment**（`mjlab/scripts/csv_to_npz.py:56` 的 z 偏移是注释掉的）。40 clip CSV 已实测对齐，NM 的 floor shift ≤ 7.2e-08 m（见 `docs/NABLAMIMIC_EVAL_AUDIT.md` 的二次复测） |
| **Frame alignment** | `supermimic/evaluate.py:63` `reset(ids, motion.clip_start)` = 每个 clip 的**首帧**；`env.py:204-206` 第一个 reward/termination 打分在 `f1 = f+1`；`evaluate.py:83` 把 `f` 夹在 `clip_end-1`；总分步数 `clip_len-1` | `src/nablamimic/evaluate.py:44`（`frames=starts`）、`:144-146`（`steps = lengths.max()-1`）；`env.py:375` `frame = min(frame+1, last)`；第一次 reward 在 `evaluate.py:52` 的 `s2`，即 `start+1`；`:70` `clip_ended` 结束 | `benchmark/evaluator/env.py:25-30, 128-131` 说明 mjlab `reset()` 把 `time_steps` 推到 1，首帧仅作初值、第一个打分帧为 1；`runner.py:126` `n_steps = num_frames-1`。**与两份原始实现的「首帧起步、frame+1 打分」一致**；PR #12 里描述的 `+1` 偏移是重构前 v1 打分 0..T-2 与官方实现的差异，不是与原始实现的差异。PR #13 的 `8e2ee4b` 时间索引修复只涉及策略观测的 anchor 帧（`benchmark/policies/nablamimic.py:527-533`），不改奖励帧 |
| **Root / anchor semantics** | **pelvis**：`mdp.py:113-118` 用 `m.root_p/root_R`（pelvis 的 CSV 根，`motion.py:26, 48-49`）做 re-anchoring（`dR = yaw_mat(R @ rR^T)`，`anchor=(p[:2], rp[2])`）；`mdp.py:123-124` 的 root_pos/root_ori 误差也用 pelvis。终止同样用 pelvis（`mdp.py:149-151`） | **reward 用 torso_link，termination 用 pelvis**：`env.py:47` `ANCHOR="torso_link"`、`:168` `anchor_k`；`env.py:389-402` 的 re-anchoring 绕 torso（`ra_p/ra_q` = 机器人 torso，`ma_p/ma_q` = 参考 torso，`d_ori=yaw_quat(ra_q*ma_q^{-1})`）；`:411-412` root_pos/root_ori 用 torso。**但** `:428-433` 终止用 `qpos[2]` 与 `qpos[3:7]`（freejoint = pelvis），`evaluate.py:56-57` 的同名诊断也是 pelvis | mjlab 官方 `anchor_body_name="torso_link"`（`mjlab/tasks/tracking/config/g1/env_cfgs.py:34`）；`mjlab/tasks/tracking/mdp/commands.py:157-165, 200-205, 385-407` 的 `anchor_pos_w`/`robot_anchor_pos_w` 与 `body_pos_relative_w` 都绕 torso；**终止项 `mjlab/.../terminations.py:27-53` 也用 torso**。V1 `constants.py:81` `ANCHOR_BODY="torso_link"`。因此 V1 与 NM 的 reward anchor 相同，与 SM 不同；V1 的终止 anchor 与两份原始实现都不同 |
| **Action / control** | `supermimic/env.py:198-202`：`target = default_q + xi_offset + alpha*a`，`alpha = 0.25*effort/kp`（`robot.py:88-90, 133-146`），4 个物理子步、50 Hz（`env.py:40-45`）；`mdp.py:133` action-rate 为 `\|\|a-a_prev\|\|²`（原始动作差） | `src/nablamimic/env.py:320-353`（`ctrl_of`/`action_base`/`action_change`），`robot.py:103-109` 放宽 ctrlrange；`evaluate.py:139-141` 用 `action_rate_space="setpoint"`、按 run 配置决定 `residual_action`。`env.py:332-345` 明确：setpoint 空间的 action-rate 等价于「同一组 setpoint 的绝对动作」之差；绝对动作配方（`configs/reported/ours_absolute.args`）下就是 `a_t-a_{t-1}` | V1 策略契约是绝对动作 `q0 + action_scale*a`（`docs/POLICY_INTERFACE.md`、`benchmark/evaluator/constants.py:228-229`）；`mjlab/envs/mdp/rewards.py:56-62` 的 `action_rate_l2` 也是 `\|\|a-a_prev\|\|²`。PR #13 adapter 把残差动作重写成等价绝对动作 `a + (ref[k]-q0)/scale`（`benchmark/policies/nablamimic.py:527-563`），因此 **V2 复用官方 `action_rate_l2` 即可同时对齐 SM 的绝对动作与 NM 的 setpoint 语义** |
| **Physics / contact** | `supermimic/env.py:40-45`（Isaac Lab 3 + Newton/`MJWarpSolverCfg`，`njmax=500`、`nconmax=128`、pyramidal、implicitfast、4 子步）、`:107-118`（eval 摩擦 1.0）；**`robot.py:141` `enabled_self_collisions=False`** —— 原始 SM 评估没有自碰撞物理 | `src/nablamimic/robot.py:8-16, 60-100`：mjlab 匹配的 Menagerie G1 + 7 段脚底胶囊 + **显式全自碰撞 pair**（`:90-99`，排除父子与 4 对）；`env.py:123-136` PD/armature；`evaluate.py:139-143` MuJoCo Warp，10 次 Newton 迭代 | `benchmark/evaluator/constants.py:38-42, 236-251`、`spec.py:1-15`、`protocol.json:6`：mjlab 1.6.0 `FULL_COLLISION`，脚摩擦 0.6 / 地面 1.0，MuJoCo Warp 3.11.0。**两份原始实现的物理后端与自碰撞设置确实不同**，V2 无法同时复现两者数值 |

---

## 2. 已知差异的重新核实（结论：全部属实）

| 已知差异 | 核实结果 | 证据 |
|---|---|---|
| 两份原始实现都没有 `ee_body_pos` termination | ✅ 属实 | SM `mdp.py:147-152` 只有 height/tilt；`env.py:205-209` 只有 `term\|timeout`。NM `env.py:428-433` 只有 height/tilt/nonfinite；`evaluate.py:69, 94` 的失败原因只有 height/tilt/both/nonfinite。V1 有：`constants.py:98-104, 301` + `mjlab/tracking_env_cfg.py:271-278` |
| 两份原始实现都没有 self-collision reward penalty | ✅ 属实 | SM `mdp.py:125-135` 只出现 8 项；全仓 `grep -rn "penalty\|self_collision" supermimic/` 唯一命中是 `robot.py:141` 的**物理开关**。NM `env.py:58-59` 的 `W` 只有 8 个 key，`:418-422` 只构造这 8 项；`robot.py` 的自碰撞是**物理 pair**，不是奖励项。V1 有：`constants.py:288` |
| 原始 joint-limit penalty = `-2.0`，当前 mjlab = `-10.0` | ✅ 属实 | SM `mdp.py:48`；NM `env.py:59`；V1 `constants.py:287`（源自 `mjlab/tracking_env_cfg.py:241-245`） |
| 原始评估均使用完整 clip、从首帧开始 | ✅ 属实 | SM `evaluate.py:3-4, 63`；NM `evaluate.py:3, 44, 145`。V1 也如此（`runner.py:126`、`env.py:87` `sampling_mode="start"`）——**这一项 V1 本来就是对齐的** |
| 原始归一化分母为完整 clip 理论最大奖励 | ✅ 属实 | SM `evaluate.py:84` + `mdp.py:36`；NM `evaluate.py:84, 90` + `env.py:425-426`；三者都是 `(T-1) * 0.1`。V1 `runner.py:157`、`constants.py:293` 也对齐——**这一项 V1 本来也是对齐的** |
| 当前 Benchmark 额外对每条 normalized return 截断 `[0,1]` | ✅ 属实 | `benchmark/evaluator/runner.py:164`（`min(max(raw,0),1)`）与 `benchmark/core.py:169-170`（>1 或负数直接判 INVALID）。SM `evaluate.py:84-90`、NM `evaluate.py:90, 114` 均无截断，且两者都允许负值（惩罚可以超过正奖励） |

**结论**：V1 与原始评估的偏差集中在本轮要修的 4 个点上——
`self_collisions` 奖励项、`joint_limit` 权重、`ee_body_pos` 终止项、逐 clip `[0,1]` 截断，
以及一个此前未被明确列出的**终止项 anchor 身体部位**（V1/mjlab 用 torso，两份原始实现用 pelvis）。
分母、首帧起步、失败 clip 定义、6 个 tracking 权重/std 本来就是对齐的，V2 不改它们。

---

## 3. 本轮新发现的实质差异（不能假装相同）

### 3.1 Reward 的 root / anchor 身体部位不同（最重要）

- SM：`mdp.py:113-124` 的 re-anchoring 与 root 误差都绕 **pelvis**（`robot.py:52-53` 注释）。
- NM：`env.py:47, 389-402, 411-412` 都绕 **torso_link**。
- mjlab 官方 term（V1 复用的）也绕 **torso_link**（`mjlab/.../env_cfgs.py:34`、`commands.py:385-407`）。

三者的 re-anchoring **数学形式相同**（机器人 anchor 的 xy + 参考 anchor 的 z；`yaw(robot_anchor ⊗ ref_anchor^{-1})`；参考 body 相对参考 anchor 平移旋转），
但 **anchor 身体部位不同**，因此 `body_pos`、`body_ori`、`root_pos`、`root_ori` 四项在机器人偏离参考时数值不同。
这不是浮点误差，是定义的差异。**V2 采用 NM≡mjlab 的 torso anchor**（理由见评估文档 §3），并明确记录 SM 的 pelvis 变体无法同时复现。

### 3.2 姿态误差的三种数值写法（数学等价）

| 实现 | 写法 | 位置 |
|---|---|---|
| SM | `acos(clamp((tr-1)/2, -1+1e-6, 1-1e-6))²` | `supermimic/rotations.py:94-98` |
| NM | `(2*atan2(\|\|v\|\|, \|w\|))²`，`e = q1 ⊗ q2^{-1}` | `src/nablamimic/env.py:101-105` |
| mjlab | `quat_error_magnitude(q1,q2)² = \|\|log(q1⊗conj(q2))\|\|²` | `mjlab/utils/lab_api/math.py:590-604, 688-699` |

三者都是**测地角平方**。仅在接近 π 时因 clamp/eps 不同有极小数值差异；`tests/test_benchmark_v2.py` 用多组随机 quaternions 做了数值一致性测试。

### 3.3 速度误差的坐标系约定

- SM：`kinematics.py:82` 递推 link-origin 世界线速度 `v_i = v_parent + w_parent × (p_i - p_parent)`；`mdp.py:121-122` 直接与国际参考 `body_v/body_w` 相减（世界系，不做 yaw 旋转）。
- NM：`motion.py:138-141` 用 `mj_objectVelocity` 取 body-origin 世界速度；机器人侧 `env.py:383-387` 把 `cvel`（subtree-COM 空间速度）换算到 body origin。
- mjlab：`commands.py:192-197` 的 `body_link_lin_vel_w/ang_vel_w`；`rewards.py:80-113` 同样在世界系直接相减。

**误差坐标系一致（世界系、不旋转）**，但 body-origin 线速度的换算路径不同；PR #12 的 `reports/alignment_v1_vs_mjlab.md` 曾测到重构前 v1 自建 FK 与官方 converter 的非根 body 线速度差 1–7.7 m/s。
V2 沿用 mjlab 官方 converter 与官方 reward term，不引入新 FK，此差异作为「统一 canonical G1 物理/运动学带来的残差」记录。

### 3.4 自碰撞物理本身不一致

SM `robot.py:141` 关闭自碰撞；NM `robot.py:90-99` 开启全自碰撞；V1/V2 canonical G1 开启全自碰撞。
**V2 保留全自碰撞物理，但删除评分里的自碰撞惩罚**——这两件事必须区分：物理保留是任务定义，惩罚删除是评分定义。

### 3.5 Nonfinite

只有 NM 显式写了 nonfinite 终止（`env.py:432-433`）；SM 没有显式 guard；V1 用 glue 补了（`env.py:180-181`）。
任务书把 nonfinite 列为 V2 共性语义，V2 保留它（数值安全下限），并在文档中注明它来自 NM 与 V1 glue，不是 SM 的显式条款。

### 3.6 重采样与 floor alignment

- 帧数公式：SM `motion.py:31` `floor(dur/dt)+1`；NM `motion.py:79` / mjlab `csv_to_npz.py:71-73` `arange(0, dur, dt)`。
  PR #12 曾报告「20/40 clip 差一帧」；`docs/NABLAMIMIC_EVAL_AUDIT.md` 二次复测用真实 CSV 得到 **0/40 差帧**，20/40 只是 `(N-1)%3==0` 的计数，残差来自 float32 时间网格（≤6.2e-04 rad）。
- floor alignment：NM `motion.py:111-117` 做 5% 分位脚底归零；SM / mjlab 不做。实测 40 clip 的偏移 ≤ 7.2e-08 m（数据本身已对齐）。

### 3.7 物理后端

SM = Isaac Lab 3 / Newton + MJWarpSolver；NM = MJX / patched MuJoCo Warp / NablaEngine；V1/V2 = mjlab 1.6.0 + MuJoCo Warp 3.11.0。
三者的接触模型、求解器迭代、锥、积分器、足底几何都不同。**V2 不声称能同时复现两套物理后端的全部数值。**

### 3.8 动作空间

NM 的正式对比配方是绝对动作（`configs/reported/ours_absolute.args`），推荐配方与 8 分钟 checkpoint 用残差动作（`configs/ours.args`），其评估把 action-rate 放在 setpoint 空间（`env.py:332-353`, `evaluate.py:139-141`）。
SM 用绝对动作（`env.py:198`）。V2 的评估契约是绝对动作；PR #13 的 adapter 已把残差动作重写为等价绝对动作，所以 `action_rate_l2` 语义可统一。若未来有算法直接输出非绝对语义动作，需自行保证等价映射。

---

## 4. 数学定义一致性（V2 复用的官方组件）

| V2 项 | 官方 mjlab 函数 | 官方定义 | 与 SM/NM 的一致性 |
|---|---|---|---|
| root_pos | `motion_global_anchor_position_error_exp` `mjlab/.../rewards.py:26-33` | `exp(-\|\|anchor_pos_w - robot_anchor_pos_w\|\|²/std²)` | = NM `env.py:411`；SM 用 pelvis（`mdp.py:123`），不同身体 |
| root_ori | `rewards.py:36-41` | `exp(-quat_error_magnitude²/std²)` | = NM `env.py:412`；SM 用 pelvis（`mdp.py:124`） |
| body_pos | `rewards.py:44-59` | `exp(-mean_b\|\|body_pos_relative_w - robot_body_pos_w\|\|²/std²)`；re-anchoring 见 `commands.py:385-407` | = NM `env.py:399, 413`；SM `mdp.py:115-119` 形式同、anchor 不同 |
| body_ori | `rewards.py:62-77` | `exp(-mean_b angle²/std²)` | = NM `env.py:400, 414`；SM `mdp.py:118, 120` 形式同 |
| body_lin_vel | `rewards.py:80-95` | `exp(-mean_b\|\|ref_lin_w - robot_lin_w\|\|²/std²)`，世界系 | = SM `mdp.py:121` / NM `env.py:415`（都不旋转） |
| body_ang_vel | `rewards.py:98-113` | 同上 | = SM `mdp.py:122` / NM `env.py:416` |
| action_rate | `mjlab/envs/mdp/rewards.py:56-62` | `Σ(a_t-a_{t-1})²` | = SM `mdp.py:133` / NM `env.py:347`（绝对动作） |
| joint_limit | `mjlab/envs/mdp/rewards.py:82-95` | `Σ[relu(soft_lo-q)+relu(q-soft_hi)]` | = SM `mdp.py:134` / NM `env.py:421-422`；软限位 `mjlab/entity/entity.py:745-760` 的 `mid ± 0.5*range*0.9` = SM `mdp.py:88-89` / NM `env.py:164-166` |
| `scale_rewards_by_dt` | `mjlab/.../tracking_env_cfg.py:302-313` 的 `decimation=4` + manager | 每步 `dt * Σ(weight*value)` | = SM `mdp.py:135` / NM `env.py:418-422` |

**结论**：除 root/anchor 的身体部位外，V2 只需做两处**配置级**修改（去掉 `self_collisions`、把 `joint_limit` 权重改回 `-2.0`）即可让 8 个 reward term 的数学定义与 NM 完全一致、与 SM 除 anchor 外完全一致。
Termination 无法用官方组件表达（官方用 torso，原始用 pelvis），因此需要最小 Glue。
