# Checkpoint audit — 2026-11-09

目标：为 canonical evaluator 找到**已训练完成、接口兼容**的 G1 checkpoint，优先复用、不重训。

结论先行：**没有找到兼容的 checkpoint。** 找到的唯一真实权重是 SuperMimic 的
IsaacLab 基线，它的策略接口（10 帧历史 + 目标向量）与 canonical evaluator 的
mjlab 单帧 160 维观测不是同一个 MDP 观测设计，无法在不改动语义的前提下装载。
因此本轮**没有执行任何正式评估**，也没有产生可提交成绩。

## 1. 盘点范围与结果

搜索了 `~/Downloads`、`~/Documents`、`~/Desktop`、`~/nm_e2e_work`、`~/.wandb`
（不存在）、Google Drive 挂载点，以及既有 run 目录下的 manifest / summary。

| 来源 | 训练引擎 | 权重位置 | 可用性 |
| --- | --- | --- | --- |
| SuperMimic 基线（`colab_baseline`） | IsaacLab @ `46d6cc3d…` | `~/Downloads/gptfiles/research/supermimic/colab_baseline/ckpt_015000.pt` | **本地存在，68 MB** |
| NablaMimic `ea_s0_20261106T142202Z` | NablaEngine (JAX/MJX) | Drive `MyDrive/supermimic/nablamimic/runs/ea_s0_20261106T142202Z` | 本地无；Drive 不可读（见 §5） |
| NablaMimic `ea_verify_s0_20261107T085809Z` | 同上 | 同上（run 目录） | 同上 |
| NablaMimic `ea_s1_20261107T143843Z` | 同上 | 同上（run 目录） | 同上 |
| NablaMimic `eb_mjw_s0_20261106T040000Z` | MuJoCo Warp Adjoint | 无 | manifest 明确 `"checkpoints (VM destroyed)"`，且 `evaluation never ran` |
| mjlab（canonical） | mjlab @ `0fb8a68…` | — | **无任何已训练权重** |

补充：`find` 覆盖 `*.pt/*.ckpt/*.pth/*.safetensors/*.npz/orbax-checkpoint/params.msgpack`，
除上表外只命中我自己的 smoke fixture 与无关的 CV 权重（`bisenet/79999_iter.pth`）。

一个容易误判的点：NablaMimic 的 `eb_mjw` run 用了 MuJoCo Warp，听起来最接近 mjlab，
但它是 `etaoxing/mujoco_warp` 的分支 + 自研 adjoint 后端，**不是 mjlab**，且权重已随 VM 销毁。

## 2. 找到的 checkpoint：身份与结构

**文件**：`ckpt_015000.pt`（71,782,213 B）
**SHA256**：`93bc5c43185d21f4ca9be7cde526d313c0cf71fcee4ef46c26ff6e29c23955e7`

**来源**（`BASELINE.md`，同一目录）：
- 算法实现：`zita-ch/supermimic_vibed` @ `8bfc1158bd968f729fcc65d5e897a05b05ea3d36`
- 环境：**IsaacLab** @ `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`（kit-less 原生安装，非 Docker）
- 训练：15000 iters、`num_envs 4096`、seed 0，Colab G4（RTX PRO 6000 Blackwell 96 GB），1.45 h
- 数据：`/content/supermimic_vibed/data/lafan1_g1`（同一批 40 条 LAFAN1 G1 重定向动作）

**网络结构**（从权重张量直接读出，共 46 个张量）：

| 分支 | 形状 | 说明 |
| --- | --- | --- |
| `pi.enc` | 730 → 2048 → 1024 → 512 → 256 → **32** | 目标编码器，输出 32 维隐变量 |
| `pi.dec` | **1022** → 2048 → 1024 → 512 → 256 → **29** | 动作解码器 |
| `wm.net` | 133 → 2048 → 2048 → 2048 → **70** | 世界模型（`wm_only_iters=500` 预热） |
| `pi.obs_norm` | mean/var 各 **99** | 本体观测归一化 |
| `pi.goal_norm` | mean/var 各 **730** | 目标归一化 |
| `wm.in_norm` / `wm.out_norm` | 133 / 70 | 世界模型输入输出归一化 |

`1022 = 32 (z) + 990 = 32 + 10 × 99`，即解码器吃 **32 维隐变量 + 10 帧 × 99 维观测历史**；
`730 = 10 帧 × 73` 是未来 10 帧的目标特征。这与论文口径一致：本体观测每帧
`29 关节位置偏差 + 29 关节速度 + 3 角速度 + 3 投影重力 + 29 上一步动作 = 93`，
实现里为 99（多 6 维），策略看最近 10 帧，目标来自未来 10 帧。

**动作**：29 维，缩放后作为低增益 PD 的关节设定点（非力矩）。这一点与 canonical 一致。

**机器人**：IsaacLab 的 Unitree G1 29-DoF；与 canonical 用的 mjlab `g1.xml`
同为 31 body / 29 DoF，但由 PhysX 而非 MuJoCo 编译。

## 3. 兼容性判定：不兼容，且不是打包问题

canonical evaluator 的 actor 观测是 **单帧 160 维**（mjlab `tracking_env_cfg.py` 的
`actor_terms` 八项之和）：command 58（参考关节位置 29 + 速度 29）+ anchor 位置 3 +
anchor 姿态 6 + base_lin_vel 3 + base_ang_vel 3 + 关节位置 29 + 关节速度 29 + 上一动作 29。

| 维度项 | SuperMimic 权重 | canonical evaluator | 结论 |
| --- | --- | --- | --- |
| 单帧本体观测 | 99 | — | 定义不同 |
| 观测历史 | 10 帧 × 99 = 990（进解码器） | **无历史** | 结构性差异 |
| 参考/目标 | 独立 goal 向量 730（进编码器） | 并入观测的 command 58 | 结构性差异 |
| 策略输入总维 | 730 + 990 | 160 | 不可互喂 |
| 动作维 | 29 | 29 | ✅ 唯一一致项 |

机器校验（`scripts/read_torch_ckpt.py` 直接读 zip，无需 torch）：
- 权重中**不存在**任何 `actor` / `critic` 键；
- 权重中**不存在**任何形状含 160 的张量；
- `pi.dec.8.weight` 输出 29（动作维一致）。

用 canonical 加载器实测装载，被明确拒绝：

```
PolicyFormatError: no actor MLP found in checkpoint:
expected <prefix>.0.weight / .0.bias, found top-level keys ['pi', 'wm']
```

（本轮顺带把 `.pt` 读取改成不依赖 torch：`benchmark/evaluator/torchzip.py`，
因此上面的拒绝是**结构性**判定，而不是"本机没装 torch"。）

所以它既不能当作 `MlpActorPolicy`（要求 160→512→256→128→29）装载，也不是把
99 维切片成 160 维就能解决——它的输入里根本不含 mjlab 的 command/anchor/base_lin_vel，
反而多出 10 帧历史和 goal。**强行拼接会得到一个语义错误、分数无意义的策略**，
这正是本次不执行评估的原因。

同源证据：NablaMimic 的 `src/nablamimic/env.py::policy_obs` 同样是
`(history (HIST*93,), goal)` 结构（`hist.reshape(-1)`），也是历史+目标接口。
两个方法在接口层面都与 mjlab 单帧观测不同，这是它们各自的观测设计，不是打包差异。

## 4. 与「方法原有 evaluate.py」的对照

**无法完成逐 clip 对照**，因为同一 checkpoint 无法在 canonical evaluator 中运行。
可报告的是方法自报口径与其测量条件：

| 项 | SuperMimic 自报 |
| --- | --- |
| 文件 | `ckpt_015000_eval.json` |
| 归一化回报 | **0.84165**（`norm_return_framew` 0.84421） |
| 失败 clip | **0 / 40** |
| 逐 clip | 40 条齐全，区间 0.7564（`fallAndGetUp1_subject5`）～ 0.9301（`walk4_subject1`） |
| 测量环境 | IsaacLab/PhysX，非 mjlab |

这些数字**不能**与 canonical 的 `0.82` 闸门直接比较：引擎不同（PhysX vs MuJoCo）、
观测接口不同、奖励实现由各自代码决定。按既定约束，它们也不得登记为 canonical 成绩。
因此「实际数值差异」在本轮是**不可测**，而不是「测出来差很多」——我不会为此编造数字。

## 5. 若要继续，需要满足的前置条件

1. **Drive 访问**。NablaMimic 的权重按 manifest 记录在
   `MyDrive/supermimic/nablamimic/runs/<run_id>/`。当前进程无法读取：
   `~/Library/CloudStorage/GoogleDrive-10kpapers@gmail.com` 存在，但进入目录报
   `Operation not permitted`（macOS 隐私授权），子目录报 `No such file or directory`
   （未同步/未物化）。获取方式：在**系统设置 → 隐私与安全性 → 完全磁盘访问权限**中
   允许当前终端/WorkBuddy，或直接在 Drive 网页端把对应 run 目录下载到
   `~/Downloads` 后告知路径。**不需要在聊天里提供任何密钥。**
   注意：即使拿到，NablaMimic 的接口同样是历史+目标，仍不兼容（§3）。
2. **真正可行的一步**：需要一个**在 mjlab G1 tracking 配置下训练**的 checkpoint
   （obs 160 / action 29 / rsl_rl actor，或其等价导出）。这需要一次新训练，
   属于本次明确排除的动作（不重训、不启 seed 2/3），需要单独批准。
3. 若希望保留旧权重而仍要出分，唯一诚实的路径是**为每个方法各建一个与其原环境一致的
   评估器**，但那就是另一个 benchmark，且与「独立、统一、mjlab 口径」的协议目标冲突。

## 6. 本轮实际产出

- `scripts/read_torch_ckpt.py`：不依赖 torch 的 torch-zip checkpoint 读取器（本次审计工具，
  也可用于后续把 rsl_rl `.pt` 转成 canonical `.npz`）。
- 本文件：上述全部事实与判定。
- 未执行评估、未修改协议、未修改门槛、未填写任何非本机硬件信息。
