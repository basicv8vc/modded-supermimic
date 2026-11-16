# Canonical Evaluator v1 — `canonical-mjlab-g1-v1`

可实际运行的 G1 Motion Tracking Benchmark 评估器。本 PR 只做评估，不改 SuperMimic /
NablaMimic 的训练算法，不跑训练，不登记任何旧实验成绩。

基座保持不变：`benchmark/protocol.json`、`benchmark/core.py`、`benchmark.py` 的协议与
结果校验逻辑一行未改，评估器只是新增了「产结果的物理端」。

---

## 1. 修改文件及核心设计

新增（`benchmark/evaluator/`，约 1.3k 行）：

| 文件 | 职责 |
| --- | --- |
| `constants.py` | 所有冻结数字：commit、关节顺序、tracked bodies、执行器分组与增益推导、仿真选项、观测项、奖励表、终止阈值 |
| `spec.py` | 用官方 `g1.xml` 构建 MJCF：重写 `*_collision` 几何为 full self-collision、加 `terrain` 平面、按官方分组追加 `<position>` 执行器并写入 armature，输出结构指纹 |
| `motion.py` | LAFAN1 CSV → 50 Hz 参考轨迹（重采样 + slerp + FK 速度），逐帧前向运动学复用同一 `MjData` |
| `env.py` | 确定性闭环：复位写第 0 帧、每步 4 个子步、终止/奖励在 `mj_forward()` 之前求值以复刻 mjlab 的「一拍延迟」语义 |
| `policy.py` | 最小 Policy Adapter：`.npz`（线性层权重）与 rsl_rl `.pt`，校验 obs 维度/动作维度/关节顺序/action scale，确定性推理取均值动作 |
| `runner.py` | 逐 clip rollout、`ClipResult`、与协议兼容的记录装配、提交门禁检查 |
| `manifest.py` | 机器人模型 36 文件哈希、数据集 40 clip 哈希、manifest SHA256 |
| `protocol.py` | 读取既有 `benchmark/protocol.json`（不复制协议） |

新增脚本：`run_canonical_eval.py`（唯一入口）、`freeze_manifest.py`（`--check` 可复核）、
`export_actor_npz.py`（rsl_rl `.pt` → 规范 `.npz`）、`fetch_lafan1.py`、`diagnose_clip.py`（逐帧定位终止项）。

新增测试：`tests/test_evaluator.py`（35 个）；文档：`docs/CANONICAL_EVALUATOR.md`；
冒烟记录：`reports/smoke_replay_40clips.json`、`reports/smoke_checkpoint.json`。

核心设计取舍：

- **不重新实现一套协议**：记录结构直接满足 `benchmark/core.py` 的读取契约，验证器独立计算均值/失败数，不信任评估器自报的聚合值。
- **保真优先于简洁**：mjlab 的 manager 在 `step()` 里「先算奖励/终止、后 `mj_forward()`」，导致二者看到的是落后一个子步的运动学；本实现照抄这一顺序，并在文档 §9 明确记录，而不是「修正」它。
- **歧义显式冻结而非静默改门槛**：`docs/CANONICAL_EVALUATOR.md` §9 列出 6 项必须做选择的语义（启动扰动关闭、`n_steps = T−1`、一拍延迟、相对位姿滞后、终止步奖励仍计入、CPU 后端），阈值一个没动（0.82 / 0 failed / 0.25 / 0.8）。
- **指标不新造**：`normalized_return = Σ r_t / ((T−1) · 0.1)`，`0.1 = 6.0 × 0.02` 与官方 `max_reward` 及已发布 baseline 的定义一致；未使用任何训练端自报分数。

## 2. 冻结的 Commit、依赖版本、模型与数据 Hash

| 项 | 值 |
| --- | --- |
| mjlab | `v1.6.0` @ `0fb8a681136be94ffc636a3dd423cabb97d91f10` |
| `mujoco` / `mujoco-warp` / `warp-lang` | `3.11.0` / `3.11.0` / `1.16.0` |
| numpy / python | `2.5.4` / `3.13.12` |
| 机器人模型 | `assets/g1/g1.xml` + 35 STL，共 36 文件，`robot_model_sha256 = ae606600e1089fe79412054852092673ebbef8df881caa9c3e291b8894b0396a` |
| 数据集 | `lvhaidong/LAFAN1_Retargeting_Dataset@g1` @ `07ff8845a94d59d4195a8ac04fe918d4e6338239`，40 clips，`dataset_manifest_sha256 = 595fd98a3e75a8ed1408cec0e473db19ec7f1022cbdd52480613ce39305a5085` |
| 协议 | `protocol_sha256 = ddec7124dbee4c999986f995a8c2aa88097a494e1a0f4df00f2967366c15935b` |
| 编译后结构指纹 | `nbody 31 · njnt 30 · ngeom 69 · nq 36 · nv 35 · nu 29 · ngeom_robot 68` |

`python3 scripts/freeze_manifest.py --check` 可复现上述哈希（本机已通过）。40 个 clip 的
校验和与 NablaMimic 仓库自带的 40 个校验和逐一吻合，说明数据来源一致。

数据集与 checkpoint 未入库（`.gitignore` 已排除 `data/`、`checkpoints/`）；入库的只有
19 MB 机器人网格，因为模型指纹必须离线可复现。

## 3. 真实 Smoke Test 结果

环境：本机 CPU，`mujoco 3.11.0`，单进程。

**A. 40 clip 机制冒烟**（`--smoke-policy replay`，参考关节角直接当动作）：

```
clips            : 40
mean normalized  : 0.0020  (gate 0.82)
failed clips     : 40
mode             : smoke
validator status : INVALID
  - GPU model is not Colab G4 RTX PRO 6000 Blackwell
  - [submission gate] evaluation.mode is not 'checkpoint'
  - [submission gate] evaluation.policy_kind is not 'checkpoint_actor'
```

40 条全部加载、复位到各自第 0 帧并运行，累计 441,080 步；`eval_wall_s = 210.9`，
`setup_wall_s = 12.8`。每条在 15–42 步内终止，终止项 40/40 都是 `ee_body_pos`。

**这不是崩了，也不是评估器判错**。逐帧诊断（`scripts/diagnose_clip.py`，
`dance1_subject1`）显示躯干稳定、踝关节 1 cm 跟住，只有手腕在 0.8 s 内下垂 26 cm：

| step | 骨盆 Δz | 投影重力差 | 踝 z 误差 | 腕 z 误差 |
| --- | --- | --- | --- | --- |
| 10 | 0.018 | 0.0004 | 0.014 | 0.177 |
| 20 | 0.035 | 0.0069 | 0.014 | 0.214 |
| 39 | 0.159 | 0.0038 | 0.010 | 0.261 → `ee_body_pos` |

原因是 mjlab 的伺服本来就软：手臂 `kp = 14.25 Nm/rad`，把 0.35 m 的手臂撑住需要约
`0.5 rad` 的稳态偏差，而 PD 位置伺服没有重力前馈。已用冻结常量复核
（`0.139e-4 · (3.5556 · 4.5)² + … = 3.610e-3` ⇒ `kp = 3.610e-3 · 62.83² = 14.25`），
腿部同样吻合（`7520_14 → 40.18`、`7520_22 → 99.10`），所以这是官方模型而不是抄错参数。
训练好的策略会在动作里输出这项补偿，开环回放做不到。

因此该记录**只证明机制跑通**（40 条确定性 rollout、复位忠实、终止忠实、记录结构被验证器接受），**不证明区分能力**，`0.0020` 不可作为基线，也不能和 0.82 相比。

**B. checkpoint 加载冒烟**（2 clip，走真实 `.npz` 文件与真实 SHA256）：

```
[dance1_subject1] ret=0.0005 FAILED  alive=7/6573    ref=3.7s sim=0.1s
[walk1_subject1 ] ret=0.0002 FAILED  alive=6/13064   ref=8.9s sim=0.4s

checkpoint       : 04638e777fc11db756dd3e9746f886158e12989c3d24b21bed79accd802fdd0a
mode             : checkpoint
validator status : INVALID
  - GPU model is not Colab G4 RTX PRO 6000 Blackwell
  - checkpoints[0].per_clip must have exactly 40 entries
  - [submission gate] record covers a clip subset, not the full protocol of 40
```

这一条验证的是**加载与校验链路**：文件被真正读取、元数据被校验、推理被执行、SHA256 被记录；权重是随机初始化的 fixture，分数无意义。退出码 2 是预期行为（记录应当被拒）。

**C. 测试**：`tests.test_benchmark` 21/21 通过（原有用例未改）；
`tests.test_evaluator` 35/35 通过（212 s），覆盖结构指纹、full self-collision、足部接触参数、
增益与 action scale、参考轨迹帧数与速度、复位等价于第 0 帧、观测 160 维各段偏移、奖励分母、
近似参考位姿时六项约等于 1、确定性复跑、失败语义、长时接触稳定、策略元数据拒绝、子集记录被验证器拒绝。

## 4. 尚未解决的阻塞

1. **没有可用的已训练 checkpoint**。mjlab 用 W&B artifact 分发 G1 tracking 权重（`evaluate.py`
   直接 `wandb:...` 拉取），需要鉴权；本机镜像也不可达。因此**没有产出任何正式成绩**，
   也没有把任何旧实验分数登记为 canonical 结果。加载器本身已就绪并测过。
2. **rollout 跑在 CPU 后端**。`mujoco-warp`/`warp-lang` 已按协议版本安装并写入记录，但尚未接进
   rollout，所以 `hardware.gpu_name` 这一项在本机无法满足（验证器会拒），实测 0.99 ms/step。
3. **尚未与官方 `evaluate.py` 做同 checkpoint 对齐跑**。这是端到端保真度的最后一道验证，
   依赖第 1 项。

## 5. 是否具备执行正式 40-Clip Benchmark 的条件

**不具备，差的正好是「一个真实 checkpoint」这一件外部输入。** 拆开说：

- 已具备：模型/数据/协议全部冻结且可复核；确定性 rollout 能跑完整片（`dance1_subject1`
  为 13,654 步，harness 不做步数截断，只在真实终止时提前退出，且终止后步数本就按 0 计入）；
  40 条逐 clip 结果能产出；验证器能拒绝缺失 clip、重复 clip、不一致 checkpoint、错误版本、
  无效成绩（这五类在 `tests/test_benchmark.py` 与 `tests/test_evaluator.py` 里各有对应用例）；
  一条命令即可跑；测试与冒烟记录齐备。
- 还差：把训练机上导出的 checkpoint 放进来跑一次 40 clip（`export_actor_npz.py` →
  `run_canonical_eval.py`），以及接入 `mujoco-warp` 后端以满足 GPU 字段。

按当前实测，拿到 checkpoint 后跑一次 40 clip 约需 7 分钟仿真 + 3 分钟参考轨迹构建，成本很低。

**已停止**：不启动训练、不跑 seed 2/3、不碰训练算法、不消耗 GPU。

---

### 复现命令

```bash
python3 scripts/freeze_manifest.py --check
python3 -m unittest discover -s tests -v

# 40-clip 机制冒烟（无需 checkpoint）
python3 scripts/run_canonical_eval.py --smoke-policy replay --out reports/smoke.json

# 正式 40-clip（需要真实 checkpoint）
python3 scripts/run_canonical_eval.py \
    --checkpoint checkpoints/g1_tracking_model_30000.npz \
    --algorithm SuperMimic --training-commit <40-hex> --seed 0 \
    --gpu-name "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --training-wall-s 5220 --out reports/canonical_eval.json
```

退出码：0 = 记录已产出且自洽；2 = 产出但会被 `benchmark/core.py` 拒绝（冒烟即属此类，预期如此）。
