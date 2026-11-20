<h1 align="center">modded-supermimic</h1>

<p align="center">
  <b>Unitree G1 全身动作跟踪竞速基准。</b><br>
  在单卡 Colab G4（RTX PRO 6000）上从零训练可部署的 29-DoF 策略，比拼谁能最快在全部 40 条 LAFAN1 动作上达到 ≥ 0.82 归一化回报且 0 / 40 失败。
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

<p align="center">
  <a href="https://github.com/basicv8vc/modded-supermimic/actions/workflows/ci.yml"><img src="https://github.com/basicv8vc/modded-supermimic/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI 状态"></a>
  <a href="benchmark/v2/protocol_v2.json"><img src="https://img.shields.io/badge/protocol-FULL__COLLISION-2f6feb" alt="FULL_COLLISION 协议"></a>
  <img src="https://img.shields.io/badge/hardware-1%C3%97%20Colab%20G4%20(sm__120)-lightgrey" alt="单卡 Colab G4">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0%20assets-green" alt="MIT 与 Apache-2.0 资产"></a>
</p>

<p align="center">
  <img src="assets/g1-flash.png" alt="G1 闪电侠 —— 更快的 Unitree G1 训练竞速" width="160"><br>
  <sub><i>更快的 Unitree G1 训练 —— 从零起步，以竞速节奏攻克全部 40 条动作。</i></sub>
</p>

灵感来自 [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)：**固定任务、硬件与独立裁判；完全开放训练算法、框架与仿真器。**

---

## 世界纪录榜

所有成绩均在 **Google Colab RTX PRO 6000** 上，由冻结的独立裁判按[评测协议](benchmark/v2/protocol_v2.json)评测：

- **物理配置**：开启 FULL_COLLISION 全自碰撞物理
- **裁判运行时**：mjlab 1.6.0 · mujoco 3.11.0 · mujoco-warp 3.11.0 · warp-lang 1.16.0
- **回报计算**：不做 [0, 1] 截断
- **碰撞惩罚**：无自碰撞奖励惩罚

| #     | 纪录                    | 达标时间（0/40 且 ≥0.82） | 训练框架 / 仿真器                                                            | 归一化回报  | 失败数     | 论文 / 仓库                                                                                            | 证据与卡片                                                                                                                                                   |
|:-----:|:----------------------- |:-------------------------:|:---------------------------------------------------------------------------- |:-----------:|:----------:|:------------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1** | **InstantTrack v1.4**   | **6m 45s**                | PyTorch + mjlab / MuJoCo Warp<br><sub>InstantTrack v1.4.0 + BlinkMimic</sub> | **0.82169** | **0 / 40** | [BlinkMimic](https://arxiv.org/abs/2611.16422) · [Repo](https://github.com/Huy-Quang-Dao/InstantTrack) | [纪录卡片](records/v2/003_instanttrack_v14/README.md) · [评测 JSON](reports/v2/instanttrack_v14_s0/ckpt_002300_v2.json)                                      |
| **2** | **NablaMimic**          | **9m 32s**                | JAX + NablaEngine 0.1.0                                                      | **0.84108** | **0 / 40** | [∇Mimic](https://arxiv.org/abs/2610.41654) · [Repo](https://github.com/sailab-ai/NablaMimic)           | [纪录卡片](records/v2/002_nablamimic_nablaengine/README.md) · [评测 JSON](reports/v2/nablamimic_25m_bisect/v2_it_001500.json)                                |
| **3** | **SuperMimic Official** | **36m 51s**               | PyTorch + mjlab<br><sub>HybridRobotics/supermimic</sub>                      | **0.82641** | **0 / 40** | [SuperMimic](https://arxiv.org/abs/2610.28480) · [Repo](https://github.com/HybridRobotics/supermimic)  | [纪录卡片](records/v2/001_supermimic_official_mjlab/README.md) · [二分报告](baselines/supermimic_mjlab/out/official_35k_20261115T120918Z/bisect_report.json) |

- **#1 InstantTrack v1.4**（总榜最快 & mjlab 赛道最快）：**6m 45s**（含首轮编译 405.23 s，iter 2,300；扣除 201.07 s 首次编译后约 3m 24s / 204.16 s；15m 终点回报 0.84308）— 合入 [BlinkMimic](https://arxiv.org/abs/2611.16422) 并对齐 FULL_COLLISION，含编译对比官方 HybridRobotics/supermimic 快约 **5.5×**（扣编译快约 **10.8×**），比 NablaMimic 快约 **1.4×**。
- **#2 NablaMimic**（NablaEngine / JAX 赛道最快）：**9m 32s**（572.18 s，iter 1,500；25m 终点回报 0.86896）— [∇Mimic](https://arxiv.org/abs/2610.41654)。
- **#3 SuperMimic Official**（官方基线）：**36m 51s**（2,210.83 s，iter 15,000；稳定 0 / 40 达标时间为 40m 53s / 2,453.09 s，回报 0.83635；峰值回报 0.85751）— HybridRobotics/supermimic。
- **机读注册表**：[reports/v2/registry.json](reports/v2/registry.json)（由 CI 自动审计）。

---

## 规则

1. **锁定硬件**：1× Google Colab G4（NVIDIA RTX PRO 6000，96 GB 显存）。
2. **统一门槛（同一 Checkpoint）**：单个 checkpoint 跑完全部 40 条标准 LAFAN1 动作，必须**同时**满足：
   - **全片平均归一化回报 ≥ 0.82**（按完整片段长度计算的未截断无权均值）。
   - **失败 Clip 数 = 0 / 40**（在 T - 1 步前不得因 root_height < 0.25 m、root_tilt > 0.8 rad 或非有限数值而提前终止）。
3. **仅计训练耗时（Training-Only Wall Time）**：从训练 step 0 到产出达标 checkpoint 的实际训练壁钟时间，不含数据集下载、环境配置、一次性 JIT / CUDA-graph 编译及事后评测耗时。
4. **从零训练与可部署控制契约**：
   - 权重必须仅在 40 条标准 LAFAN1 动作上从零训练（禁止加载外部预训练权重）。
   - 评测时，冻结裁判仅提供 [StepInput](docs/POLICY_INTERFACE.md)（joint_pos、joint_vel、base_lin_vel、base_ang_vel、anchor_pos_b / anchor_pos_b_prev、anchor_ori_b / anchor_ori_b_prev、previous_action 与只读 ReferenceClip），并接收 50 Hz 的 (29,) 维关节指令（target = action * action_scale + default_joint_pos）。
5. **开放训练栈、冻结裁判端**：
   - 允许自由选择训练框架、观测编码、历史 / 前瞻窗口、RL / 模仿学习算法或 GPU 训练物理引擎（排行榜中会显式标注「训练框架 / 仿真器」）。
6. **可复现与审计证明**：
   - 榜单探索阶段接受单 seed（seed=0）；高关注度破纪录提交可按需补充多 seed 核验。
   - 提交 PR 必须附带可复现启动脚本（或固定 commit + patch）、训练日志、checkpoint_sha256 以及由 scripts/v2/fast_eval.py 生成的完整 40-clip 评测 JSON。

---

## 快速开始

```bash
git clone https://github.com/basicv8vc/modded-supermimic.git
cd modded-supermimic

# 1. 在 Linux CUDA 环境（Colab G4）安装锁定的运行时
pip install "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0" pytest
pip install "mjlab @ git+https://github.com/mujocolab/mjlab.git@0fb8a681136be94ffc636a3dd423cabb97d91f10"
pip install --no-deps --force-reinstall "mujoco==3.11.0" "mujoco-warp==3.11.0" "warp-lang==1.16.0"

# 2. 拉取锁定版本的 40 条 LAFAN1 数据并校验硬件/指纹
python3 scripts/fetch_lafan1.py
python3 scripts/freeze_manifest.py --check
python3 scripts/v2/gpu_runtime_check.py --scope acceptance

# 3. 评测任意实现 reset(context) / act(step) 的策略（G4 上约 150-170 秒）
python3 scripts/v2/fast_eval.py \
    --backend batched --batch-envs 40 --cache-references \
    --policy-file ../my_algo/policy.py:make_policy \
    --checkpoint ../my_algo/ckpt.pt \
    --algorithm MyAlgo \
    --training-commit <40-hex> \
    --training-wall-s <seconds> \
    --seed 0 \
    --out reports/v2/my_algo_seed0.json

# 4. 离线校验格式与注册表一致性（纯 CPU，无需 GPU）
python3 scripts/v2/audit_registry.py
```

reset(context) / act(step) 接口规范见 [docs/POLICY_INTERFACE.md](docs/POLICY_INTERFACE.md)；若需在 Colab G4 上端到端复现任一基线，见 [baselines/](baselines/README.md)。

---

## 仓库结构

| 目录                                                                           | 用途                                                                                                                                                                                                                    |
|:------------------------------------------------------------------------------ |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [benchmark/](benchmark/), [assets/g1/](assets/g1/), [scripts/v2/](scripts/v2/) | **冻结的裁判系统**（mjlab 1.6.0 + mujoco-warp FULL_COLLISION）、策略接口与 CLI / 一致性工具（benchmark.py）。                                                                                                           |
| [baselines/](baselines/README.md)                                              | 各基线在 Colab G4 + Google Drive 上的一键训练与评测工作流：[instanttrack](baselines/instanttrack/README.md)、[nablamimic](baselines/nablamimic/README.md) 与 [supermimic_mjlab](baselines/supermimic_mjlab/README.md)。 |
| [records/v2/](records/README.md)                                               | 面向阅读的竞速纪录卡片（001_...、002_...、003_...），关联 [reports/v2/](reports/v2/registry.json) 中经审计的完整 40-clip JSON。                                                                                         |
| [docs/](docs/)                                                                 | 协议设计、策略接口文档及 G4 基线与归因报告。                                                                                                                                                                            |
| [tests/](tests/)                                                               | CPU 契约 / 一致性单元测试（test_benchmark_v2.py）与 G4 CUDA 运行时验收测试（test_v2_gpu_runtime.py）。                                                                                                                  |

---

## 提交新纪录

1. 在 1× Colab G4 上从零训练，使用 scripts/v2/fast_eval.py --backend batched --batch-envs 40 跑完 40-clip 评测，并运行 2-clip 串行 vs 批量一致性校验（scripts/v2/compare_evals.py）。
2. 新建 records/v2/\<NNN\>\_\<name\>/，包含简洁的 README.md、训练配置 / patch / 启动脚本、train.log 及完整 40-clip 评测 JSON（放在 reports/v2/ 或纪录目录内，并在 [reports/v2/registry.json](reports/v2/registry.json) 登记）。
3. 在本地运行 python3 scripts/v2/audit_registry.py 与 pytest，确认**未改动 benchmark/ 与 assets/g1/ 下任何文件**后提交 PR。

---

## 开源许可

本仓库自撰的代码与文档采用 **MIT License** —— 见 [LICENSE](LICENSE)。

第三方例外：assets/g1/（从 [mujocolab/mjlab](https://github.com/mujocolab/mjlab) 原样复制的 Unitree G1 29-DoF MJCF 与 35 个 STL 网格文件）仍沿用 **Apache-2.0** 协议（见 [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)）。本仓库不重新分发 LAFAN1 动作数据，scripts/fetch_lafan1.py 会按锁定 revision 自动下载。完整声明见 [NOTICE](NOTICE)。
