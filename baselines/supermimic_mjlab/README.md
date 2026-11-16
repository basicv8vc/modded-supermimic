# Official SuperMimic (`HybridRobotics/supermimic`) — Colab + Google Drive Workflow

将**数据集缓存**、**每 500 iter 快照**、**最终模型**、**V2 评测报告**与**实时进度日志**全部与 Google Drive 无缝打通，避免 Colab VM 回收导致训练产物丢失或重复下载转换数据集。

## Google Drive 目录结构

所有持久化内容统一存放在 `MyDrive/supermimic_v2/`（可通过 `DRIVE_BASE` 覆盖）：

```text
MyDrive/supermimic_v2/
├── data/                         # 训练集与规范参考缓存（40 LAFAN1 clips）
│   ├── lafan1_g1/                # 40 条原始 CSV
│   ├── lafan1_g1_npz/            # V2 评测器 50 Hz NPZ
│   ├── lafan1_official_npz/      # HybridRobotics/supermimic MotionLoader NPZ
│   └── DATA_READY                # 完整性标记
└── runs/
    └── <RUN_ID>/                 # 如 official_35k_20261115T110000Z
        ├── STATUS.txt            # 每 60s 更新：当前阶段、iter、快照数、最佳成绩
        ├── pipeline.log          # setup / train / eval 完整日志
        ├── snapshots/            # 每 500 iter 落盘即原子同步：actor_snap*.pt + .wall + .meta.json
        ├── logs/                 # 官方 logs/SuperMimic-GenCoord-G1/<ts>/ (actor.pt, world_model.pt, critic.pt)
        └── evals/                # *_v2.json, sweep_summary.json, selected.json, 2-clip parity JSON
```

> [!NOTE]
> **为什么不把 `.venv` 或训练实时 I/O 直接跑在 `/content/drive` FUSE 上？**
> - `.venv` 包含数千个小文件，放 Drive FUSE 上会极大拖慢 Python 启动与 `uv sync`。
> - 训练热路径在 VM 本地 NVMe（`/content`）读写，由后台线程/进程在**每个 checkpoint 保存完成的瞬间**原子同步到 Drive，并每 60s 同步日志与评测结果——兼顾速度与不丢数据。
> - 每 500 iter 保存一次快照（35k iters 共 71 个 `actor.pt`，每个 ~46.4 MB），单次 run 约占 **~3.3 GB** Google Drive 空间。

## 快速开始

```bash
cd baselines/supermimic_mjlab

# 1. 一键启动：申请 G4 -> 挂载 Drive (需浏览器完成一次授权) -> 恢复/缓存数据集 -> tmux 训练 + 自动评测 + 实时同步 Drive
./run.sh all

# 2. 随时查看远程训练进度与 Drive 同步状态
./run.sh status

# 3. 列出 Drive 上已保存的所有 runs 与 STATUS.txt
./run.sh ls-drive

# 4. 下载汇总日志与 selected.json / sweep_summary.json 到本地 ./out/<RUN_ID>/
./run.sh collect

# 5. 释放 GPU（所有快照与评测已保存在 Drive）
./run.sh down
```

## 常用变体与断点续评

```bash
# 使用代理或私有仓库 PAT
PROXY=http://127.0.0.1:7897 GH_TOKEN=github_pat_xxx ./run.sh all

# 自定义迭代数与保存间隔（例如 25001 iters，每 1000 iter 存一次）
ITERS=25001 SAVE_INTERVAL=1000 ./run.sh all

# 即使原 VM 已回收，新开一台 G4 直接从 Drive 恢复快照并继续做 V2 评测
./run.sh up
./run.sh step setup
RUN_ID=official_35k_20261115T110000Z ./run.sh step eval
```

## 默认守卫

- `REQUIRE_DRIVE=1`：如果 `colab drivemount` 未成功挂载 `/content/drive/MyDrive`，脚本会**直接报错终止**，绝不在无 Drive 持久化的状态下空烧 GPU。
- `REQUIRE_G4=1`：校验 GPU 为 `sm_120`（Colab G4 `RTX PRO 6000 Blackwell`）；如需在 A100/L4 上调试，传 `REQUIRE_G4=0 GPU=A100`。
