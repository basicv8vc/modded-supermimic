# NablaMimic Colab 实验 Workflow

复现 ∇Mimic 论文 8 分钟实验（NablaEngine 版）的可重复工程：Colab CLI 管理会话、
固定 commit、CUDA 12.9 + 锁定依赖、Google Drive 数据缓存与结果同步、官方评估协议、
标准化报告。设计沿用已验证的 [SuperMimic Workflow](../../supermimic/colab_workflow/)（会话编排、
tmux 后台训练、周期同步、结果回收），训练/评估部分替换为 NablaMimic 的官方脚本。
基线结果与固定配置见 [../colab_baseline/BASELINE.md](../colab_baseline/BASELINE.md)。

## 前置条件

```bash
# 本机 colab CLI（一次性）+ OAuth 授权（任一 colab 命令打印 URL，浏览器授权）
uv tool install google-colab-cli   # 或 pip install google-colab-cli
colab sessions
```

- 有 Colab 额度的 Google 账号；本地网络需要代理时传 `PROXY=http://127.0.0.1:7897`
- 不需要 Docker / root；Colab VM 内自动装 CUDA toolkit 与 Python 3.12 venv

## 一键与分步

```bash
cd baselines/nablamimic

./run.sh all          # up → clone → deps → data → smoke → train → 等评估完 → collect
```

分步（调试/恢复用）：

```bash
./run.sh up           # 创建/复用 G4 会话 + 挂载 Drive + 上传脚本
./run.sh step clone   # 固定 commit 克隆 NablaMimic（校验 NablaEngine pin 不漂移）
./run.sh step deps    # CUDA 12.9 toolkit + Python 3.12 venv + 锁定依赖 + NablaEngine（首次 ~20 min）
./run.sh step data    # Drive 镜像恢复或官方脚本下载 LAFAN1（sha256 校验）+ 50Hz 缓存
./run.sh step smoke   # 37 项测试 + NablaEngine sm_120 编译 + 2 迭代真实训练路径（无 fallback 断言）
./run.sh step train   # tmux 启动 8 分钟训练 → 自动官方评估 → 标准化报告（断开本地不影响）
./run.sh status       # 随时查看训练/评估/同步进度
./run.sh wait         # 阻塞等待远端流水线结束（all 内含）
./run.sh collect      # 下载到 out/<run_id>/：逐文件 3 次重试 + 源端 sha256 校验，可断点续收，全过才写回执
./run.sh report       # 离线重新生成本地报告（不动远端）
./run.sh aggregate    # 多 seed 汇总 mean/std（拒绝混算不同版本/配置）
./run.sh down         # 停会话；持久化未经验证一律拒绝（fail-closed，FORCE=1 可强行）
```

会话已死、只剩 Drive 副本时（CPU 会话即可，不用 GPU CU）：

```bash
# 挂载 Drive 后直接指向真实目录，无需再手工 ln -s
SOURCE=/content/drive/MyDrive/supermimic/nablamimic/runs/<run_id> RUN_ID=<run_id> \
  COLLECT_CKPT=all ./run.sh collect
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `SESSION` | `nablamimic` | Colab 会话名（多会话并行实验时各用各的） |
| `GPU` | `G4` | G4 = RTX PRO 6000 Blackwell（论文用卡） |
| `RUN` | `ea` | 实验名（run_id 的前缀） |
| `SEED` | `0` | 训练 seed（0/1/2/...） |
| `BUDGET_MINUTES` | `8` | 有效训练预算（转成 `--hours` 覆盖，不动训练逻辑） |
| `DRIVE` | `1` | `0` = 不挂载/不同步 Drive |
| `COLLECT_CKPT` | `latest` | `latest`/`all`/`none`——collect 下载并校验哪些权重（`latest` 取迭代号最大者，不是字典序最大的文件名） |
| `PROXY` | — | 本地代理，如 `http://127.0.0.1:7897` |
| `RUN_ID` | — | 指定 run_id（status/collect/report/down 恢复用） |
| `SOURCE` | — | 恢复用：`collect` 的真实远端 run 目录，如 Drive 挂载后的 `/content/drive/.../runs/<run_id>`；默认仍是 `/content/nabla/runs/<run_id>`，需要 `RUN_ID` |
| `FORCE` | — | `1` = 跳过覆盖/持久化保护 |

例：`SEED=1 RUN=ea_s1 BUDGET_MINUTES=8 ./run.sh all`、`DRIVE=0 SESSION=dry ./run.sh step smoke`。

## run_id 与防覆盖

`run_id = <RUN>_s<SEED>_<UTC 时间戳>`（`tools/run_id.sh` 生成，如 `ea_s0_20261106T061500Z`）。
每次 `step train` 生成新 run_id，旧实验目录永不覆盖；VM 上记录在
`/content/nabla/logs/active_run_id.txt` 与 `run_history.txt`。

## Google Drive 布局

```text
MyDrive/supermimic/nablamimic/
├── data/
│   ├── data_mirror.tar.gz          # lafan1_g1/ + cache/ 的镜像（原子写 .part→rename）
│   └── data_mirror.tar.gz.sha256
└── runs/
    └── <run_id>/                   # 每次实验唯一目录
        ├── manifest.json           # 标准报告记录（见下）
        ├── config.json             # train.py 锁定的生效参数（argv 全量）
        ├── provenance.json         # 模型/engine/环境溯源
        ├── train.log  log.jsonl
        ├── checkpoints/            # it_*.pkl（上游 ckpt/ 规范化命名）
        ├── evaluation/
        │   ├── eval_nominal.json   # 官方评估原始输出（每 checkpoint × 40 clip）
        │   ├── per_clip.csv        # 逐 clip 明细（全部 checkpoint）
        │   └── summary.json
        ├── curve.csv  report.md  summary.json
        ├── .sync_status            # 同步心跳（时间戳 + 最新训练行）
        └── .sync_receipt           # 最终校验回执（源/目标逐文件 size+sha256）
```

- 训练先写本地 SSD，`90_sync.sh` 每 120 s rsync 到 Drive（`--partial` 断点续传），
  不在 Drive 挂载目录直接做高频写入；训练+评估结束后做最终同步并**逐文件校验**。
- **回执是校验结果，不是打卡记录**：`.sync_receipt` 只有在 8 个关键产物 + **每一个**
  checkpoint 的源/目标 size 与 sha256 全部相等时才写成 `verified=ok`；报告类文件缺失
  但 `log.jsonl` 与全部 checkpoint 一致时写 `verified=partial`；任一哈希不符或 rsync
  失败则**删除旧回执**并留 `.sync_receipt.invalid`，脚本返回非零。因此"回执存在"
  就等于"数据确实在 Drive 上且与源一致"。
- **持久化门禁（fail-closed）**：`./run.sh down` 只在这两种情况下放行——
  ① Drive 回执是 `verified=ok`（失败/不完整的 run 可接受 `verified=partial`），或
  ② 本地归档通过**逐文件哈希复核**：`collect_manifest.txt` 存在且 `ckpt_expected > 0`，
  并且 `down` 会向仍在运行的 VM **重新要一份 size+sha256 快照**，把每一个源文件与本地
  副本逐字节比对（缺失、尺寸不符、哈希不符、快照取不到都算不通过），同时核对 checkpoint
  数量与必需产物齐全。回执里的 `ckpt_local` **只作诊断，不作为证据**——它记录的是"当时
  收了多少"，不是"这些字节现在还在且完好"。
  查询失败、状态不可读、`UNSYNCED`、以及"本地只有 manifest.json"一律**拒绝**，
  必须显式 `FORCE=1` 才覆盖。默认 `COLLECT_CKPT=latest` 只下最新一个 checkpoint，
  因此想靠本地归档过关要用 `COLLECT_CKPT=all`。
- `DRIVE=0` 时一切只存 VM 磁盘，必须先 `collect`（且 `COLLECT_CKPT=all`）再 `down`。
- 首次 `step data` 会下载 ~2 GB 数据并回传镜像（比无 Drive 慢几分钟）；后续实验直接恢复。
- LAFAN1 许可为 **CC BY-NC-ND 4.0（学术非商业）**，数据永不进 GitHub。

## 结果回收（`collect` 的可靠性）

`colab download` 本身既不重试也不校验：代理断流会留下截断的 checkpoint，而旧版 collect
照样写一张"回收完成"的回执，于是坏文件会被后续流程当成归档成功。现在回收由
`tools/collect_run.py` 负责，并且**不信任任何一方自述**：

1. **先取源端事实**：`colab exec` 在 VM 上现算每个待回收文件的 `size` + `sha256`
   （`COLLECT_SNAPSHOT_BEGIN/END` 框住的机器可读快照，同时给出全部 checkpoint 的迭代号）。
   快照取不到、框不完整、行格式异常 → 立即失败（`FORCE` 不覆盖这一类失败）。
2. **逐文件最多 3 次重试**，每次先写 `<name>.part.<pid>`：size 或 sha256 与源端不符算失败；
   只有逐字节匹配的临时文件才会被 `rename` 到最终路径（原子替换）。
3. 因此**已校验的文件永远不会被未校验的字节覆盖**：重试失败只丢临时文件，
   上一次成功回收的归档原样保留。
4. **只有全部必需文件校验通过才写 `collect_manifest.txt`**（先写临时文件再原子替换，
   并逐项记录 `verify=ok`、`ckpt_expected`、`ckpt_local`、`files_verified`）。任何失败都会
   删除该回执、留下 `collect_manifest.txt.failed`（逐条原因）并返回非零——不存在
   "看起来成功"的中间态。
5. `COLLECT_CKPT` 语义不变，但 `latest` 现在按**迭代号**取最新（`it_1000` 比 `it_500` 新，
   字典序会取错），`all` 必须收齐并校验每一个 checkpoint——部分成功不算成功。
6. 状态查询、快照获取、网络中断一律 fail-closed：pipeline 仍 `RUNNING` 时拒绝回收
   （结果还在写）；`none/latest/all` 里 `latest`/`all` 在 VM 上一个 checkpoint 都没有时直接失败
   （只要日志请用 `COLLECT_CKPT=none`）。
7. 必需文件随状态而定：`manifest/config/log` 永远必需；pipeline 为 `PERSISTED`/`COMPLETE` 时
   `summary/curve/report` 与 `evaluation/{eval_nominal.json,per_clip.csv}` 也必需
   （与 `down` 的本地归档要求一致）；历史 run 缺失的 `train.log`/`ea_report.json` 仍可容忍。
8. **本地 I/O 错误与远端错误同等对待**：`mkdir`/`stat`/`sha256`/`rename` 任一处抛 `OSError`
   （磁盘满、权限不足、目标路径不是普通文件）都会被当成一次失败的传输尝试重试；若它逃出
   逐文件循环，顶层统一转成 `fail()`——旧回执照样删除、写 `.failed`、退出码 **5**
   （`3` = VM 查询失败，`4` = 校验失败，`5` = 本地文件系统错误）。`FORCE=1` 重收已有归档时
   中途遇到磁盘错误，绝不会留下上一次的成功回执继续"证明"归档完整；万一连回执都删不掉，
   会改写其内容为 `verify=FAILED` 并大声报错，而不是静默退出。

### 断点续收（C-1）

重跑 `collect` 不再重下已经在盘上的字节，但也**不假设**它们是好的：

- **可信源清单每次重新取**（不是读回上次的回执），源端任何哈希变化都会让本地副本作废；
- 盘上已存在的文件**逐个重新量** size + sha256，**两者都与当前源清单完全一致**才跳过下载；
  缺失、截断、内容损坏、`CLI` 之外被替换 → 一律重下；
- 每个文件校验通过后立刻把进度原子写进独立的 `.collect_partial.json`（先写 tmp 再
  `rename`）。这个文件**永远 `verify=partial`**，只是本次尝试的进度记录，**绝不能被当成
  成功回执**：它不参与任何跳过决策（跳过只由"重新测量 + 当前源清单"决定），全量校验通过后
  还会被删除，成功的归档里不留这个中间态文件；
- `.collect_partial.json` 损坏、truncated、字段类型不符（例如 `"files": 42` / `true` /
  `"all"`）、属于别的 run、甚至自称 `verify=ok` → 逐字段做类型校验后**忽略整个文件**，只打一行
  说明并继续，绝不抛异常、绝不中断收集；它始终只是诊断信息，不是证据；
- **先撤销、再改动**：打算写入任何归档字节之前，若存在旧的 `collect_manifest.txt`，先用原子
  `rename` 移到 `collect_manifest.txt.superseded`（没有任何门禁、脚本或报告读这个名字），并在
  `.collect_partial.json` 里记下 `archive_state=collecting` 与被撤销回执的 verdict/时间/哈希。
  因此 `FORCE=1` 重收途中被 **SIGINT / SIGTERM / SIGKILL** 打断、或 VM 直接消失，归档里都
  **不可能**留下一张 `verify=ok` 为已经被换掉一半的字节背书；`down` 会因为没有成功回执而拒绝。
  旧回执原文只作为取证材料留存在 `.superseded`，新回执写成功后再把它删掉；
- 旧的 `collect_manifest.txt` 仍是**唯一**的成功凭证；任何一次失败都会删掉它并写
  `.collect_manifest.txt.failed`，哪怕失败发生在"续收"过程中；
- `./run.sh collect` 可以直接进入"有目录但没有成功回执"的残局目录（那正是被中断的收集），
  但**含 `collect_manifest.txt` 的已校验归档仍然需要 `FORCE=1`**。

回执里因此多了 `files_downloaded=` / `files_resumed=`，重跑一次"全部已在校验过"的归档会看到
`files_downloaded=0`。

### 从指定远程目录恢复（C-2）

`--source`（`run.sh` 里是 `SOURCE=`）把收集源从 `/content/nabla/runs/<run_id>` 指向任意真实
run 目录，典型用法是 Drive 挂载后的副本——会话已回收时不必再手工建软链接。

路径按**路径**校验，不按文本拼接：必须是绝对路径、`normpath` 后不再变化（`/a//b`、`/a/./b`、
带尾斜杠一律拒绝，避免"看起来同一个目录其实不同"）、不含 `..`、不含引号/空白/`;|&$<>*?!#`
一类字符，且**最后一段必须等于 run_id**；随后再用源目录自己的 `manifest.json` 里的
`experiment_id` 复核 **run identity**。显式 `--source` 是人工指定的目录，所以这里**必须严格**：
只有 `experiment_id` 非空且等于 `--run-id` 才允许建立成功归档；缺失、JSON 解析失败、或指向别的
run 一律以 `rc=2` 拒绝、不写回执、也不下载任何文件（一份字节完好的错误实验目录仍然是错误实验）。
默认的 VM 源路径不是人工拼的（run id 来自 VM 上的状态查询），为兼容早期没有该字段的 run，
仍保留原行为，并把 `run_identity=-` 如实写进回执。远程程序里的路径用 `repr()` 注入、下载用
argv 传参，不经过 shell。

**完整性 ≠ 来源可信**，两者分开记录，绝不合并成一句"已验证"：

| 事实 | 字段 | 含义 |
|---|---|---|
| 字节与源一致 | `verify=ok` | 收到的字节等于**这次的源**的快照 |
| 源就是训练 VM 的产物 | `provenance=` / `provenance_degraded=` | 源本身可归因于训练 VM |

从 Drive 恢复时，Drive 里两个文件哈希相同只能证明**这份副本自身完整**，不能证明它就是训练 VM
写出来的东西。归因需要证据：训练 VM 上的 `vm/90_sync.sh` 在逐文件比对源与 Drive 之后写下的
`.sync_receipt`。收集器要求：

- 收的**每一个**文件都在回执里有一行 `OK <rel> <size> <sha256>`，且 size/sha256 与本次快照
  一致；
- 回执的 `dst=` 就是这个源目录，`verified=` 为 `ok` 或 `partial`。

全部满足 → `provenance=drive-recovery-verified`（`provenance_evidence=vm-sync-receipt`，并记录
回执的 `verified_at`、checkpoint 计数和回执文本的 sha256）。任何一条不满足（回执缺失、`dst`
对不上、某个文件没被回执背书、哈希不符）→ **降级**为 `provenance=drive-copy-only` 且
`provenance_degraded=1`，并在 stderr 明确写出"这批字节经过校验，但这**不是**它们来自训练 VM
的证据"。降级不会伪造 `verify=ok`，也不会把降级包装成失败：文件完整性是真的，归因不是。

恢复模式**不改 `down` 的门禁**，且这里要准确说明它原本就不是"总会重比远端哈希"：门禁有两条
并列的放行分支——① Drive 上的 `.sync_receipt` 为 `verified=ok`（失败/不完整的 run 允许
`verified=partial`）时**直接放行**，不再向 VM 取快照逐文件比对；② 只有 Drive 回执不是 `ok`
时，才向仍可查询的 VM **重新要一份 size+sha256 快照**、逐文件复核本地字节（并要求
`collect_manifest.txt` 存在、`ckpt_expected>0`）。两条都失败则拒绝关机。因此一份刚从 Drive
恢复、但没有 VM 可以再取快照的归档，**不会**靠自己打开 `down` 门禁；`collect` 的本地回执与
Drive 回执都只是线索，不是通行证。

### 采集前诊断（C-3）

开始搬字节之前，`out/<run_id>/collect_diagnostics.txt` 先落一份（结束时原子重写并写上
`outcome=`）：`started_at_utc`、`run_id`、`source_path`、`data_source`（`training-vm` 或
`drive-recovery`）、`colab_session`、`colab status` 的**返回码与输出**（或 `timeout` /
`FAILED` + stderr）、`colab_session_id`（解析不到就直说 `(not reported)`）、源端
`.sync_receipt` 状态、`pipeline_state` 及其**来源**（CLI 还是回执）、`provenance`。

- 状态查询失败**只记为失败**，绝不写成 `ok`；
- 诊断是**尽力而为**：它自己写不下去、或 `colab status` 查不到（会话已回收，正是恢复场景），
  都不会阻断一个完整性已经验证通过的归档——`collect_diagnostics.txt` 不是证据。


## 标准报告（manifest.json / report.md 字段）

Experiment ID · Algorithm · Git SHA · Seed · GPU · Engine · Model · Training Time ·
Updates · Normalized Return · Failed Clips · Drive Path · Status（PASS/FAIL/INCOMPLETE），
外加 checkpoint 时间曲线（`curve.csv`）、`config_fingerprint`、`eval_protocol_fingerprint`
和 `integrity_problems`。`Normalized Return` 取**训练预算内最后保存的那个 checkpoint**
（"8 分钟时的指标"；预算后的溢出存档不算 headline，另记于 summary）——若这个点没被
评估，run 是 INCOMPLETE，而不是悄悄退到更早的点报一个更好看的数。

`PASS` 需要同时满足：预算点已评估、该点 40 条 clip 齐全、`actor_key == actor_ema`、
return 有限且落在 [0,1]。否则写入 `integrity_problems` 并标 INCOMPLETE——不完整的评估
不会混进多 seed 均值。

两个指纹各管一件事：

- `config_fingerprint`：规范化后的**算法配置**（`argv` 解析成 key→value，后者覆盖前者，
  去掉 seed/out 等非算法参数并排序）。同一配置换个参数顺序、换 seed、换输出目录，
  指纹不变；改训练预算则变。
- `eval_protocol_fingerprint`：评估协议（脚本、40 clip、EMA actor、确定性、
  10 次 solver iteration、摔倒计分规则）单独哈希。

`./run.sh aggregate`（= `aggregate_seeds.py --scan out --series $RUN`）只在同一实验系列内
汇总 mean / std。"系列"由**完整 run_id 语法** `<series>_s<seed>_<UTC 时间戳>` 解析得出，
**不是前缀匹配**：`--series ea` 只选 `ea_s0_…` / `ea_s1_…`，不会把 `ea_verify_s0_…`
（系列 `ea_verify`）或 `exp-b2_run_s0_…`（系列 `exp-b2_run`）拉进同一个均值；
要汇总校验实验就写 `--series ea_verify`。系列名可含下划线、数字与连字符。

不符合该语法的 run_id（时间戳残缺、尾部多余字符、run_id 里的 seed 与 manifest 不符）不会被
静默纳入：逐条打印 `SKIP: … 原因` 到 stderr，并记入 `aggregate.json` 的 `excluded_run_ids`；
`aggregate.json` 同时记录本次选择的 `series`。选不到任何 run 时以 `rc=2` 明确失败。

git SHA、两个指纹、模型或引擎任一不同即**拒绝混合**（`rc=3`），
同一 seed 出现多次也会拒绝（`rc=4`，除非 `--allow-duplicate-seeds`，那衡量的是
run-to-run 方差而不是 seed 方差）。

## 训练进程与断线

- `step train` 用远端 `Popen(start_new_session)` + tmux 双层托管：本地 CLI 断连、
  kernel 重连都不影响训练（此前手动编排的教训已固化为机制）。
- 流水线顺序：train → 官方评估 → finalize 报告 → 最终 Drive 同步 → 状态文件
  （`PERSISTED` / `UNSYNCED` / `INCOMPLETE` / `INCOMPLETE_UNSYNCED` / `FAIL`：
  `PERSISTED` 才是"算完了且数据已验证落盘"，`UNSYNCED` 表示计算完成但归档未校验，
  `down` 会拒绝后者）。后台另有 fallback 看门狗：训练日志若出现
  `warp physics`（静默 fallback）立即杀进程判失败。
- 恢复：会话在跑时 `./run.sh status` 即可；若评估中断，`./run.sh step eval` / `step report`
  可单独重跑（已评估 checkpoint 自动跳过）；VM 丢失时从 Drive `runs/<run_id>/` 取产物。

## 实测验证状态

| 功能 | 状态 |
|---|---|
| 完整 train→eval→report 链路（NablaEngine，G4） | ✅ 已实测（Experiment A 归档；workflow 端到端见下） |
| CUDA 12.9 安装、engine sm_120 编译、502 pairs、无 fallback | ✅ 已实测 |
| LAFAN1 下载/sha256/50Hz 缓存 | ✅ 已实测 |
| Drive 挂载/镜像回传/周期同步/`.sync_receipt` | ✅ 已实测（2026-11-06 首跑，镜像 875 MB + run 3.25 GB） |
| run.sh 编排（run_step 轮询/collect/down 门禁） | ✅ 已实测（含 `./run.sh down` 关会话） |
| 完整性保障（源/目标校验、可靠回收、本地 I/O 撤销、严格系列匹配、fail-closed 门禁、指纹、评估完整性） | ✅ 308 项非 GPU 测试，含负向用例（详见「测试」） |
| 多 seed aggregate | ✅ fixture + 真实归档（重复 seed 拒绝、跨协议拒绝） |

### 首次端到端实测（2026-11-06，`ea_s0_20261106T142202Z`）

`PROXY=... SESSION=nm ./run.sh` 串起 up→clone→deps→data→smoke→train→eval→report→sync→collect，
产物见 `out/ea_s0_20261106T142202Z/`，Drive 上 28 文件 / 3.25 GB / 7 个 checkpoint 全量。

| 指标 | 本次 | 论文（Fig.1 / Table 13） |
|---|---|---|
| it_1250 wall clock | 7.99 min（iter 0.377 s） | 8 min（0.38 s） |
| normalized return | **0.8341** | 0.82（3-seed mean） |
| failed clips | **0/40** | — |

首跑暴露并修掉 4 个只在真实环境出现的问题（均已包含在本目录代码中）：

1. `run_step` 用 BRE `\|` 做交替判断，macOS/BSD grep 不认 → 步骤已 `DONE` 仍轮询到 45 min 超时，
   且 `FATAL` 检测一并失效（失败被掩盖）。改 `grep -qE`。
2. `upload_all` 任一文件上传失败即 `set -e` 静默退出。改为 3 次重试 + 显式告警。
3. `20_deps.sh` 用 `"Cuda" in type(d).__name__` 判 GPU：jax 0.12 的设备类名统一为 `Device`
   （repr 仍显示 CudaDevice），断言必然失败。改为 `jax.default_backend() == "gpu"` + platform 校验。
4. `50_pipeline.sh` 调 `90_sync.sh once` 未传 `DST`，脚本里 `DST` 默认空 → 最终同步 `FATAL: DST not set`。
   改为 `90_sync.sh` 自动从 `logs/<run_id>.drive_dir.txt` 取默认值。

成本：G4 全程约 90 min，14.21 CU。

### 审核整改（Request Changes → 已修）

按 review 意见重做了完整性保障，未改动算法、模型或任何训练超参数：

| 问题 | 现在的行为 |
|---|---|
| P0-1 回执只对目标做哈希 | 源/目标逐文件 size+sha256 比对，覆盖**每一个** checkpoint；rsync 失败或缺 checkpoint 则删除旧回执、返回非零 |
| P0-2 `down` 在未知状态放行 | fail-closed：查询失败/状态不可读/`UNSYNCED`/"只有 manifest" 全部拒绝；仅 `verified=ok` 回执，或**经逐文件 size+sha256 复核**（重新向 VM 取快照，非读回执里的计数）的本地归档放行 |
| P0-3 同步失败仍报 COMPLETE | 拆分 `PERSISTED`/`UNSYNCED`/`INCOMPLETE`/`INCOMPLETE_UNSYNCED`/`FAIL`，`sync_rc` 参与判定，不再存在"COMPLETE"这个会骗人的状态 |
| P1-4 指纹含 seed、协议未校验 | `config_fingerprint` 规范化（去 seed/out、last-value-wins、排序）+ 独立 `eval_protocol_fingerprint`；`aggregate --series` 按系列选择，重复 seed 拒绝（`rc=4`） |
| P1-5 部分评估也能 PASS | 预算点必须已评估、40 clip 齐全、`actor_key==actor_ema`、数值有限；否则 INCOMPLETE 并列出 `integrity_problems` |
| P1-6 上传失败继续跑 | 3 次重试后直接中止，并核对远端脚本 sha256 与本仓库一致，防止跑旧脚本却记录新 commit |

顺带修掉两处归档一致性问题：`finalize_run.py` 现在也能读 `evaluation/eval_nominal.json`
（`collect` 的布局），预算点取"预算内最后保存的 checkpoint"而不是"最后一个已评估的"。
两个归档已按新指纹重建，`aggregate --series ea --allow-duplicate-seeds` 给出
mean 0.8238 / std 0.0146（同 seed 两次运行的 run-to-run 方差，不是 seed 方差）。

### 第二轮审核整改（Workflow 可靠性）

真实 G4 端到端验收已 PASS（`ea_verify_s0_20261107T085809Z`：normalized return 0.8307、
failed clips 0/40、Drive 与本地归档校验通过），本轮修两个 Workflow 可靠性 P1（外加复审追加的一个本地 I/O 完整性小修），
**不碰算法、G1 模型、训练超参数与评估协议**，也没有重跑 GPU：

| 问题 | 根因 | 现在的行为 |
|---|---|---|
| P1-1 `aggregate --series` 误匹配 | 用 `startswith(prefix)` 选系列，`--series ea` 会把 `ea_verify_s0_…` 一起选中，两个实验系列混进同一个 mean（且因为 config/protocol 指纹相同，一致性门禁拦不住） | 按完整 run_id `<series>_s<seed>_<UTC>` 解析出系列再**精确比较**；非法 run_id 逐条 `SKIP` 并记入 `excluded_run_ids`；`aggregate.json` 记录 `series` |
| P1-2 `collect` 无重试无校验 | `colab download` 一次失败即 `\|\| true` 放过，且旧回执只统计 checkpoint 个数，截断/损坏文件会被当成"回收成功" | 源端 size+sha256 快照 → 逐文件 3 次重试 → `.part` 原子替换 → 全过才写 `collect_manifest.txt`；查询失败 fail-closed；`latest` 按迭代号取最新；`all` 不容忍部分成功 |
| P1-3 本地 I/O 错误绕过撤销 | `part.stat()`/`sha256_of()`/`os.replace()`/`mkdir` 抛出的 `OSError` 无人捕获，顶层只捕 `RemoteError` → `FORCE=1` 重收途中磁盘出错会直接 traceback 退出，**旧的 `verify=ok` 回执原地幸存** | `OSError` 在逐文件循环内算一次失败尝试（重试后可失败告终），逃出循环的由顶层统一转成 `fail()`：删旧回执、写 `.failed`、退出码 `5`；连回执都删不掉时改写为 `verify=FAILED` 并大声报错 |

第二轮（collect 的耐久性）：

| 问题 | 根因 | 现在的行为 |
|---|---|---|
| C-1 `collect` 不可恢复 | 重跑一次就把 1.5 GB 全部重下；逐文件哈希只用于当场放行，不留任何进度，"下到 72 MB 断线"等于从零开始 | 每次重新取**可信源清单**，再逐个重新测量本地文件：size+sha256 与当前源完全一致才跳过，缺失/损坏/源哈希变化一律重下；每验完一个文件就原子更新 `.collect_partial.json`（恒为 `verify=partial`，全量通过后删除，任何情况下都不被当成成功回执）；失败仍撤销 `collect_manifest.txt` |
| C-2 只能从 VM 固定路径收 | `run.sh`/`collect_run.py` 写死 `/content/nabla/runs/<run_id>`，会话一回收就只能手工 `ln -s` 到 Drive 副本，且没有任何来源证据 | `--source`（`SOURCE=`）指向真实 run 目录；路径做绝对/规范化/字符集/`..`/尾段=run_id 校验，再用源内 `manifest.json` 的 `experiment_id` 复核 run identity；**完整性**（`verify=ok`）与**归因**（`provenance=`）分开记录：只有 `.sync_receipt` 逐文件背书时才 `drive-recovery-verified`，否则降级为 `drive-copy-only` + `provenance_degraded=1` 并明说"这不是来自训练 VM 的证据" |
| C-3 采集前无诊断 | 出问题后只能靠回忆"当时是哪个会话、查没查到状态" | collect 前写 `collect_diagnostics.txt`：UTC 起始时间、run_id、数据源路径与 `data_source`、`colab status` 的返回码/输出/超时/stderr、解析出的 session id、源端 `.sync_receipt` 状态、pipeline 状态及其来源、最终 `outcome`；探不到状态只记失败，绝不阻断一个完整性已验证的归档 |

### 第三轮整改（PR #9 复审 REQUEST CHANGES）

| 复审发现 | 修复 |
|---|---|
| **P1** `FORCE=1` 重收被 SIGINT/SIGTERM/SIGKILL 打断时，旧 `verify=ok` 直到 `fail()` 才撤销；进程直接死掉会让旧回执继续为已经换掉一半的字节背书 | 在**第一个归档字节被改动之前**，用原子 `os.replace` 把旧回执移到 `collect_manifest.txt.superseded`（没有门禁读这个名字）；`.collect_partial.json` 写明 `archive_state=collecting` 和被撤销回执的 verdict/时间/sha256；新回执成功后才删掉取证副本。连"撤销失败"也 fail-closed：改不动旧回执就一个字节都不写 |
| **P1** `.collect_partial.json` 里 `"files": 42` / `true` 会让 `len(old.get("files") or {})` 抛 `TypeError`，逃出只捕 `OSError`/`ValueError` 的处理，traceback 退出且旧回执幸存 | 读回时按 schema 逐字段校验类型（根为 dict、`kind`/`verify`、`files` 是 dict 且每项含合法 size/sha256、`run_id`/`updated_at` 是字符串），不符即**忽略整个文件**；再加兜底异常保护，任何形状的进度文件都不可能中断收集 |
| **P2** 显式 `--source` 的 `manifest.json` 缺 `experiment_id` 或 JSON 解析失败时，identity 为空，旧代码跳过比对就可能把错误的 Drive 目录当成目标实验收下来 | 显式源**必须** `experiment_id == --run-id`，否则 `rc=2`、不写回执、不下载任何文件；`IDENTITY -`/裸 `IDENTITY` 也正确解析成"无身份"而不是当成快照格式错误。默认 VM 源保持历史兼容 |
| 文档：`down` 被写成"总会重新对比远端文件与本地哈希" | 改成代码的真实两分支：Drive `.sync_receipt` 为 `verified=ok`（或失败 run 的 `partial`）时直接放行；只有它不成立时才向 VM 重新取快照逐文件复核 |

修改文件：`tools/aggregate_seeds.py`、`tools/collect_run.py`（新增）、`run.sh`（`collect` 重写 +
`COLLECT_CKPT` 校验）、`tests/run_tests.sh`、`README.md`、`vm/20_deps.sh`（仅 CUDA 版本探测行）。
历史实验数据、指标与指纹一字未改。

第二轮只动四个文件：`tools/collect_run.py`、`run.sh`（`collect` + `SOURCE` 校验）、
`tests/run_tests.sh`、`README.md`。算法、模型配置、评估协议、`vm/*.sh`、`down` 门禁与
`90_sync.sh` 的回执语义均未改动，也没有新增任何历史实验结果。

复审（REQUEST CHANGES）后的第三轮只动 `tools/collect_run.py`、`tests/run_tests.sh`、
`.gitignore`、`README.md`：修掉"异常中断留下旧回执""畸形 partial manifest 抛
`TypeError`""`--source` 身份缺失仍放行"三个问题，并把 `down` 门禁描述改成与代码一致。
`run.sh`、`vm/*.sh`、训练/评测代码与历史结果一字未改。

## 测试

```bash
cd tests && ./run_tests.sh     # 308 项，全本地、无 GPU、无 Colab
```

覆盖：shell 语法/shellcheck、参数校验、run_id、finalize/aggregate（用 `out/` 真实归档做
fixture）、Drive 同步的**正/负向**校验（成功、rsync 失败删回执、partial、缺 checkpoint）、
`down` 门禁（RUNNING、查询不可读、只有 manifest、本地不完整、UNSYNCED、Drive ok、
FORCE 覆盖）、上传失败与远端哈希不符的阻断、指纹归一化（seed/顺序/重复覆盖/预算）、
评估完整性（半数 clip、非 EMA actor、NaN、预算点缺失）、aggregate 的系列选择与
跨协议/重复 seed 拒绝。

本轮新增（第 12、13 节，共 83 项）：

- **严格系列匹配**：`ea` 与 `ea_verify` 同时存在时统计互不污染（mean 分别是 0.8221 / 0.8307）；
  含下划线+数字+连字符的 `exp-b2_run`；`--series ea` 不吃 `ea_s10`（seed 10 属系列 `ea`）、
  `--series ea_s1` 只吃 `ea_s1_s0`；前缀不匹配（`exp`/`exp-b2` 都选不到 `exp-b2_run`）；
  4 种非法 run_id（短时间戳/缺 Z/尾随字符/无时间戳）被 `SKIP` 且不进 `per_seed`；
  run_id 与 manifest seed 不一致时排除；`--series` 含 `/` 时 `rc=2`；真实归档上
  `--series ea` = 0.8238（不再被 `ea_verify` 拉成 0.8261）。
- **可靠回收**（本地 mock `colab` + 假 VM 目录，不连 Colab）：首次失败第二次成功、
  连续 3 次失败、SHA256 不符（同尺寸不同字节）、截断、快照查询失败、状态查询失败、
  active-run 查询失败、pipeline 仍 RUNNING、`all` 缺一个 checkpoint、`latest` 最新那个失败时
  不回退到旧权重、`latest` 取 `it_1000` 而非字典序的 `it_500`、`PERSISTED` 缺 `report.md`
  被拒、`INCOMPLETE` 允许缺报告产物、`none` 不要求 checkpoint 且日志照收、
  VM 无 checkpoint 时 `all` fail-closed；
  **失败不产生虚假回执**（无 `collect_manifest.txt`、留 `.failed`、文件不落盘）、
  **失败的重试不破坏已校验归档**（逐文件 sha256 比对前后一致，随后成功重收可修复回执）、
  新回执仍满足 `down` 的本地归档门禁（`all` 放行、`latest` 拒绝）。
- **本地 I/O 失败不绕过撤销**（复审第 3 轮）：下载回来后读不出（`chmod 000`）→ 记为
  `local i/o error` 并重试；校验通过但目标路径是非空目录导致 `rename` 失败 → 旧
  `verify=ok` 回执被撤销、`.failed` 落盘、其它已校验文件不动；`mkdir` 抛出的
  `OSError` 逃出逐文件循环 → 退出码 `5`、回执被撤销、输出里没有 `Traceback`；
  修好文件系统后再收一次能恢复 `verify=ok` 并清掉 `.failed`。
  这 16 项在修复前的代码上有 8 项失败（其中正是"撤销旧回执"两条），已用作回归极性验证。

第二轮新增（第 14 节，共 82 项，同样是本地 mock `colab` + 假 VM/假 Drive 目录）：

- **C-1 恢复语义**：首次收集把每个文件都下了一遍；第二次运行 `files_downloaded=0` 且
  mock 一次 `download` 都没收到（用下载调用日志证明"跳过"是真的跳过，而不是"下了但没写"）；
  单个 checkpoint 被同尺寸改写 → 只重下这一个并逐字节修复；源端哈希变化 → 本地副本作废重下。
- **C-1 中断与续收**：第三个 checkpoint 连续 3 次失败 → 失败收场、无回执、留
  `.collect_partial.json`（断言 `kind=collect-partial`、`verify=partial`、必不含 `verify=ok`）；
  去掉故障后**不加 FORCE** 直接续收 → 只下缺的那一个、回执恢复、进度文件被删除。
- **C-1 续收中途再次失败**：第一次停在 `it_500`，第二次让它坏在 `it_1000` → 仍然 rc≠0、
  无成功回执、`.failed` 在、已校验的 `it_250`/`it_500` 一个字节不动；第三次才收齐。
- **C-1 回执撤销**：已有 `verify=ok` 的归档被一次失败的 `FORCE=1` 重收 → 旧回执消失、
  `.failed` 出现、幸存文件保留；修好后再收一次能恢复 `verify=ok` 并清掉 `.failed`。
- **C-1 进度文件不可信**：内容不是 JSON、或者（更狠）伪造一份 `verify: ok` 且把
  `summary.json` 标成"已校验"的进度文件 → 都只被忽略并重新测量，损坏文件照样重下。
- **C-2 `--source`**：从假 Drive 目录恢复成功，且 mock 的下载日志里全是
  `drive/MyDrive/...`、没有一条 `/content/nabla/runs/...`（证明走的是 Drive、不需要软链接）；
  回执 `data_source=drive-recovery`、`source_path` 正确、`state_source=sync receipt`。
- **C-2 来源证据**：带训练 VM `.sync_receipt` → `provenance=drive-recovery-verified`、
  `provenance_degraded=0`；回执缺失、某个文件被回执写成不同哈希、回执 `dst=` 指向别的目录、
  `verified=failed` → 一律降级为 `drive-copy-only` + `provenance_degraded=1`，并在 stderr
  明说"这不是来自训练 VM 的证据"，同时 `verify=ok` 只表达字节完整。
- **C-2 非法来源**：相对路径、含 `..`、非规范（`/a/./b`）、含 shell 元字符（`; rm -rf`）、
  尾段不是 run_id、`SOURCE` 但没给 `RUN_ID`、源 `manifest.json` 的 `experiment_id` 不符
  → 全部拒绝且不写回执；**恢复不绕过 `down` 门禁**（收完后 `down` 仍拒绝不可证实的一次）。
- **C-3 诊断**：`collect_diagnostics.txt` 里的 UTC 时间戳、run_id、源路径、`data_source`、
  `colab status` 输出与解析出的 session id、`outcome=verified`；`colab status` 失败时记
  `colab_status=FAILED` + `rc=1` + stderr 原文，**不写 `ok`**，且不阻断一个完整性已验证的归档。
- **兼容性**：老的 `collect_run.py` 调用（不含 `--source`）行为不变——默认
  `/content/nabla/runs/<run_id>`、`data_source=training-vm`、`provenance=training-vm`、
  历史回执字段（`ckpt_expected`/`ckpt_local`/`integrity`）照旧。
- 顺带修掉两处"测试本身随数据漂移"的问题：真实归档聚合改为扫描**已提交 manifest 的副本**
  （否则本地多一个未提交的 seed 1 run 就会把 0.8238 变成 0.8254，把代码检查变成数据检查）；
  "失败的重收"用例先删掉本地 `manifest.json` 再注入下载故障（可恢复收集会跳过已校验文件，
  不删的话故障根本触发不到）。

复审新增（第 15 节，共 51 项，全是本地 mock `colab` + 假 VM/假 Drive，无 GPU/Colab/真实 Drive）：

- **P1 异常中断的回执残留**：mock 在 `download` 中途对 collector 的 python 进程真发
  `SIGKILL` **和** `SIGTERM`（用 `dl_killed_by` 标记证明确实是被信号杀掉、不是走了正常失败
  分支），两种情形下都断言：没有 `collect_manifest.txt`、旧回执只以
  `collect_manifest.txt.superseded` 存在、`.collect_partial.json` 明确写着
  `archive_state=collecting` / `verify=partial` / 被撤销的 `verify=ok`、已校验文件没被破坏、
  `down` 因无回执仍然拒绝；随后**不加 FORCE** 的一次续收恢复 `verify=ok`，并清掉
  `.superseded` 与进度文件。
- **P1 畸形进度文件**：`files` 为 `42` / `true` / `"all"` / `{"a":1}` / 字段类型错误、
  `run_id` 为数字、`updated_at` 为 `null`、根结点是数组等 8 种形状，逐个断言收集正常完成、
  输出里出现"ignoring"、**没有 `Traceback`**（旧代码在 `len(files)` 上抛 `TypeError`）。
- **P2 严格来源身份**：Drive 源 `manifest.json` 缺 `experiment_id`、JSON 解析失败、
  指向别的 run → `rc=2`、不写回执、一个文件都不下载；身份匹配仍正常成功且回执记
  `run_identity=<run_id>`；默认 VM 源保持历史宽松行为并如实记 `run_identity=-`。

不为测试 Workflow 重跑 GPU 实验；真实 Colab 冒烟测试先报成本再授权。

## 与手动流程的对应（Experiment A）

`10_clone`+`20_deps` = 当时的 `colab_ea_run.sh env`（CUDA toolkit/venv/setup_env.sh）；
`40_smoke` = `preflight`（pytest + engine 构建 + 2 迭代真实路径）；`50_train`+pipeline =
`train`+`eval`+`report` 三段；差异仅在 run_id 唯一化、Drive 持久化与标准化产物。
