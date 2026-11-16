# PR: 让 Evaluator 不再假设 Policy 的结构

**范围**：`research/modded-supermimic/`。目标是把「算法必须长成 mjlab 的样子」改成「算法自己负责推理，Evaluator 只收 29 维动作」，同时不动物理、评分和门槛。

## 1. 删除了哪些不合理约束

| 原约束 | 位置 | 现状 |
| --- | --- | --- |
| 观测必须是 160 维 mjlab actor 向量 | `runner.evaluate_clip` 直接把 `env.observation()` 交给策略 | 删除。策略自己从 `StepInput` 构造任意编码 |
| checkpoint 必须含 `actor.*` 的 MLP 键 | `policy.load_policy` 被当作唯一入口 | 降级为**一种**实现。`--policy-file` 路线下 Evaluator 完全不解析权重 |
| 不符合该结构即判定「不兼容」 | `PolicyFormatError: no actor MLP found` | 该错误现在只属于 mjlab 原生 loader；不再阻止任何算法接入 |
| 提交门禁要求 `policy_kind == "checkpoint_actor"` | `runner.submission_errors` | 改为「声明了 kind 且不是 scripted smoke」 |
| 冒烟策略依赖 `obs[:29]`（160 维布局） | `scripts/run_canonical_eval.py::SmokePolicy` | 改用 `context.reference` |

没有做的事（按约束）：没写 Adapter Registry、没有通用权重转换器、没有多层抽象、没有为 SuperMimic / NablaMimic 写转换器、没改门槛、没改物理、没跑训练。

## 2. 保留了哪些核心模块（未改写）

| 模块 | 改动 |
| --- | --- |
| `spec.py`（G1 模型构建、full self-collision、执行器推导） | 一行未改 |
| `motion.py`（LAFAN1 → 50 Hz 参考轨迹） | 一行未改 |
| `constants.py`（全部冻结数值） | 一行未改 |
| `manifest.py` / `protocol.py`（哈希冻结） | 一行未改 |
| `benchmark/protocol.json`、`benchmark/core.py`、`benchmark.py` | 一行未改 |
| `env.py` 的物理/奖励/终止 | 未改语义；仅把观测拼装抽成 `step_input()` + 保留 `observation()` 作为委托 |

`env.observation()` 仍然返回原来的 160 维向量，仍被原有测试逐位校验——它现在是「某一实现的编码」，而不是接口。

## 3. 最小推理接口

`benchmark/evaluator/policy_api.py`，两个方法，无基类、无注册表：

```python
class MyPolicy:
    def reset(self, context): ...   # 每条 clip 一次
    def act(self, step):
        return action               # (29,) 关节指令，50 Hz
```

* `ClipContext`：`clip`、`joint_order`、`default_joint_pos`、`action_scale`、`control_dt`、`reference`（只读）。
* `StepInput`：`step`、`reference_index`、`time_s`、`joint_pos`、`joint_vel`、`base_lin_vel`、`base_ang_vel`、`anchor_pos_b`、`anchor_ori_b`、`previous_action`、`reference`。
* `ReferenceClip`：整条参考轨迹（`joint_pos/joint_vel/root_pos_w/root_quat_w`，全部置为不可写）。给整条而非当前帧，是为了不惩罚用 look-ahead 的算法；所有提交看到的数据一致。
* 算法拿不到 `MjModel`/`MjData`/奖励项/终止函数——`StepInput` 只有本体感知与任务信息。
* 接入方式：`--policy-file path.py:make_policy`；有 `--checkpoint` 时工厂收到路径，Evaluator 只算 SHA256 交给记录。
* 示例：`examples/scripted_replayer.py`（约 40 行，可直接跑）。

**动作校验**（唯一由 Evaluator 强制的算法侧约束）：形状必须严格 `(29,)`、dtype 数值、全部有限，否则 `ActionFormatError` 并指出 clip 与步号。不做 clip / reshape / 静默修补。

## 4. 新增测试与运行结果

`tests/test_policy_interface.py`，17 个新用例：

| 要求 | 用例 |
| --- | --- |
| 不再要求 160 维观测 | `test_seven_dim_observation_is_enough` —— 7 维观测跑真机，断言宽度集合 `{7}` |
| 不再要求 `actor`/`critic`/MLP 键 | `test_checkpoint_without_actor_mlp_is_fine` —— 自定义 `.npz`（`policy/gain`）可用，同时断言 mjlab loader 会拒绝它（说明约束只属于该实现） |
| 简单策略能驱动真实 rollout | `test_own_history_buffer`、`test_cli_plugin_entry_point`、`test_interface_is_exactly_reset_and_act` |
| 非法动作被拒 | `ActionValidationTest` 7 例：宽 28/30/1、rank `(1,29)/(29,1)/(29,29)`、NaN/±inf、`None`/字符串/bool/dict |
| 现有能力不回归 | 见下方 40-clip 对比 |

运行结果：

* 全量测试 **76 passed**（38 evaluator + 21 validator + 17 interface），0 skipped。
* `freeze_manifest.py --check` → `OK: manifest matches the frozen hashes`，rc=0。
* **40-clip 回归逐条比对**：重构前后 `normalized_return`、`failed`、`alive_steps`、`fall_step`、`fall_terms` **完全一致**（0 差异），总步数同为 441,080，均值同为 `0.001993`。`reports/smoke_replay_40clips.json` 已更新为重构后产物。
* 新入口实跑：`--policy-file examples/scripted_replayer.py:make_policy` 2 条 clip，记录 `policy_kind = example-scripted-replayer`。
* 旧通路实跑：`--checkpoint checkpoints/smoke_fixture_actor.npz` 仍为 `mode=checkpoint`、真实 SHA `04638e77…`，门禁只剩「子集 + GPU」两项，符合预期。

## 5. 是否仍存在影响公平比较的问题

**有，且必须明说：**

1. **没有真实 checkpoint，所以没有真实成绩。** 本次只验证了机制与接口，未产生任何可提交记录，也没有把冒烟数字登记为成绩。
2. **跨训练模拟器的分数不是严格同物理的比较。** 观测与网络已开放，但所有提交仍在同一个 MuJoCo 模型里评分。在 IsaacLab / MJX 里训练的策略，sim-to-sim 差距会被计入成绩——这是被测差异的一部分，不是 Evaluator 的偏差，但读数字时要知道。
3. **参考动作整条可见。** 这是设计选择（不惩罚 look-ahead），代价是「背 40 条动作」在原则上可作为一种策略；由于任务是逐 clip 跟踪且数据公开，这不构成越权，但意味着极端过拟合不会被机制排除。
4. **`submission_errors` 依赖自报的 `policy_kind`。** 门禁能挡住「标了 smoke 的记录」和「没有 checkpoint 哈希的记录」，但无法证明提交的权重就是训练出来的那个——只记录该文件的 SHA256。
5. **CPU 后端。** `mujoco-warp` 仍只是记录版本、未接入 rollout，协议里的 GPU 字段在这台机器上无法满足。

### 尚未完成（明确列出）

* **真实 Policy 评估**：无 checkpoint，未跑。
* **CPU / MuJoCo Warp 后端一致性验证**：未做。同一策略在两后端上的逐 clip 差异尚未测量，因此当前不得宣称后端无关。
* **与官方 `evaluate.py` 的对齐**：未做，依赖第 1 项。
