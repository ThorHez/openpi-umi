# PickXTimes No-Teacher 5.33% 结果解释与论文表述备忘录

日期：2026-08-30

> **2026-09-01 更新：** `5.33 ± 1.15%` 是旧
> `symbolic-grounded-subgoal/79999` 接口下的历史诊断结果。论文主表已改用
> FrameSamp+Modul 初始化、action-only recurrent MEM 的刷新结果
> **`36.00 ± 2.00%`（54/150）**。本文其余内容只用于解释旧系统为何失败，不能再按
> 第 10 节旧建议把 5.33% 填入主表。

## 1. 本文档要回答的问题

PickXTimes 的 No-Teacher 闭环成功率为 `5.33%`，明显低于另外三个 RoboMME
任务。本文档记录：

1. 这个数字是否是有效实验结果；
2. `5.33%` 可能由什么机制导致；
3. 当前实验能够支持什么论文结论；
4. 哪些结论仍需额外消融才能成立。

## 2. No-Teacher 的准确含义

当前 No-Teacher 不是完全没有标签，而是移除了训练期的特权中间轨迹监督，仅保留
episode 末尾的 terminal answer：

```text
L_no_teacher = L_terminal_answer
```

训练时未使用：

- 中间 `event_type/entity/region/swap_pair` 标签；
- 每个时刻的 semantic state；
- teacher latent、teacher delta 或 teacher readout；
- previous-state teacher forcing。

Student 仍然接收 RGB、proprio、prompt 和 12 帧 causal chunk，模型结构及
MEM-to-action 接口保持不变。因此论文中更准确的名称是：

> **w/o teacher trajectory distillation (terminal-only supervision)**

不建议仅写模糊的 `w/o Teacher`，否则读者可能误解为该模型完全没有监督。

## 3. 正式闭环结果

评测固定使用官方 `symbolic-grounded-subgoal/79999` action checkpoint，每条 episode
最多运行 1300 simulator steps，使用 action seeds 7、17、27，每个 seed 评测 50 条：

| Action seed | 成功数 | 成功率 |
|---:|---:|---:|
| 7 | 3/50 | 6.00% |
| 17 | 3/50 | 6.00% |
| 27 | 2/50 | 4.00% |
| **Mean ± SD** | **8/150** | **5.33 ± 1.15%** |

这个数字来自完整 `3 seeds × 50 episodes` 正式协议，不是 10 条 smoke test。三个
action seed 的结果相近，说明 `5.33%` 不是由某一个异常 seed 单独造成的。不过总成功数
只有 8，统计不确定性仍然较大；按 150 条运行作二项统计时，95% Wilson 区间约为
`[2.7%, 10.2%]`。

## 4. 一个看似矛盾但关键的现象

No-Teacher Pick MEM 的 memory-only 指标为：

| 指标 | 结果 |
|---|---:|
| Terminal answer exact（completed count + done） | 100.00 ± 0.00% |
| Transition state exact（完整 schema） | 0.34 ± 0.60% |
| Hold state exact | 19.77 ± 1.57% |
| All-state exact | 17.03 ± 1.26% |
| Full final-state exact | 2.22 ± 3.85% |
| Full-sequence exact | 0.00 ± 0.00% |

模型能够准确预测 episode 最终应该完成几次，却几乎不能在线产生正确的状态转移。这说明
terminal answer 可以通过 prompt 或终点特征被 shortcut 学到，但该能力不等价于可供
action controller 使用的 causal memory。

例如要求重复三次时，terminal-only 监督只明确约束：

```text
initial ------------------------------------> completed_count = 3
```

它没有告诉模型三个增量各自应该发生在哪个视觉事件边界。动作闭环真正需要的是：

```text
0 --第一次放置完成--> 1 --第二次放置完成--> 2 --第三次放置完成--> 3
```

## 5. 5.33% 的主要可能原因

### 5.1 缺少 event-aligned completed-count 监督（当前证据最强）

PickXTimes 要求 MEM 判断每一次 pick-place 是否真正完成，并只在完成事件发生时把
`completed_count` 加一。Terminal-only loss 没有提供这些中间边界，长时程梯度必须从
episode 末尾反推数百帧中的更新时刻，时间信用分配高度欠定。

正式 150 条闭环 trace 显示：

- 116/150（77.33%）episode 在 oracle 仍处于第一次 pick 时就提前增加 count；
- 46/150（30.67%）episode 在第一次 pick 时就提前输出 done；
- 第一次 pick 阶段的 1276 个 action query 中，29.39% 已提前增加 count；
- 同一阶段有 11.05% query 已提前输出 done。

因此最直接的失败机制是**事件尚未完成，记忆状态已经提前推进**。

### 5.2 递归错误会在闭环中累计

一旦 MEM 过早从 `count=0` 更新到 `count=1`，后续窗口会基于错误的前一状态继续递归。
错误状态不仅影响下一次更新，还会立即改变 action expert 读取到的 latent，使机器人开始
执行错误 ordinal 的动作。Terminal-only 训练没有逐步纠正这种 free-rollout 漂移。

### 5.3 PickXTimes 对中间轨迹的依赖强于 region 任务

VideoUnmask、VideoUnmaskSwap 和 VideoPlaceOrder 更接近“观察后输出当前 region/state”。
PickXTimes 则要求动作执行、事件判断、计数更新和停止决策反复交替：

```text
pick -> place -> count+1 -> pick -> place -> count+1 -> ... -> press
```

因此相同的 terminal-only 缺陷在 PickXTimes 中会被重复放大。它不只要求最终 readout
正确，还要求整个在线 memory trajectory 与动作阶段同步。

### 5.4 action expert 对错误 ordinal latent 很敏感

MME action checkpoint 持续读取 MEM 映射后的 semantic latent。即使最终 count 正确，只要
中间把第 `k` 次误写成第 `k+1` 次，action expert 接收到的条件就偏离其训练分布。闭环成功
要求一系列中间状态连续正确，而不是只要求最后一个 query 正确。

### 5.5 Phase tracking 不是主要解释

我们额外执行了 `No Teacher + oracle phase` 控制：保留 MEM 的 completed-count 预测，仅用
simulator oracle 修正 `pick/place/press` phase。

| 条件 | Phase exact | Count exact | 闭环成功率 |
|---|---:|---:|---:|
| No Teacher | 42.11% | 50.11% | 5.33%（8/150） |
| + oracle phase | **100.00%** | **61.28%** | 6.67%（10/150） |

Phase 被完全修复后，闭环仅提高 `+1.33 pp`；seed 27 的两个 discordant pair 做 exact
two-sided McNemar 检验为 `p=0.5`，没有可靠提升证据。这排除了“Pick 下降主要因为还要
预测 pick/place/press phase”这一简单解释，剩余核心仍是 event-aligned count。

## 6. 为什么 Teacher 能带来显著改善

Teacher 的主要价值不是给出更强的最终答案，而是将稀疏 terminal supervision 分解为：

- 当前窗口是否包含真实事件；
- 事件发生前的 semantic state；
- 事件 payload；
- 事件发生后的 semantic state；
- hold window 应保持不变的目标；
- 与这些语义状态一致的 latent trajectory。

因此 Teacher 同时缓解时间信用分配、更新时机判断、递归状态漂移和 action-conditioning
分布错位。对于 PickXTimes，它把一个欠定问题：

```text
从最终 count 反推出所有中间递增时刻
```

转换为局部、可学习的状态转移问题：

```text
(previous state, current causal observation) -> event/no-event -> next state
```

## 7. 当前实验能够支持的论文结论

### 可以支持

1. 去除 event-aligned teacher trajectory supervision，只保留 terminal-state supervision，
   会使 PickXTimes 闭环成功率降至 `5.33 ± 1.15%`。
2. Terminal answer accuracy 不足以衡量 recurrent memory 是否可用于动作；本实验中
   terminal answer 为 100%，但 transition accuracy 接近 0，闭环成功率仅 5.33%。
3. PickXTimes 的主要失败来自错误事件提交和 completed-count 提前推进，而不只是 phase
   tracking 错误。
4. 长时程重复操作需要 event-aligned recurrent trajectory supervision。

### 当前不能直接支持

1. “Teacher 模型结构本身是全部提升的来源。”
2. “任何不使用 Teacher 的方法都无法学会 PickXTimes。”
3. “Teacher 比直接使用同等密度的 simulator GT 更优。”
4. “Full model 在 PickXTimes 上相对 5.33% 的正式增益已经确定。”

第 4 点尤其需要注意：当前 Full Pick 只有既有的 10-episode smoke `4/10`，尚未完成与
No-Teacher 相同的 `3 action seeds × 50 episodes` 正式评测，不能把 `4/10` 和 `8/150`
直接作为正式配对比较。

## 8. 仍需补充的关键因果消融

为了区分“Teacher 模型必要”与“密集状态转移监督必要”，应补充：

| 条件 | 监督 | 能回答的问题 |
|---|---|---|
| Terminal-only No Teacher | 只有最终答案 | 当前已有基线 |
| Direct dense GT, no Teacher | 直接逐事件 GT，不经过 Teacher | 密集监督是否已足够 |
| Teacher distillation | Teacher trajectory/latent | Teacher 是否有额外表征价值 |

解释规则：

- 若 direct dense GT 接近 Teacher，主要贡献应表述为 event-aligned dense supervision；
- 若 direct dense GT 明显低于 Teacher，才支持 Teacher 的软分布或 latent trajectory 具有
  额外价值；
- 若 direct dense GT 高于 Teacher，应把 Teacher 定位为降低特权标注成本的可扩展近似方案，
  而非性能上界。

## 9. 推荐论文表述

### 中文

> 去除事件对齐的 Teacher trajectory supervision、仅保留 episode 终态标签后，
> PickXTimes 的闭环成功率下降至 5.33%。值得注意的是，该模型的终态答案准确率仍为
> 100%，但 transition-state accuracy 仅为 0.34%，说明终态监督容易产生终点 shortcut，
> 无法确定重复操作中每一次状态递增的时机。进一步使用 oracle 修正动作 phase 后，成功率
> 仅提升至 6.67%，表明主要瓶颈是 completed-count 与真实 pick-place 完成事件未对齐，
> 而非简单的 phase classification 错误。

### English

> Removing event-aligned teacher trajectory supervision and retaining only the
> episode-level terminal target reduces the PickXTimes closed-loop success rate
> to 5.33%. Despite achieving 100% terminal-answer accuracy, the model obtains
> only 0.34% transition-state accuracy, indicating that terminal supervision
> encourages an endpoint shortcut but does not identify when individual
> pick-place completions should advance the recurrent count. Correcting the
> action phase with an oracle improves success only to 6.67%, suggesting that
> the dominant failure is event-count misalignment rather than phase
> classification alone.

## 10. 主表与附录建议

- 主表行名使用 `Ours w/o Teacher (action-only recurrent)`；
- 主表报告刷新结果 `36.00 ± 2.00%`，并明确来自 3 个 action seeds、总计 `54/150`；
- 旧 `5.33 ± 1.15%（8/150）` 只放入历史诊断附录，并注明其 action/MEM 接口不同；
- Oracle Phase 控制放入诊断消融表，不作为主方法变体；
- Full Pick 与刷新 No-Teacher 均已完成 `3 × 50`，但两者 updater 架构不完全同构；
  `82%-36%=46 pp` 只报告为协议匹配的系统差距，不写成严格 loss-only teacher 因果增益。

## 11. 对应实验产物

- 四任务 No-Teacher 汇总：
  `docs/robomme_four_task_no_teacher_results_260830.md`
- Oracle Phase 控制：
  `docs/pickxtimes_no_teacher_oracle_phase_control_260830.md`
- No-Teacher Pick checkpoints：
  `checkpoints/robomme_no_teacher_pick_seed{260923,260924,260925}_260830/`
- 正式 No-Teacher 闭环：
  `../robomme_policy_learning/runs/evaluation/no-teacher-pick-action-test50-seed{7,17,27}-260830/`
- 刷新后的主表 No-Teacher 闭环与汇总：
  `../robomme_policy_learning/runs/evaluation/pick-no-teacher-recurrent-3seed50-260901/summary.json`
- 正式 Oracle Phase 闭环：
  `../robomme_policy_learning/runs/evaluation/no-teacher-pick-oracle-phase-action-test50-seed{7,17,27}-260830/`
