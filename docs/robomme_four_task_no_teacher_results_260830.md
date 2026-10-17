# RoboMME 四任务 No-Teacher 消融结果（2026-08-30）

> **2026-09-01 接线审计更正：** VideoPlaceOrder No-Teacher 的旧值
> `86.0% (129/150)` 无效。`learned_event_mem` 读取独立的
> `region_grounding_learned_event_training_dir`，旧启动器只覆盖 causal 路径，因而
> 回退加载了 Full/Teacher checkpoint。修正后的正式结果为
> **`28.0 ± 0.0% (42/150)`**，且全部 150 条 trace 均核验为 No-Teacher checkpoint。

> **2026-09-01 最终刷新：** 四个任务均已按当前 deployable 流水线重跑。
> 论文主表 No-Teacher 行为 PickXTimes `36.0 ± 2.0%`、VideoUnmask
> `26.0 ± 0.0%`、VideoUnmaskSwap `20.7 ± 1.2%`、VideoPlaceOrder
> `28.0 ± 0.0%`，合计 **166/600（27.67 ± 0.58%）**。下文保留的旧 5.33%、
> 34.0%、24.0%、32.7% 仅是历史诊断，不再进入主表。

## 1. 实验定义

本实验严格移除训练期的特权轨迹 teacher，只保留 episode terminal answer：

```text
L_no_teacher = L_terminal_answer
```

训练未使用中间 `event_type/entity/region/swap_pair`、逐步 semantic state、teacher
latent/delta/readout 或 previous-state teacher forcing。模型结构、RGB/proprio/prompt、
12 帧 causal chunk、训练 episode 和 MEM-to-action 接口不变。

前三个 region 任务使用原 `pooled_soft_causal` explicit-event MEM；PickXTimes 使用原
unified semantic-feedback MEM。两类模型分别训练 3 个 seed，并只根据 dev terminal
指标选择用于 action 的 checkpoint：

- region：seed 260920/260921/260922，选择 260921；
- Pick：seed 260923/260924/260925，选择 260925。

Action 固定使用官方 `symbolic-grounded-subgoal/79999` checkpoint。正式协议为每任务
3 action seeds（7、17、27）× 50 test episodes、每条最多 1300 simulator steps。

## 2. Memory-only 结果

### Region MEM：3 个训练 seed 的 locked-test 均值

| Metric | No Teacher |
|---|---:|
| VideoUnmask final query | 44.44 ± 15.40% |
| VideoUnmaskSwap final query | 20.51 ± 8.01% |
| VideoPlaceOrder final query | 35.56 ± 15.40% |
| Transition state exact | 13.91 ± 2.53% |
| Hold state exact | 15.80 ± 9.56% |
| All-state exact | 15.55 ± 7.96% |
| Hold false-positive rate | 99.28 ± 1.25% |
| Full-update recall | 6.71 ± 4.70% |

No-Teacher region MEM 偶尔能猜中最终 query，但几乎在每个 hold window 都提交更新，
没有形成可用的 causal state trajectory。

### PickXTimes：3 个训练 seed 的 locked-test 均值

| Metric | No Teacher |
|---|---:|
| Terminal answer exact（completed count + done） | 100.00 ± 0.00% |
| Transition state exact（全 schema） | 0.34 ± 0.60% |
| Hold state exact | 19.77 ± 1.57% |
| All-state exact | 17.03 ± 1.26% |
| Full final-state exact | 2.22 ± 3.85% |
| Full-sequence exact | 0.00 ± 0.00% |

Pick 的 terminal answer 达到 100%，但在线 transition 几乎为零。模型学到的是终点
shortcut，不是边观察边更新的 memory。

## 3. 10-episode 接口 smoke

固定 action seed 7：

| Task | Success |
|---|---:|
| PickXTimes | 0/10 = 0% |
| VideoUnmask | 4/10 = 40% |
| VideoUnmaskSwap | 3/10 = 30% |
| VideoPlaceOrder | 3/10 = 30% |

smoke 确认 checkpoint、MEM bridge 和 MME action 均可正常加载执行，随后固定配置进入
正式评测；smoke episode 不并入正式统计。

## 4. 原始系统的正式闭环成功率（Pick 5.33% 不再用于主表）

| Task | seed 7 | seed 17 | seed 27 | Mean ± SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| PickXTimes | 3/50（6%） | 3/50（6%） | 2/50（4%） | **5.33 ± 1.15%** | **8/150** |
| VideoUnmask | 17/50（34%） | 17/50（34%） | 17/50（34%） | **34.00 ± 0.00%** | **51/150** |
| VideoUnmaskSwap | 12/50（24%） | 12/50（24%） | 12/50（24%） | **24.00 ± 0.00%** | **36/150** |
| VideoPlaceOrder | 17/50（34%） | 16/50（32%） | 16/50（32%） | **32.67 ± 1.15%** | **49/150** |
| **Four-task macro / total** | 49/200（24.5%） | 48/200（24.0%） | 47/200（23.5%） | **24.00 ± 0.50%** | **144/600** |

该原始系统的四任务汇总为 **24.0%**；它已被下节刷新结果替代，不再作为主表数字。

### 4.1 第一次刷新（仅 Pick；已被 4.2 替代）

| Task | seed 7 | seed 17 | seed 27 | Mean ± SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| PickXTimes（refreshed action-only recurrent） | 19/50（38%） | 18/50（36%） | 17/50（34%） | **36.00 ± 2.00%** | **54/150** |
| VideoUnmask（原正式结果） | 17/50（34%） | 17/50（34%） | 17/50（34%） | **34.00 ± 0.00%** | **51/150** |
| VideoUnmaskSwap（原正式结果） | 12/50（24%） | 12/50（24%） | 12/50（24%） | **24.00 ± 0.00%** | **36/150** |
| VideoPlaceOrder（原正式结果） | 17/50（34%） | 16/50（32%） | 16/50（32%） | **32.67 ± 1.15%** | **49/150** |
| **Four-task macro / total** | 65/200（32.5%） | 63/200（31.5%） | 62/200（31.0%） | **31.67 ± 0.76%** | **190/600** |

刷新后的 Pick 条件与 Full Pick 使用相同 70 个训练 episode、34,214 个 action
timestep、官方 FrameSamp+Modul 初始化、3,000 步预算、12-step causal sampling，以及
相同 3×50 闭环协议。训练不使用 teacher event/state/latent、SimpleSG 或 GroundSG，
只用 action flow-matching loss 更新 128×64 recurrent MEM。它是当前 Pick 主表应使用的
No-Teacher 基线。

需要注意：该模型的 neural RMT updater 与 Full 的 explicit-event stack 并非完全同构，
因此 `82%-36%=46 pp` 是协议匹配的系统差距，不应写成严格的 loss-only teacher 因果
效应。本小节的 region 数字来自旧流水线，不再进入主表。

### 4.2 当前正式主表 No-Teacher 行（四任务均刷新）

| Task | seed 7 | seed 17 | seed 27 | Mean ± SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| PickXTimes | 19/50（38%） | 18/50（36%） | 17/50（34%） | **36.00 ± 2.00%** | **54/150** |
| VideoUnmask | 13/50（26%） | 13/50（26%） | 13/50（26%） | **26.00 ± 0.00%** | **39/150** |
| VideoUnmaskSwap | 10/50（20%） | 11/50（22%） | 10/50（20%） | **20.67 ± 1.15%** | **31/150** |
| VideoPlaceOrder | 14/50（28%） | 14/50（28%） | 14/50（28%） | **28.00 ± 0.00%** | **42/150** |
| **Four-task macro / total** | 56/200（28.0%） | 56/200（28.0%） | 54/200（27.0%） | **27.67 ± 0.58%** | **166/600** |

三个 region 对照保留当前模型结构、observable execution FSM、unique-anchor grounding，
以及 Swap 的 RGB permutation correction。训练只使用 episode terminal answer；不使用
event/state trajectory、teacher forcing、teacher checkpoint 初始化或轨迹指标选点。正式
450 条 region rollout 的 18 个 evaluator/policy 日志没有 traceback、OOM 或 runtime error。

## 5. 与当前 Full model 的正式配对比较

当前 Full model 已有相同 3×50 协议的三个 region 任务结果。配对比较为：

| Task | Ours w/o Teacher | Ours | Teacher gain |
|---|---:|---:|---:|
| VideoUnmask | 26.00 ± 0.00% | 90.00 ± 7.21% | **+64.00 pp** |
| VideoUnmaskSwap | 20.67 ± 1.15% | 92.67 ± 5.77% | **+72.00 pp** |
| VideoPlaceOrder | 28.00 ± 0.00% | 86.00 ± 0.00% | **+58.00 pp** |
| **Three-task macro** | **24.89%** | **89.56%** | **+64.67 pp** |

旧版 paired McNemar 数字对应已废弃的 region checkpoint，不能套用到本次刷新结果。
若论文需要配对显著性，应从两组当前 `progress.json` 按相同 episode ID 重新计算。
VideoPlaceOrder 修正后为 42/150，对照 Full 的 129/150，支持 teacher 带来闭环提升。

PickXTimes 的最新 teacher-distilled MEM 已在相同 test episode、3 action seeds × 50 条、
1300 步上限下完成正式评测，结果为 44/50、41/50、38/50，即
**82.00 ± 6.00%（123/150）**。2026-09-01 已补跑 FrameSamp+Modul 初始化、相同训练
数据和 3,000 步预算的 action-only recurrent No-Teacher 条件，得到 19/50、18/50、
17/50，即 **36.00 ± 2.00%（54/150）**。因此旧接口的 5.33% 不再进入主表。

刷新条件与 Full 的 action 初始化和评测协议匹配，但 neural RMT updater 与 Full 的
explicit-event stack 并非完全同构，所以 `82%-36%=46 pp` 应表述为协议匹配的系统
差距，不能声称为严格 loss-only 的 teacher 因果增益。

## 6. 闭环 trace 错误归因

以下 region trace 统计来自 2026-08-30 的旧 No-Teacher checkpoint，仅用于历史故障
诊断，不代表 4.2 的刷新 checkpoint。每个 chunk 最多有两个 micro-event，写入密度为
`committed_event_count / (2 × chunks)`：

| Task | No Teacher write density | Full write density | No Teacher mean commits/episode |
|---|---:|---:|---:|
| VideoUnmask | 63.50% | 17.33% | 7.62 |
| VideoUnmaskSwap | 96.09% | 16.13% | 24.36 |
| VideoPlaceOrder | 100.00% | 2.59% | 150.44 |

No-Teacher 尤其在 PlaceOrder 上几乎对每个 chunk 的两个 micro-event 都执行写入，而 Full
模型只在少数真实事件边界更新。这与 memory-only 的 99.28% hold FPR 一致。

对 Pick 正式 150 条、2531 个 action-query trace 统计：

- 116/150（77.33%）episode 在 oracle 仍处于第一次 pick 时提前增加 completed count；
- 46/150（30.67%）episode 在 oracle 仍处于第一次 pick 时提前输出 done；
- 第一次 pick 阶段的 1276 个 query 中，29.39% 已提前增加 count，11.05% 已输出 done。

因此闭环失败的直接机制是错误事件提交和提前状态推进，而不是 action backbone 无法读取
正确 memory。

## 7. 结论

No-Teacher 不是简单地降低最终分类精度，而是破坏了在线 causal credit assignment：

- 旧 region checkpoint 的 FPR 接近 100%，说明 terminal loss 很难决定应在哪个 chunk 更新；
- Pick 能以 100% 预测最终 count/done，但 transition 约 0%，证明 terminal answer 可以被
  shortcut 学到，却不能产生动作所需的中间状态；
- 刷新后 Unmask 与 Swap 分别下降 64.0 和 72.0 个点，仍支持 trajectory teacher 对
  在线 identity--region binding 很关键；
- PlaceOrder 从 No-Teacher 的 42/150 提升到 Full 的 129/150，说明 terminal answer
  无法单独训练出稳定的 ordinal updater，teacher 是该任务当前性能的重要来源；
- 当前四任务 task-specific No-Teacher 汇总为 27.67%（166/600），不是旧的 42.17%。

因此 teacher 的主要价值体现在需要在线 identity--region 或 ordinal--region 绑定的任务：
它把长时程 terminal supervision 分解成可学习的事件边界、payload 和状态转移轨迹。
不过该行仍是 task-specific system comparison，不能把差距全部解释成单一 loss 的
严格因果效应，也不能无条件推广到所有记忆任务或所有执行接口。

## 8. 产物

训练实现：

- `scripts/mem/train_robomme_explicit_event_bottleneck_ablation.py`
- `scripts/mem/train_pickxtimes_semantic_feedback_student.py`

Action 启动脚本：

- `../robomme_policy_learning/scripts/run_pick_semantic_feedback_action.sh`
- `../robomme_policy_learning/scripts/run_no_teacher_pick_action_3seed_50ep.sh`
- `../robomme_policy_learning/scripts/run_causal_mem_action_3seed_50ep.sh`
- `../robomme_policy_learning/scripts/run_no_teacher_region_refresh_3seed50_260901.sh`

No-Teacher checkpoints：

- `checkpoints/robomme_no_teacher_region_seed{260920,260921,260922}_260830/`
- `checkpoints/robomme_no_teacher_pick_seed{260923,260924,260925}_260830/`
- `checkpoints/robomme_no_teacher_native_single_strict_seed260908_260901/`
- `checkpoints/videoplaceorder_no_teacher_learned_event_ordinal_seed260831_260901/`

正式闭环结果：

- `../robomme_policy_learning/runs/evaluation/causal-event-mem-action-test50-three-task-seed{7,17,27}-no-teacher-260830/`
- `../robomme_policy_learning/runs/evaluation/no-teacher-pick-action-test50-seed{7,17,27}-260830/`
- `../robomme_policy_learning/runs/evaluation/no-teacher-region-refresh-3seed50-260901/summary.json`

当前主表 Pick 与刷新 region 合计 600 个 progress entries；刷新 region 的 450 条均完整，
launcher exit code 为 0，18 个最终日志的 error matches 为 0。
