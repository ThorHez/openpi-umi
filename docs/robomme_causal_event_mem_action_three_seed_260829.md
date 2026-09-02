# 原版 causal-event MEM 三 action seeds 闭环测试（2026-08-29）

## 结论

原版 `pooled_soft_causal` MEM 接官方 `symbolic-grounded-subgoal/79999` action
checkpoint，在 action seeds 7、17、27 上结果完全一致：每个 seed 都为 **25/30
（83.3%）**。名义总计 **75/90**，任务成功率为 Unmask 90%、Swap 90%、Place
70%，seed 间标准差均为 0。

| Action seed | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | 总计 |
|---:|---:|---:|---:|---:|
| 7 | 9/10 | 9/10 | 7/10 | 25/30 |
| 17 | 9/10 | 9/10 | 7/10 | 25/30 |
| 27 | 9/10 | 9/10 | 7/10 | 25/30 |
| **汇总** | **27/30** | **27/30** | **21/30** | **75/90（83.3%）** |

## 实验控制

- MEM checkpoint：
  `/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/robomme_explicit_event_pooled_soft_causal_seed260908_260829`
- Action checkpoint：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999`
- 数据：RoboMME test，每任务 episode 0–9；
- 最大执行步数：1300；
- GroundSG 接口、region grounding、episode 顺序完全相同；
- 只改变 action policy seed：7、17、27。

## 不是简单的字节级重复

三个 seed 的 rollout trace 长度均值分别为 12.53、11.70、12.43 个 action chunks，
多数 episode 的 grounded subgoal/action trace 并不相同，说明 action seed 确实改变了执行
轨迹；但所有 episode 的最终成败保持一致。因此标准差为 0 表示该 MEM region 条件下的
动作执行对这三个 policy seeds 稳定，而不是简单复用了同一个输出文件。

## MEM 与 action 的稳定因果关系

每个 seed 都满足：

- MEM region 完整且正确：25 条，action 成功 25/25；
- MEM region 错误或缺失：5 条，action 成功 0/5。

三 seed 合并：semantic region 正确时 action 成功 **75/75**，错误时成功 **0/15**。
所以当前 83.3% 的上限由 MEM 的 25/30 deployment-region exact 决定，而不是 action
controller 的 seed 方差。

## 统计解释

75/90 可以用于描述 action stochastic repeat 的稳定性，但三组测试共享相同的 30 个
environment episodes，不能当作 90 个独立场景来缩小环境泛化置信区间。论文主结果仍应
表述为：

- 30 个唯一 test episodes 上 25/30；
- 对三个 action seeds 重复后均为 25/30；
- action-seed standard deviation 为 0。

若要形成更强的论文主表，应扩展唯一 episode 数，而不是继续增加相同 episode 上的
policy seeds。

## 最终判断

原版 causal MEM 可以作为当前正式 action 模型：

- 相比旧 recurrent MEM 的 6/30，提升到 25/30；
- 达到 oracle region 30/30 的 83.3%；
- 三个 action seeds 无性能波动；
- 剩余失败完全可归因于 MEM semantic region。

下一步优先跑每任务更多唯一 test episodes，并继续保留这版 checkpoint 作为不可回退的
action baseline。

## 50-Episode 正式协议更新

后续已完成每任务50条、三个action seeds的统一评测。最终分任务均值为 Unmask
72.0±0.0%、Swap 70.0±2.0%、Place 70.67±1.15%，总计319/450，宏平均
70.89±0.77%。因此本文件的25/30结果只保留为smoke记录；正式结果以
`docs/robomme_causal_event_mem_action_3seed50_260829.md`为准。

## 结果目录

- seed 7：`robomme_policy_learning/runs/evaluation/causal-event-mem-action-test10-three-task-seed7-260829`
- seed 17：`robomme_policy_learning/runs/evaluation/causal-event-mem-action-test10-three-task-seed17-260829`
- seed 27：`robomme_policy_learning/runs/evaluation/causal-event-mem-action-test10-three-task-seed27-260829`
