# Ordinal-binding recurrent MEM 接入 GroundSG action 测试（2026-08-29）

## 结论

Ordinal-only MEM 在离线消融中提升了 PlaceOrder，但接入官方 GroundSG action 后为
**20/30（66.7%）**，低于原 causal-event MEM 的 **25/30（83.3%）**。它不应替换
当前 action MEM。

| MEM | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | 总计 |
|---|---:|---:|---:|---:|
| Oracle semantic region | 10/10 | 10/10 | 10/10 | 30/30（100.0%） |
| 原 causal-event MEM | 9/10 | 9/10 | 7/10 | 25/30（83.3%） |
| **Ordinal-only MEM** | **7/10** | **6/10** | **7/10** | **20/30（66.7%）** |

Ordinal-only 没有提升 Place action，并使 Unmask 降低 2 条、Swap 降低 3 条。

## 严格配对结果

两组使用同一批 test episode、seed 7、每条最多 1300 步、同一个官方
`symbolic-grounded-subgoal/79999` action checkpoint。

| 任务 | Ordinal 改善 | Ordinal 退化 | 净变化 |
|---|---|---|---:|
| VideoUnmask | 无 | ep1、ep7 | -2 |
| VideoUnmaskSwap | 无 | ep3、ep6、ep7 | -3 |
| VideoPlaceOrder | ep2 | ep7 | 0 |

PlaceOrder 的 ordinal loss 确实修复了 ep2 的 first-target region，但同时令 ep7 的
目标变为 missing，因而没有产生净 action 收益。

## MEM 与 action 的因果核验

逐条检查 predicted semantic region 与 oracle region：

- MEM 完整且 region 正确：20 条，action 成功 **20/20**；
- MEM region 错误或缺失：10 条，action 成功 **0/10**。

分任务也完全一致：Unmask 7/7、Swap 6/6、Place 7/7 的正确 region 均成功执行。
所以性能下降来自 ordinal checkpoint 改坏了 semantic memory，不是 action controller
的随机执行失败。

错误构成：

- Unmask：2 条错误 region，1 条目标缺失；
- Swap：4 条错误 region；
- PlaceOrder：3 条 queried ordinal 缺失。

## 为什么离线提升没有迁移到 action

1. 离线报告的是三个训练 seed 的数据集平均；闭环使用特定 checkpoint seed260909 和
   固定的 action test episode，存在明显 checkpoint/episode 方差。
2. Ordinal loss 通过共享 executor 反向传播。虽然提高平均 final query，却改变了
   Unmask/Swap 的视觉事件决策边界；离线均值没有暴露这组固定 episode 上的退化。
3. Place final exact 从三-seed均值看提高，但错误类型从“选错 region”部分转移为
   “输出 none”。对 action 来说二者都直接导致失败。
4. 当前 action bridge 已验证为近似确定性放大器：semantic region 正确就成功，错误就
   失败。因此不能用平均 query 指标代替逐 episode deployment-region accuracy。

## 最终决策

- 保留原 causal-event checkpoint 作为当前 action MEM：
  `/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/robomme_explicit_event_pooled_soft_causal_seed260908_260829`
- ordinal-only 保留为离线消融证据，不进入最终 action 模型；
- completeness-only 和 combined 更不进入 action 测试；
- 后续 checkpoint selection 应加入 deployment-matched episode-level region exact，
  并采用跨 seed ensemble/平均或先在固定 action demos 上无动作筛选，避免再次用平均
  final-query 指标误选 checkpoint。

## 产物

- 评测结果：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/ordinal-event-mem-action-test10-three-task-seed7-260829`
- 候选 checkpoint：
  `/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/robomme_explicit_event_ordinal_only_seed260909_260829`
- 对应离线消融：`docs/robomme_query_loss_factorial_ablation_260829.md`
