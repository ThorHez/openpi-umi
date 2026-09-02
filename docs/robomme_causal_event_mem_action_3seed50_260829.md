# 原版 causal MEM：3 seeds × 50 episodes 统一 Action 评测（2026-08-29）

## 统一协议

- MEM：`pooled_soft_causal` seed260908，best step 1300；
- Action：官方 `symbolic-grounded-subgoal/79999`；
- Action seeds：7、17、27；
- 任务：VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder；
- 每个任务、每个 seed：test episode 0–49，共50条；
- 每条最多1300步；
- 每任务150次action trials，总计450次；
- 指标：各seed成功率、跨seed均值±样本标准差、合并成功次数。

## 分任务成功率

| 任务 | Seed 7 | Seed 17 | Seed 27 | 3-seed均值±标准差 | 合并计数 |
|---|---:|---:|---:|---:|---:|
| VideoUnmask | 36/50（72%） | 36/50（72%） | 36/50（72%） | **72.0±0.0%** | **108/150** |
| VideoUnmaskSwap | 35/50（70%） | 34/50（68%） | 36/50（72%） | **70.0±2.0%** | **105/150** |
| VideoPlaceOrder | 36/50（72%） | 35/50（70%） | 35/50（70%） | **70.67±1.15%** | **106/150** |
| **三任务宏平均/总计** | **107/150（71.33%）** | **105/150（70.0%）** | **107/150（71.33%）** | **70.89±0.77%** | **319/450** |

因此论文和后续实验统一使用的 headline 应更新为：原版 causal MEM + 官方 action 在
3 seeds × 50 episodes 标准下为 **70.89±0.77%**，而不是前10条得到的83.3%。

## Seed 稳定性

按50个唯一episode统计跨seed一致性：

| 任务 | 三seed全部成功 | 三seed全部失败 | Seed-sensitive episode |
|---|---:|---:|---:|
| VideoUnmask | 36 | 14 | 0 |
| VideoUnmaskSwap | 34 | 14 | 2（ep18、ep39） |
| VideoPlaceOrder | 35 | 14 | 1（ep11） |

150个唯一 task-episodes 中只有3个受action seed影响。绝大多数成败由episode和MEM
semantic region决定，而不是policy采样波动。

## 难度分解（合并三个seeds）

| 任务 | Easy | Medium | Hard |
|---|---:|---:|---:|
| VideoUnmask | 63/78（80.8%） | 27/36（75.0%） | 18/36（50.0%） |
| VideoUnmaskSwap | 54/78（69.2%） | 32/36（88.9%） | 19/36（52.8%） |
| VideoPlaceOrder | 60/78（76.9%） | 24/36（66.7%） | 22/36（61.1%） |

主要性能损失集中在 hard split，尤其是 Unmask 和 Swap。这比继续调整已经稳定的
action seed 更值得优先分析。

## 与10-episode结果的关系

之前每任务前10条的结果为 Unmask 90%、Swap 90%、Place 70%，总计83.3%。扩展到
完整前50条后：

- Unmask：90% → 72%，下降18 pp；
- Swap：90% → 70%，下降20 pp；
- Place：70% → 70.67%，基本一致；
- 总体：83.3% → 70.89%。

说明前10条对 Unmask/Swap 明显偏容易。后续所有模型比较必须固定使用相同的
3-seed×50-episode协议，不能再将10条smoke结果作为主成功率。

## 结论

原版 causal MEM 仍是当前最好的 action MEM，但其可靠的统一成功率是约71%，不是83%。
下一步应针对每个任务14个左右的跨seed一致失败episode分析semantic region错误，并以
3seed×50ep作为所有新方案的替换门槛。

## 原始结果

- Seed 7：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/causal-event-mem-action-test50-three-task-seed7-260829`
- Seed 17：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/causal-event-mem-action-test50-three-task-seed17-260829`
- Seed 27：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/causal-event-mem-action-test50-three-task-seed27-260829`
