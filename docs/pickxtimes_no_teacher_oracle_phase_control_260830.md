# PickXTimes No-Teacher + Oracle Phase 控制实验（2026-08-30）

## 目的

检验 PickXTimes 的 No-Teacher action success 明显低于三个 region 任务，是否主要因为
Pick MEM 还承担了 `pick/place/press` phase tracking，而 region 任务由 simulator simple
phase oracle 提供动作阶段。

## 控制条件

固定 No-Teacher checkpoint：

`robomme_no_teacher_pick_seed260925_260830/best/params`

保持 No-Teacher MEM 预测的 `completed_count`；target color 和 required count 仍来自 prompt。
Simulator oracle 只将当前 simple-SG 映射成：

```text
pick  -> holding=0, ready=0, done=0
place -> holding=1, ready=0, done=0
press -> holding=0, ready=1, done=0
```

随后在 MME latent codebook 中，仅依据 MEM 的 completed-count logits 选择满足 oracle phase
的合法状态。实现不解析 simple-SG 中的 `first/second/...` 文本，因此不会显式把 oracle
ordinal 作为 count 输入；但 phase 的合法状态集合会约束 count 范围，press phase 也必然
意味着所有要求次数已经完成。

Action checkpoint、episode、1300-step 上限和 seeds 均与原 No-Teacher 正式实验相同。

## Smoke

固定 seed 7、相同前 10 条：

| Condition | Success |
|---|---:|
| No Teacher | 0/10 |
| No Teacher + oracle phase | 0/10 |

## 正式 3 seeds × 50 episodes

| Condition | seed 7 | seed 17 | seed 27 | Mean ± SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| No Teacher | 3/50（6%） | 3/50（6%） | 2/50（4%） | 5.33 ± 1.15% | 8/150 |
| + oracle phase | 3/50（6%） | 3/50（6%） | 4/50（8%） | **6.67 ± 1.15%** | **10/150** |
| Delta | 0 pp | 0 pp | +4 pp | **+1.33 pp** | +2/150 |

相同 episode ID 的配对变化：

| seed | phase-only success | baseline-only success | both success |
|---:|---:|---:|---:|
| 7 | 1 | 1 | 2 |
| 17 | 0 | 0 | 3 |
| 27 | 2 | 0 | 2 |

seed 27 只有两个 discordant pair，exact two-sided McNemar `p=0.5`；其余两个 seed
没有净配对增益。因此 +1.33 pp 不能解释为可靠提升。

## Trace 诊断

在能够从 oracle 文本还原当前 ordinal 的 action-query rows 上：

| Condition | Phase exact | Count exact | Count ahead | Count behind |
|---|---:|---:|---:|---:|
| No Teacher | 42.11% | 50.11% | 49.89% | 0.00% |
| + oracle phase | **100.00%** | **61.28%** | 24.88% | 13.84% |

oracle phase 完全修复了 holding/ready/done，并通过合法状态过滤减少了一部分 count 提前；
但仍有 38.72% 的 query count 不正确。错误 count 继续让 latent codebook 选择错误的重复
ordinal，所以 phase 修正没有转化成明显 action gain。

## 结论

Pick No-Teacher 的低成功率不能主要归因于缺少 simple phase。即使 phase exact 达到
100%，正式 success 仍只有 6.67%。主要剩余瓶颈是 event-aligned completed-count：
terminal-only loss 可以从 prompt 学会最终 count，却不能学习每次 pick/place 完成后何时递增。

这个控制实验支持以下论文解释：

> The large PickXTimes degradation is not explained by phase tracking alone.
> Oracle phase correction makes the phase state exact, but action success remains
> nearly unchanged because terminal-only training fails to align the recurrent
> count with individual pick-place completion events.

## 产物

- 实现：`../robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- 参数：`semantic_feedback_oracle_phase`
- smoke：`../robomme_policy_learning/runs/evaluation/no-teacher-pick-oracle-phase-smoke10-seed7-260830/`
- 正式结果：`../robomme_policy_learning/runs/evaluation/no-teacher-pick-oracle-phase-action-test50-seed{7,17,27}-260830/`
