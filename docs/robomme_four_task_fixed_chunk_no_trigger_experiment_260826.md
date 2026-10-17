# RoboMME 固定分块、无 Event Trigger 的统一 MEM 蒸馏实验

日期：2026-08-26

## 实验目的

验证是否可以完全移除滑动窗口和显式 event trigger，仅用固定、不重叠的连续视频 chunk 驱动统一 recurrent MEM：

```text
continuous video
  -> non-overlapping 12-frame chunks (stride=12)
  -> shared visual encoder
  -> shared recurrent memory updater
  -> memory state
```

模型没有 event detector、event type/argument 输入或 task-specific head。唯一 mask 是变长 episode 的 padding mask。

## 数据与 Teacher 对齐

- train/dev/test：280/60/60 episodes，四任务严格平衡；
- 最大序列长度：96 chunks；
- train 有效 chunks：8098；
- 状态变化 chunks：1074（13.3%）；
- 状态不变 chunks：7024（86.7%）。

每个 chunk 的监督目标是该 chunk 因果结束时的 canonical teacher memory/state。没有变化的 chunk 目标保持不变；发生变化的 chunk 应更新到新的 canonical state。

为避免 86.7% no-change chunk 主导训练，状态变化 chunk 使用 6× state weight。该权重只影响训练 loss，不改变推理结构。

## 训练设置

- chunk/stride：12/12 frames；
- max recurrent steps：96；
- batch size：4，每 batch 四任务平衡；
- steps：2000；
- peak LR：3e-4；
- teacher target：canonical memory；
- frozen teacher readout state loss；
- best step：1700。

96 次更新通过 `nn.scan` 共享同一组 updater 参数，因此序列变长没有增加模型参数量。

## Held-out Test

| Task | Field | All-chunk state | Change-chunk state | No-change state | Final | Full sequence |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 79.5% | 21.2% | 6.7% | 23.5% | 6.7% | 0.0% |
| PickXtimes | 79.3% | 18.6% | 7.2% | 20.5% | 0.0% | 0.0% |
| VideoPlaceOrder | 79.3% | 19.9% | 0.0% | 21.1% | 0.0% | 0.0% |
| VideoUnmask | 90.9% | 53.5% | 20.0% | 67.6% | 20.0% | 0.0% |
| VideoUnmaskSwap | 76.2% | 18.8% | 3.5% | 24.4% | 6.7% | 0.0% |

## 与 Oracle Event Window 上界对比

| Method | Field | Full state | Final | Full sequence |
|---|---:|---:|---:|---:|
| Oracle 12-frame event windows | 91.8% | 57.2% | 46.7% | 26.7% |
| Fixed chunks, no trigger | 79.5% | 21.2% | 6.7% | 0.0% |

固定分块显著降低了结构复杂度，但当前直接 updater 无法在 40–95 次连续更新中稳定保持并正确修改长期状态。

## 视觉与时序诊断

| Test input | All-chunk state | Change-chunk state | Final |
|---|---:|---:|---:|
| Normal | 21.2% | 6.7% | 6.7% |
| Zero video | 3.6% | 0.0% | 0.0% |
| Reverse chunk order | 20.5% | 6.7% | 6.7% |
| Shuffle video across episodes | 22.4% | 7.1% | 5.0% |

清零视频后状态性能大幅下降，说明模型确实使用视觉特征；但倒序几乎不下降，说明它没有稳定学习长时事件顺序。主要失败模式不是“完全忽略视觉”，而是：

1. 大量普通 chunk 反复写入造成 memory drift；
2. 状态变化只占 13.3%，变化识别监督稀疏；
3. Pick/PlaceOrder 需要 40–95 次更新，误差持续累积；
4. 当前 updater 每步必经 cross-attention 和 LayerNorm，没有显式的 identity-preserving training constraint。

## 结论

本实验不支持直接用“每 12 帧无条件更新一次”作为最终方案。它结构最简单，但 held-out final 仅 6.7%，明显不足以接入 action。

下一步仍可以保持“无滑动窗口、无显式 event detector”的原则。最低复杂度的优化不是增加新 head，而是加入无参数的 no-change consistency loss：

```text
L_keep = ||M_t - stop_gradient(M_{t-1})||
```

仅在 teacher state 不变的训练 chunk 上启用，同时继续对 change chunks 做 canonical teacher distillation。它不改变推理结构，也不需要 event trigger；teacher state difference 只用于生成训练 loss mask。

## 产物

- Sequence builder：`scripts/mem/build_robomme_four_task_fixed_chunk_sequences.py`
- Feature cache：`scripts/mem/cache_robomme_four_task_fixed_chunk_features.py`
- Scan student：`src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- Trainer：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- Diagnostic evaluator：`scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py`
- Best checkpoint：`checkpoints/robomme_four_task_fixed_chunk_student_v1_260826/best/params`
- Result：`checkpoints/robomme_four_task_fixed_chunk_student_v1_260826/result.json`
- Dependence diagnostic：`checkpoints/robomme_four_task_fixed_chunk_student_v1_260826/test_visual_dependence.json`
