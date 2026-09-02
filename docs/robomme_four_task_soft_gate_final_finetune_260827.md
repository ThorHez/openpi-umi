# RoboMME Soft Gate Final/Change 加权续训实验

日期：2026-08-27

## 结论

从固定 12 帧、非滑窗 soft-gate student 的原最佳 step 1700 checkpoint 出发，以低学习率和 final/change 加权续训，可把 held-out test final state 从 13.3% 提升到 35.0%，其中 PickXtimes final 从 0% 提升到 73.3%。

该收益伴随 all-state 和 no-change state 小幅下降，说明续训确实将模型容量从密集的状态保持转向了任务终态和变化状态，而不是所有指标同时提升。

## 设置

- 初始化：`robomme_four_task_fixed_chunk_soft_gate_v1_260826/best/params`（原 step 1700）；
- 固定 chunk/stride：12/12 frames；
- 无滑动窗口、无显式 event detector；
- soft gate 无 event/gate supervision；
- 续训 500 steps；
- warmup：25 steps；
- LR：peak 1e-5，cosine decay 到 3e-6；
- change-state weight：6 → 10；
- final-state weight：1 → 4；
- 如果 final state 同时是 change state，总权重为 40；
- 每 50 steps 验证；
- checkpoint 仍按 dev `(final, transition, sequence, all-state)` 字典序选择。

最佳续训 checkpoint 为 step 350。其 dev final 为 46.7%、transition 为 33.2%、full sequence 为 3.3%、all-state 为 39.9%。

## Held-out Test 对照

| Method | Field | All state | Change state | No-change state | Final | Full sequence |
|---|---:|---:|---:|---:|---:|---:|
| 固定 chunk，无 gate | 79.5% | 21.2% | 6.7% | 23.5% | 6.7% | 0.0% |
| Soft gate，原最佳 | **89.9%** | **46.0%** | 20.1% | **50.0%** | 13.3% | 0.0% |
| Soft gate，final/change 续训 | 88.8% | 43.4% | **27.7%** | 45.8% | **35.0%** | 0.0% |

相对原 soft-gate checkpoint：

- final：+21.7 pp；
- change state：+7.6 pp；
- all-state：-2.7 pp；
- no-change state：-4.2 pp；
- field：-1.1 pp。

## 分任务 Test

| Task | Final：原 gate → 续训 | Change：原 gate → 续训 | All state：原 gate → 续训 |
|---|---:|---:|---:|
| PickXtimes | 0.0% → **73.3%** | 28.9% → **40.2%** | 46.3% → 38.3% |
| VideoPlaceOrder | 20.0% → **26.7%** | 20.0% → **32.5%** | 52.7% → **54.0%** |
| VideoUnmask | 20.0% → 20.0% | 20.0% → 20.0% | 53.5% → 53.5% |
| VideoUnmaskSwap | 13.3% → **20.0%** | 5.3% → **7.0%** | 20.2% → 20.7% |

PickXtimes 的显著变化表明原来的 final=0 主要是训练目标偏置，而不是模型完全没有计数能力。但 Pick 的 all-state 从 46.3% 降到 38.3%，说明当前 final weight=4 已经形成可见的终态/轨迹权衡。

## Gate 与扰动诊断

正常 test：

- change gate：0.02170；
- hold gate：0.01771；
- change 相对 hold 高约 22.5%。

| Input | All state | Change state | Final | Mean gate |
|---|---:|---:|---:|---:|
| Normal | 43.4% | 27.7% | 35.0% | 0.0183 |
| Zero video | 10.8% | 0.0% | 0.0% | 0.1504 |
| Reverse chunks | 38.8% | 23.2% | 33.3% | 0.0185 |
| Shuffle video across episodes | 39.1% | 22.3% | 25.0% | 0.0405 |

zero-video 和跨 episode shuffle 均显著损害 final，支持模型在使用视觉内容；gate 对 change/hold 的相对选择性没有因 final 加权而消失。

倒序后 all-state 和 change state 的下降比原 checkpoint 更明显，但 final 只从 35.0% 降到 33.3%，说明终态改善不等于完整事件顺序已经解决。zero-video 时 gate 异常增大的 OOD 校准问题也仍存在。

## 决策

本实验支持：

1. 保留固定非滑窗 soft gate；
2. 使用 early-stopped final/change 加权续训 checkpoint，尤其用于 PickXtimes action 接入前的 memory 候选；
3. 不再单纯增加原始训练步数，因为优化目标决定的收益远大于延长训练。

当前模型适合进入离线 action-conditioning 兼容性测试，但在正式 closed-loop action 训练前仍应同时报告轨迹状态与 final 指标，避免只优化终态掩盖中间记忆退化。

## 产物

- Trainer：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- 最佳 checkpoint：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/best/params`
- Result：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/result.json`
- Metrics：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/metrics.jsonl`
- 扰动诊断：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/test_visual_dependence.json`

