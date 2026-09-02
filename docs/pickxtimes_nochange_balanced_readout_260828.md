# PickXTimes 独立 no-change readout 与平衡 checkpoint 选择实验

日期：2026-08-28

## 目标

保持 RGB+proprio fixed-chunk recurrent MEM 结构不变：

- 新增独立 no-change state readout CE；
- checkpoint 首先最大化 `min(transition, no-change)`；
- 尝试同时达到 locked-test transition >= 80%、no-change >= 70%。

四个 seed 均从上一轮对应的 best checkpoint 继续训练，分别使用 GPU 0--3。soft gate、teacher
delta、固定非重叠12帧 chunk 和模型参数结构均不变。

## Checkpoint 选择

新的 dev score 为：

```text
(
  min(transition_accuracy, no_change_accuracy),
  transition_accuracy,
  no_change_accuracy,
  all_state_accuracy,
  final_accuracy,
)
```

训练在 step 0 先评估 warm-start checkpoint，保证第二阶段只有在平衡指标提高时才覆盖旧 best。

## 实验 A：独立 no-change CE

配置：

- transition readout weight：2.0；
- no-change readout weight：1.5；
- no-change 字段等权；
- 1000 steps，4 seeds。

| Seed | Best step | Test transition | Test no-change | Test all-state | Test final | Sequence |
|---:|---:|---:|---:|---:|---:|---:|
| 260844 | 900 | 75.26% | 44.58% | 48.91% | 66.67% | 0% |
| 260845 | 700 | 76.29% | 43.39% | 48.03% | 80.00% | 0% |
| 260846 | 900 | 75.26% | 43.73% | 48.18% | 60.00% | 0% |
| 260847 | 800 | 76.29% | 42.37% | 47.16% | 73.33% | 0% |
| **Mean ± std** | — | **75.77 ± 0.52%** | **43.52 ± 0.79%** | **48.07 ± 0.62%** | **70.00 ± 7.45%** | **0%** |

相对上一轮无独立 no-change CE 的均值：

- transition：77.84% -> 75.77%，下降2.07pp；
- no-change：31.06% -> 43.52%，提升12.46pp；
- all-state：37.66% -> 48.07%，提升10.41pp。

## 实验 B：no-change 动态字段加权

实验 A 的 no-change CE 包含多个已经接近饱和的静态字段，因此实验 B 对 completed、holding、ready
使用与 transition 相同的动态字段权重。其他结构和主要 loss 权重不变，训练800 steps。

| Seed | Best step | Test transition | Test no-change | Test all-state | Test final | Sequence |
|---:|---:|---:|---:|---:|---:|---:|
| 260848 | 500 | 70.10% | 45.08% | 48.62% | 66.67% | 0% |
| 260849 | 800 | 69.07% | **50.17%** | **52.84%** | 66.67% | 0% |
| 260850 | 700 | 67.01% | 48.47% | 51.09% | 60.00% | 0% |
| 260851 | 700 | 72.16% | 46.27% | 49.93% | 80.00% | 0% |
| **Mean ± std** | — | **69.59 ± 1.86%** | **47.50 ± 1.96%** | **50.62 ± 1.55%** | **68.33 ± 7.26%** | **0%** |

动态字段加权继续恢复 no-change，但进一步破坏 transition；仍不存在80/70交点。

## Gate 诊断

| 配置 | Test transition gate | Test far-hold gate | Margin |
|---|---:|---:|---:|
| 实验 A | 0.04495 | 0.03763 | 0.00732 |
| 实验 B | 0.04165 | 0.03531 | 0.00634 |

虽然 gate margin 已为正，但 transition 与 far-hold 的写入幅度仍非常接近。模型没有真正学会“事件时
写、无事件时保持”，而是在同一连续更新器中用 loss 权重折中。提高 no-change CE 后，只是沿 Pareto
前沿从 transition 一侧移动到 hold 一侧。

## 结论

1. 独立 no-change CE 和平衡 checkpoint 选择实现正确，并显著改善 persistence。
2. 目标没有达到：最佳 locked-test no-change 为50.17%，对应 transition 69.07%；保持较高
   transition 的实验 A 只有约43.5% no-change。
3. 继续调大 no-change loss 不合适。实验 A/B 已显示它会持续用 transition 换 hold。
4. full sequence 在所有设置中仍为0%，因此暂不进行 action smoke test。
5. 当前瓶颈是 soft gate 的可分性，而不是缺少 readout loss 或训练步数。

## 下一步建议（结构不变）

采用两阶段优化，而不是继续联合竞争：

1. 只训练 proprio encoder + `gate_*`，用 simulator soft target 将
   `transition_gate - far_hold_gate` 提升到至少0.1；
2. 冻结已经校准的 gate，再训练 recurrent update/readout；
3. 保留 `min(transition,no-change)` checkpoint 选择；
4. 若 gate margin 达不到0.1，则说明当前全窗口均值 gate feature 不足，需要更改 gate evidence，届时
   才涉及结构变化。

## 产物

- 训练入口：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- 实验 A checkpoints：
  `checkpoints/pickxtimes_nochange_balanced_seed{260844..260847}_from*_260828/`
- 实验 B checkpoints：
  `checkpoints/pickxtimes_nochange_dynamic_weighted_seed{260848..260851}_from*_260828/`
- 实验 A 中 transition 较高的平衡 checkpoint：
  `checkpoints/pickxtimes_nochange_balanced_seed260845_from260841_260828/best/`
- 实验 B 中 no-change 最高的 checkpoint：
  `checkpoints/pickxtimes_nochange_dynamic_weighted_seed260849_from260841_260828/best/`
