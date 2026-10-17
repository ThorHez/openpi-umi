# RoboMME Memory Trajectory / Final Readout 解耦实验

日期：2026-08-27

## 结论

本实验没有实现“保留 final 优势并恢复 all-state”的目标，不应替换上一版 action 候选 checkpoint。

从 final/change 加权模型的最佳 step 350 出发，将 teacher-memory trajectory loss 与 all/change/final readout loss 解耦后，held-out test final 仅从 35.0% 提升到 36.7%，all-state 反而从 43.4% 降到 40.7%。唯一明确收益集中在 PickXtimes final（73.3% → 86.7%），另外三个任务没有整体改善。

## 解耦目标

模型结构、固定 12 帧非滑窗输入和 soft gate 均不改变。训练目标改为：

```text
L = 1.0 * L_memory_trajectory
  + 0.5 * L_state_all
  + 0.75 * L_state_change
  + 1.0 * L_state_final
  + 0.05 * L_keep
```

- `L_memory_trajectory`：teacher memory distillation，change state 权重 6，final 不额外加权；
- `L_state_all`：所有有效状态上的 frozen teacher readout CE；
- `L_state_change`：只在 teacher state 改变的位置计算；
- `L_state_final`：只在每条 episode 最后一个有效状态计算；
- `L_keep`：teacher state 不变时，约束当前 memory 接近 stop-gradient 的上一 memory。

## 设置

- 初始化：`robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/best/params`；
- 续训：500 steps；
- warmup：25；
- LR：1e-5 cosine decay 到 3e-6；
- batch size：4，四任务平衡；
- 每 50 steps dev；
- 最佳 step：450。

## Held-out Test 对照

| Method | Field | All state | Change | Hold | Final | Sequence |
|---|---:|---:|---:|---:|---:|---:|
| 固定 chunk，无 gate | 79.5% | 21.2% | 6.7% | 23.5% | 6.7% | 0.0% |
| Soft gate，原始目标 | **89.9%** | **46.0%** | 20.1% | **50.0%** | 13.3% | 0.0% |
| Soft gate，final/change 加权 | 88.8% | 43.4% | **27.7%** | 45.8% | 35.0% | 0.0% |
| Soft gate，解耦目标 | 88.1% | 40.7% | **27.7%** | 42.7% | **36.7%** | 0.0% |

解耦版相对上一版：

- final：+1.7 pp（test 仅多正确 1/60 episode）；
- change：无变化；
- all-state：-2.7 pp；
- hold：-3.1 pp；
- field：-0.7 pp。

该变化不构成有意义的总体提升。

## 分任务 Test

| Task | Final：上一版 → 解耦版 | All-state：上一版 → 解耦版 | Change：上一版 → 解耦版 |
|---|---:|---:|---:|
| PickXtimes | 73.3% → **86.7%** | 38.3% → 35.4% | 40.2% → 40.2% |
| VideoPlaceOrder | 26.7% → 26.7% | 54.0% → 50.8% | 32.5% → **35.0%** |
| VideoUnmask | 20.0% → 20.0% | 53.5% → 53.5% | 20.0% → 20.0% |
| VideoUnmaskSwap | 20.0% → 13.3% | 20.7% → 19.2% | 7.0% → 5.3% |

结果表明统一加权继续优先改善了相对容易的 PickXtimes，未解决另外三个任务的 final bottleneck。

## 为什么“形式解耦”没有恢复 trajectory

在最佳 step 450 的 dev 上，各项未经/经过权重的典型量级约为：

| Component | Raw | Weighted contribution |
|---|---:|---:|
| Memory trajectory | 0.4405 | 0.4405 |
| All-state readout | 0.3042 | 0.1521 |
| Change readout | 0.3114 | 0.2336 |
| Final readout | 0.2811 | 0.2811 |
| Keep consistency | 0.00254 | 0.00013 |

虽然 supervision mask 已经解耦，但 change+final 对总梯度的贡献仍大于 memory trajectory；`L_keep` 的实际贡献几乎为零。因此它只实现了损失 bookkeeping 解耦，没有形成足够强的 trajectory-preserving optimization。

另一个原因是实验从已经经过 final 加权的 checkpoint 开始，表示空间已经偏向 Pick final；低学习率续训更容易强化该局部最优，而不是恢复旧 trajectory。

## Gate 与输入依赖

正常 test gate：

- change：0.02290；
- hold：0.01849；
- change 相对高约 23.9%。

| Input | All-state | Change | Final |
|---|---:|---:|---:|
| Normal | 40.7% | 27.7% | 36.7% |
| Zero video | 9.8% | 0.0% | 0.0% |
| Reverse chunks | 37.0% | 23.2% | 33.3% |
| Shuffle episode video | 36.9% | 23.7% | 26.7% |

模型仍依赖视觉，soft gate 的 change/hold 选择性也保持。但倒序 final 仍高，完整顺序建模问题没有改善。

## 决策

1. 不使用本解耦版作为 action 候选；
2. 保留 `robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/best/params` 作为当前 action 接入候选；
3. 不继续沿用当前统一加权比例，因为它主要继续优化 Pick；
4. 在下一轮训练前，先对 VideoPlaceOrder、VideoUnmask、VideoUnmaskSwap 做 final per-field confusion/error audit，区分视觉事件提取、memory update 和 frozen readout 三类瓶颈；
5. 如果重试 trajectory-preserving 训练，应从原 soft-gate step 1700 而不是 final-biased checkpoint 开始，并对各 loss 做梯度范数平衡或交替优化，而不是继续固定加权求和。

## 产物

- 代码：`src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- Trainer：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- 最佳 checkpoint：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_decoupled_v1_260827/best/params`
- Result：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_decoupled_v1_260827/result.json`
- Metrics：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_decoupled_v1_260827/metrics.jsonl`
- 扰动诊断：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_decoupled_v1_260827/test_visual_dependence.json`

