# PickXTimes 两阶段 gate 校准与 min-balance 训练（2026-08-28）

## 目的

在不修改 recurrent MEM 主体结构的前提下，验证以下训练流程能否同时达到：

- transition state exact accuracy ≥ 80%；
- no-change/hold state exact accuracy ≥ 70%。

流程分为两阶段：

1. 仅训练 `proprio_*` 编码器与 recurrent updater 内的 `gate_*` 参数，以 privileged soft event target 校准写入 gate；checkpoint 按 dev `change_write_gate_mean - far_hold_write_gate_mean` 最大化选择。
2. 冻结上述 `proprio_* + gate_*` 参数，训练其余 updater/readout；加入独立 no-change readout loss，checkpoint 按 dev `min(transition, no-change)` 最大化选择。

训练均使用 12 帧固定 chunk、RGB+proprio、direct teacher delta，以及 GPU 0–3 上的四个独立 seed。

## 实现

训练脚本新增：

- `--gate-proprio-only-training`：仅更新 `proprio_* + gate_*`；
- `--freeze-gate-proprio`：冻结校准后的 `proprio_* + gate_*`；
- `--pick-no-change-field-weighting`：将 transition 字段权重显式地用于 no-change loss。默认关闭，因此本实验的 no-change CE 使用等字段权重；
- gate 校准阶段按 dev change/far-hold margin 选择 checkpoint。

训练脚本：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`。

## 阶段一：gate/proprio 校准

每个 seed 训练 500 步，初始 gate margin 约为 0.008–0.010，目标门槛为 0.10。

| seed | warm start | best step | dev change gate | dev far-hold gate | dev margin |
|---:|---:|---:|---:|---:|---:|
| 260852 | 260840 | 500 | 0.4778 | 0.1747 | 0.3031 |
| 260853 | 260841 | 500 | 0.4570 | 0.2044 | 0.2526 |
| 260854 | 260842 | 450 | 0.4595 | 0.1821 | 0.2774 |
| 260855 | 260843 | 450 | 0.4910 | 0.2031 | 0.2880 |

四个 seed 均明显超过 0.10，说明现有 RGB+gripper/EEF proprio 信息足以在独立校准目标下学出 change/hold gate 分离。

Checkpoint：

- `checkpoints/pickxtimes_gate_proprio_calibration_seed260852_from260840_260828/`
- `checkpoints/pickxtimes_gate_proprio_calibration_seed260853_from260841_260828/`
- `checkpoints/pickxtimes_gate_proprio_calibration_seed260854_from260842_260828/`
- `checkpoints/pickxtimes_gate_proprio_calibration_seed260855_from260843_260828/`

## 阶段二：冻结 gate/proprio，训练 updater/readout

每个 seed 训练 1000 步；locked test 只评估由 dev `min(transition, no-change)` 选出的 checkpoint。

| seed | best step | transition | no-change | all-state | final | sequence | test gate margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 260856 | 1000 | 75.26% | 43.39% | 47.89% | 66.67% | 0.00% | 0.0141 |
| 260857 | 1000 | 72.16% | 43.73% | 47.74% | 66.67% | 0.00% | 0.0092 |
| 260858 | 300 | 69.07% | 43.56% | 47.16% | 80.00% | 0.00% | 0.0067 |
| 260859 | 800 | 72.16% | 42.71% | 46.87% | 66.67% | 0.00% | 0.0063 |
| **mean ± std** | — | **72.16 ± 2.19%** | **43.35 ± 0.39%** | **47.42 ± 0.42%** | **70.00 ± 5.77%** | **0.00%** | **0.0091 ± 0.0031** |

Checkpoint：

- `checkpoints/pickxtimes_two_stage_minbalance_seed260856_from260852_260828/`
- `checkpoints/pickxtimes_two_stage_minbalance_seed260857_from260853_260828/`
- `checkpoints/pickxtimes_two_stage_minbalance_seed260858_from260854_260828/`
- `checkpoints/pickxtimes_two_stage_minbalance_seed260859_from260855_260828/`

## 对照

直接从原 RGB+proprio checkpoint 训练独立 no-change loss、但不做阶段一强 gate 校准的四 seed 结果为：

| 方法 | transition | no-change | all-state | final | sequence |
|---|---:|---:|---:|---:|---:|
| 独立 no-change baseline | 75.77 ± 0.52% | 43.52 ± 0.79% | 48.07% | 70.00% | 0.00% |
| 两阶段 gate 校准 | 72.16 ± 2.19% | 43.35 ± 0.39% | 47.42% | 70.00% | 0.00% |

两阶段方案没有达到 80%/70%，且 transition 比直接训练下降约 3.61 个百分点。

## 关键结论

阶段一证明了 gate 标签和输入信息本身是可学的，但“冻结 gate 参数”不能保证“冻结 gate 行为”。当前 gate 同时读取 recurrent memory summary 与 evidence summary。阶段二更新 updater/visual pathway 后，gate 的输入分布发生变化；即使 `gate_*` 和 `proprio_*` 权重不变，dev margin 仍在约 100 步内从 0.25–0.30 回落到约 0.004–0.006。最终 locked-test margin 也只有 0.006–0.014。

因此当前瓶颈不是训练步数不足，也不是 privileged 信息不足，而是 gate 与被它控制的 recurrent state/updater 存在输入分布耦合。阶段一的独立最优解在阶段二并不稳定。

在继续保持模型结构不变的约束下，下一项更合理的实验是第二阶段不冻结 gate/proprio，而是保留较小权重的 soft-gate/rank regularization，并把 checkpoint 选择改为同时约束 `min(transition,no-change)` 和 gate margin。若允许轻微结构改动，则应为 gate 使用独立、稳定的 event feature path，避免直接依赖随 updater 训练而漂移的 recurrent memory summary。

## 验证

- 四个阶段一任务正常退出；
- 四个阶段二任务正常退出并生成 locked-test 结果；
- checkpoint 恢复：每次 202 个参数叶成功恢复，无 shape mismatch；
- `src/openpi/tasks/robomme/unified_fixed_chunk_student_test.py`：11 passed。

## 后续实验：联合 gate regularization + margin-aware 选模

冻结实验表明 gate 参数冻结后，gate 行为仍会随 recurrent memory 输入漂移。因此进一步验证第二阶段联合更新全部参数，并持续加入较低权重的 gate regularization：

- privileged soft-gate loss weight：0.5；
- privileged gate rank loss weight：0.5；
- checkpoint utility：`min(transition, no-change) - max(0.1 - gate_margin, 0)`；
- 其余 state、delta、no-change loss 与上一实验相同；
- 从四个阶段一高-margin checkpoint 分别训练 1000 步。

| seed | best step | transition | no-change | all-state | final | sequence | test gate margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 260860 | 900 | 75.26% | 43.22% | 47.74% | 60.00% | 0.00% | 0.0239 |
| 260861 | 800 | 70.10% | 44.41% | 48.03% | 60.00% | 0.00% | 0.0152 |
| 260862 | 900 | 78.35% | 45.25% | 49.93% | 60.00% | 0.00% | 0.0149 |
| 260863 | 500 | 71.13% | 42.54% | 46.58% | 66.67% | 0.00% | 0.0102 |
| **mean ± std** | — | **73.71 ± 3.30%** | **43.86 ± 1.05%** | **48.07 ± 1.20%** | **61.67 ± 2.89%** | **0.00%** | **0.0161 ± 0.0050** |

Checkpoint：

- `checkpoints/pickxtimes_joint_gate_margin_seed260860_from260852_260828/`
- `checkpoints/pickxtimes_joint_gate_margin_seed260861_from260853_260828/`
- `checkpoints/pickxtimes_joint_gate_margin_seed260862_from260854_260828/`
- `checkpoints/pickxtimes_joint_gate_margin_seed260863_from260855_260828/`

相对冻结 gate/proprio 的两阶段结果，联合 regularization 将平均 transition 从 72.16% 提升到 73.71%，no-change 从 43.35% 提升到 43.86%，test gate margin 从 0.0091 提升到 0.0161；但仍未超过直接 no-change baseline 的 75.77%/43.52%，也远未达到 80%/70%。

该结果说明 0.5/0.5 的 gate regularization 能缓解但不能消除输入分布漂移。训练约 100 步时，dev gate margin 已从阶段一的 0.25–0.30 降到 0.002–0.008；最终 margin-aware 选模只能选到 0.010–0.024 的 test margin。下一步若继续保持结构不变，应先做 gate loss 权重的 Pareto sweep（例如 1、2、5），而不是继续单纯增加训练步数。
