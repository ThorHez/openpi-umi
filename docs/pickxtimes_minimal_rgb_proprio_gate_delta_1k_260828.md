# PickXTimes 最小 RGB+proprio recurrent MEM 实验

日期：2026-08-28

## 目标

在不加入滑动窗口或硬 event trigger 的条件下，将 fixed-chunk recurrent MEM 的 strict
transition accuracy 从约 43% 提升到 80%。

## 配置

- 固定非重叠 12 帧 chunk，stride 12；
- front RGB patch tokens；
- proprio：gripper state、gripper close、gripper command、observed EEF-Z、commanded EEF-Z；
- learned soft write gate，使用 simulator transition 生成训练期 soft target；
- actual recurrent memory delta 直接蒸馏 teacher delta；
- transition subtype-balanced loss；
- completed/holding/ready 字段加权；
- trajectory、transition、final、keep loss 解耦；
- 从原 Pick checkpoint warm start；
- 4 seeds 分别运行在 GPU 0、1、2、3；
- 每个 seed 1000 steps，batch size 4；
- 按 dev transition、no-change、state、final 的顺序选 best checkpoint。

旧 checkpoint 实际只有 125 steps，最佳 dev strict transition 为 43.43%。其 write gate 在
transition chunk 上约 0.0117，在 hold chunk 上约 0.0139，未形成正确事件选择性。

## Locked-test 结果

| Seed | Best step | Dev transition | Test transition | Test no-change | Test all-state | Test final | Full sequence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 260840 | 1000 | 85.86% | 76.29% | 34.07% | 40.03% | 66.67% | 0% |
| 260841 | 900 | 85.86% | 78.35% | 33.22% | 39.59% | 73.33% | 0% |
| 260842 | 700 | 85.86% | **80.41%** | 27.97% | 35.37% | 80.00% | 0% |
| 260843 | 800 | 84.85% | 76.29% | 28.98% | 35.66% | 66.67% | 0% |
| Mean ± std | — | 85.61 ± 0.44% | **77.84 ± 1.71%** | 31.06 ± 2.63% | 37.66 ± 2.16% | 71.67 ± 5.53% | 0% |

相对原 recurrent MEM + adapted decoder 的 locked-test transition 43.30%，本实验平均提升
34.54 个百分点；最佳 seed 提升 37.11 个百分点，并首次超过 80%。

## Gate 与 delta 诊断

四个 best checkpoint 的 test gate：

| Seed | Transition gate | Far-hold gate | Margin |
|---:|---:|---:|---:|
| 260840 | 0.0404 | 0.0341 | 0.0063 |
| 260841 | 0.0420 | 0.0360 | 0.0060 |
| 260842 | 0.0359 | 0.0315 | 0.0043 |
| 260843 | 0.0369 | 0.0328 | 0.0041 |

gate 已从原来的负 margin 变为正 margin，但距离训练 target 的明显分离仍很远。direct teacher-delta
test loss 约为 0.0313--0.0317，也没有充分收敛。transition 的主要提升来自 proprio evidence 与
加权 transition readout，而不是 gate 已经学好。

## 结论

1. RGB+proprio 和 transition-specific supervision 的方向有效；单 seed 已达到 80.41%。
2. 80% 尚不稳健：四 seed 平均 77.84%，只有 1/4 seed 在 locked test 超过80%。
3. 当前存在明显 transition/hold 权衡。模型为了在边界快速改变状态，破坏了非边界状态保持；
   no-change 只有31%，full sequence 仍为0%，因此不应直接进行 action smoke test。
4. 单纯继续增加当前训练步数不太可能解决 hold：best step 分布在700--1000，且 checkpoint
   selection 已经看到后期结果。下一轮需要显式增加 no-change readout/persistence，而不是继续强化
   transition loss。

## 下一项最小修正

保持输入和结构不变，只修正 loss：

- 新增独立 no-change state readout CE，权重 1.0--1.5；
- keep loss 从 0.2 提高到 0.5，但只作用于 GT no-change chunk；
- transition readout 从 2.0 降到 1.25--1.5；
- gate 先单独预训练到 transition/far-hold margin >= 0.1，再进行 joint training；
- checkpoint 选择改为 `min(transition, no-change)`，避免以牺牲 hold 换 transition；
- 验收目标：test transition >= 80%、no-change >= 70%、full sequence >= 30%。

## 产物

- proprio 构建脚本：`scripts/mem/build_pickxtimes_fixed_chunk_proprio.py`
- proprio 缓存：`artifacts/pickxtimes_fixed_chunk_proprio_v1_260828/`
- recurrent student：`src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- 训练入口：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- checkpoints：
  `checkpoints/pickxtimes_minimal_rgb_proprio_gate_delta_seed{260840..260843}_260828/`
