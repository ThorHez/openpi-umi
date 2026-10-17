# PickXTimes Unified Semantic-Feedback Student（2026-08-28）

## 目标

验证显式 semantic-state recurrent feedback 能否解决 opaque latent MEM 在 PickXTimes 上长期存在的 transition/hold 冲突与 sequence=0，同时满足：

- 使用统一的 19 字段 RoboMME semantic schema；
- 所有字段共用 value embedding、transition updater 和 classifier；
- 不增加 PickXTimes 专用 head；
- 不蒸馏 teacher latent；
- 推理阶段不使用 GT event 或 GT previous state。

## 模型

每个 12 帧固定 chunk 的更新为：

```text
previous semantic distribution
        -> shared field tokens
RGB + proprio -> multimodal evidence
        -> shared semantic delta updater
        -> next semantic distribution
        -> recurrent feedback
```

更新采用：

```text
next_logits = normalized(previous_logits) + semantic_delta
```

因此 no-change 对应显式的零 delta/identity update，不再依赖 opaque attention memory 与 scalar write gate 自行形成恒等映射。

模型输出完整 19 字段，每个任务仍由统一 `state_field_mask` 决定有效字段。当前只训练 Pick 数据，但结构本身不是 Pick 专用。

Episode 初始 semantic state 来自 prompt 可确定的信息与任务固定初始状态，例如 `task`、`required_count`、`completed_count=0`、`holding=0`；不包含任何未来 event 或最终状态信息。

## 训练

四个 seed 均训练 1000 步：

1. step 1–400：GT previous semantic state teacher forcing，学习局部 observation-conditioned transition；
2. step 401–1000：完全使用模型自身 semantic distribution，直接优化 free rollout；
3. checkpoint 按 dev free-rollout `min(transition, no-change)` 选择；
4. locked test 只评估 dev 选出的 checkpoint。

输入、划分和原 latent MEM 完全一致：70 train、15 dev、15 test；RGB 为固定 12 帧 4×4 SigLIP patch tokens，proprio 为六维时序状态。

## Locked-test 结果

以下均为推理时完全 free rollout：

| seed | best step | State | Transition | Hold | Sequence | Final |
|---:|---:|---:|---:|---:|---:|---:|
| 260868 | 1000 | 87.92% | 89.69% | 87.63% | 46.67% | 86.67% |
| 260869 | 900 | 86.32% | 88.66% | 85.93% | 46.67% | 93.33% |
| 260870 | 1000 | 85.74% | 90.72% | 84.92% | 33.33% | 93.33% |
| 260871 | 900 | 90.25% | 90.72% | 90.17% | 66.67% | 86.67% |
| **mean** | — | **87.56%** | **89.95%** | **87.16%** | **48.33%** | **90.00%** |

最佳平衡 checkpoint 为 seed 260871：transition 90.72%、hold 90.17%、state 90.25%、sequence 66.67%。

## 与 opaque latent MEM 对比

所有指标都只比较相同的四个 Pick 动态字段：

| 模型 | State | Transition | Hold | Sequence | Final |
|---|---:|---:|---:|---:|---:|
| 最佳 opaque latent MEM | 50.07% | 79.38% | 45.25% | 0% | 66.67% |
| Semantic feedback，四 seed mean | 87.56% | 89.95% | 87.16% | 48.33% | 90.00% |
| Semantic feedback，最佳 seed | 90.25% | 90.72% | 90.17% | 66.67% | 86.67% |
| Structured transition probe 上界 | 99.70% | 100% | 99.65% | 86.67% | 100% |

相对最佳 opaque latent MEM，semantic feedback 平均提升：

- state：+37.49 个百分点；
- transition：+10.57 个百分点；
- hold：+41.91 个百分点；
- sequence：从 0 提升到 48.33%；
- final：+23.33 个百分点。

这不是 loss reweighting 带来的 transition/hold 搬运，而是两项同时显著提高。

## 阶段消融

仅完成 teacher-forced 预训练、尚未进行 free-rollout finetune 的 step 400 dev 结果：

| seed | Transition | Hold | Sequence |
|---:|---:|---:|---:|
| 260868 | 0.00% | 13.10% | 0% |
| 260869 | 10.10% | 21.30% | 0% |
| 260870 | 13.10% | 28.20% | 0% |
| 260871 | 0.00% | 16.00% | 0% |

切换 free-rollout 训练后，四组在 step 800–1000 达到约 85%–93% 的 dev transition/hold。这说明：

- 显式 semantic state 解决了表示与更新目标问题；
- 但 soft semantic distribution 仍存在 teacher-forced sharp GT 与 self-rollout soft state 的输入分布差异；
- 第二阶段 free-rollout finetune 是当前实现成功的必要组成部分。

在 free-rollout finetune 后重新输入 sharp GT previous state，teacher-forced 指标反而降低。这不是 free-rollout 退化，而是模型已适应自身 soft distribution。后续可用 straight-through hard feedback 或对 semantic probabilities 做温度/校准约束，缩小两种反馈分布差异。

## 结论

实验验证了此前误差解耦的判断：PickXTimes 的主要瓶颈确实是 opaque latent representation/update interface。只把上一时刻统一 semantic state 显式反馈给 updater，就能同时恢复 transition、hold、sequence 与 final，而无需继续优化 write gate。

该结果已达到此前提出的 action 接入门槛（transition >80%、hold >70%），适合进入 oracle-action-backbone + semantic-feedback MEM 的 10–20 条 smoke test。但在动作实验前，建议先明确 action adapter 接收的是 semantic probability tokens，并保留 residual latent 接口以支持未来 VideoUnmask/VideoPlaceOrder 中 schema 之外的视觉关系。

## 下一步

1. 加入 straight-through hard semantic feedback，验证能否逼近 structured probe 的 99% 上界并减少 free-rollout finetune需求；
2. 将同一 19 字段 shared updater 扩展到 VideoUnmask、Swap、PlaceOrder，验证不是 Pick 专用收益；
3. 输出 semantic tokens 给 MME action adapter，比较 oracle semantic state 与 learned semantic-feedback state 的动作成功率。

## 产物

- 模型：`src/openpi/tasks/robomme/unified_semantic_feedback_student.py`
- 训练：`scripts/mem/train_pickxtimes_semantic_feedback_student.py`
- seed 260868：`checkpoints/pickxtimes_unified_semantic_feedback_seed260868_260828/`
- seed 260869：`checkpoints/pickxtimes_unified_semantic_feedback_seed260869_260828/`
- seed 260870：`checkpoints/pickxtimes_unified_semantic_feedback_seed260870_260828/`
- seed 260871：`checkpoints/pickxtimes_unified_semantic_feedback_seed260871_260828/`
