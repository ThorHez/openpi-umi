# PickXTimes 独立双 Gate 实现与实验（2026-08-27）

## 目的

验证将原来的递归写入 gate 拆成两个语义独立的 gate 是否有效：

- `write_gate`：保留原有职责，决定 recurrent updater 的基础更新步长；
- `event_gate`：只判断当前 12 帧 chunk 是否包含值得写入的状态变化；
- 不把事件标签直接监督到原 `write_gate`，避免破坏已经学好的递归动力学。

## 实现

实现文件：

- `src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- `scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- `src/openpi/tasks/robomme/unified_fixed_chunk_student_test.py`

Event gate 使用 memory summary、当前 evidence summary、chunk 内前后半段差分、绝对时间差分以及 evidence-memory 差分作为输入。最终更新为：

```text
base_gate = sigmoid(write_gate_head(...))
event_gate = sigmoid(event_gate_head(...))
modulation = clip(1 + strength * (event_gate - reference), min, max)
effective_gate = base_gate * modulation
memory_next = memory + effective_gate * update_delta
```

`strength=0` 时严格退化为原模型。旧 checkpoint 采用按名称和 shape 匹配的部分恢复方式加载，新增的 8 个 event-head parameter leaves 单独初始化。

训练脚本新增 event BCE、event ranking、AUPRC、AUROC、事件/非事件置信度、modulation 和 effective gate 统计，并支持：

- `--event-gate-only-training`：只训练 event head；
- `--freeze-event-gate`：冻结 event head，适配其余 recurrent memory 参数；
- 可配置调制强度、中心和上下界。

## Stage 1：特权事件信息能否蒸馏到独立 Event Gate

设置：PickXTimes，12 帧固定 chunk，400 steps；冻结原模型，只训练 event head；监督来自 simulator `state_change_mask` 构造的 Gaussian soft target；memory modulation 关闭。

| 指标 | step 0 | best step 325 |
|---|---:|---:|
| Event AUPRC | 14.00% | **29.59%** |
| Event AUROC | 50.00% | **68.27%** |
| Event confidence | 0.500 | 0.592 |
| Far-hold confidence | 0.500 | 0.434 |
| Confidence margin | 0.000 | **0.158** |
| Final state exact | 100.00% | 100.00% |
| Transition state exact | 43.43% | 43.43% |
| All-state exact | 41.78% | 41.78% |

结论：视觉 recurrent 表征包含可蒸馏的事件信息。由于 modulation 关闭，训练 event head 不改变原有 memory 结果，达到了结构解耦的第一项目标。

Checkpoint：

`checkpoints/pickxtimes_dual_gate_event_head_seed260829_260827/best/params`

注：Stage 1 运行后修正了 `gate_modulation_mean` / `effective_write_gate_mean` 的聚合 helper，因此该 run 日志中的这两个字段不可用；state strict metrics 和 event 分类指标不受影响。Stage 2 的相关 gate 指标已使用修正后的实现。

## Stage 2：Event Gate 是否能改善 Memory Update

三个实验都从 Stage-1 best checkpoint 出发，冻结 event head，以完全相同的数据顺序和 seed 训练其余参数 200 steps。以下报告各 run 按统一 checkpoint objective 选出的 best：

| 设置 | best step | Final | Transition | Hold | All-state | 平均 modulation |
|---|---:|---:|---:|---:|---:|---:|
| 无调制，`strength=0` | 0 | **100.00%** | **43.43%** | **41.33%** | **41.63%** | 1.0000 |
| 强调制，`strength=0.25`，范围 0.90–1.10 | 150 | **100.00%** | 40.40% | 39.46% | 39.59% | 1.0053 |
| 小调制，`strength=0.05`，范围 0.98–1.02 | 150 | **100.00%** | 40.40% | 38.95% | 39.16% | 1.0019 |

对应目录：

- `checkpoints/pickxtimes_dual_gate_stage2_control_seed260829_260827`
- `checkpoints/pickxtimes_dual_gate_stage2_mod025_seed260829_260827`
- `checkpoints/pickxtimes_dual_gate_stage2_mod005_seed260829_260827`

## 结论

1. **独立双 gate 的结构实现是有效的。** Event gate 从随机水平学到 AUPRC 29.59%、AUROC 68.27%，而且在关闭调制时完全不破坏旧 memory。
2. **当前乘法调制没有提升 memory accuracy。** 两种非零幅度都比无调制低约 3 个 transition 点、2.0–2.5 个 all-state 点。
3. **主要问题不是 event head 完全不会判断事件，而是 recurrent dynamics 对累计写入尺度非常敏感。** 即使平均 modulation 只有 1.0019，也可能跨越多个 chunk 累积并改变最终轨迹。
4. 当前最佳可用模型仍是双 gate checkpoint 加 `strength=0`，即把 event gate 作为独立事件置信度输出，不让它直接缩放旧 gate。它适合作为后续 action/MEM 接口的诊断信号，但当前结果不支持声称双 gate 提高了 PickXTimes memory accuracy。

## 下一步建议

不再继续搜索乘法幅度。更合理的下一步是让 event gate **选择候选更新内容**，而不是缩放原 gate：

```text
candidate = event_gate * event_update + (1 - event_gate) * hold_update
memory_next = memory + base_gate * candidate
```

这样 `base_gate` 继续控制稳定的积分步长，`event_gate` 只做 event/hold 路由；初始化时令两个 update branch 相同，可保证从旧 checkpoint 严格等价启动，再逐步学习分工。

## 验证

- `pytest -q src/openpi/tasks/robomme/unified_fixed_chunk_student_test.py`：7 passed；
- `git diff --check`：通过；
- 训练和验证均使用 15 条固定 dev episodes 及相同 seed `260829`。
