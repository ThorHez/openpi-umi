# PickXTimes 环境特权信息到 recurrent MEM 的 soft-gate 蒸馏小实验（2026-08-27）

## 结论摘要

本轮实验验证了两件不同的事：

1. 当前视觉特征中存在一定的事件可分信号。只优化 gate 监督时，dev 的事件/远离事件 gate 比值可以从约 `0.80` 提高到 `1.27`。
2. 当前 recurrent updater 中原有的 `write_gate` 不能直接当作事件概率监督。它已经与 candidate update 的幅度和递归积分过程耦合；轻微改变它的绝对尺度就会破坏已经学到的 memory dynamics。

因此，环境特权信息蒸馏路线是可行的，但正确结构应当是“保留原 update-amplitude gate，新增独立 event-confidence soft gate”，而不是用事件标签覆盖原 gate。

## 实验问题

使用 PickXTimes 数据中训练期可见的特权 `state_change_mask`，检查：

- 特权事件标签能否让视觉 recurrent MEM 在事件附近提高 soft gate；
- gate 选择性是否能改善 transition/all-state；
- 原 gate 的功能究竟是事件置信度，还是递归更新步长。

所有实验均从同一个已验证 PickXTimes checkpoint 开始：

`checkpoints/robomme_single_task_pick_equal_exposure_seed260827_260827/best/params`

固定条件：12 帧非重叠 chunk、batch size 4、seed 260829、同一 train/dev split。test 未参与本轮选择。

## 新增训练能力

训练脚本：

`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`

新增内容：

- 基于 `state_change_mask` 生成 Gaussian soft event target；
- soft-gate BCE 与 event-vs-far-hold ranking loss；
- `event_to_far_hold_gate_ratio`、`far_hold_write_gate_mean` 等诊断指标；
- `--gate-only-training`：仅更新 recurrent updater 中的 `gate_*` 参数，并只用特权 gate objective 做可分性探针；其余参数完全冻结。

特权标签仅用于训练 loss，不进入模型推理输入。

## 实验组

### A. Dense-state control

继续训练 200 step，不加入特权 gate loss，用于排除继续训练本身的影响。

产物：

`checkpoints/pickxtimes_privileged_dense_control_seed260829_260827/`

### B. Weak joint soft-gate

继续训练 200 step，同时加入低权重特权 gate BCE/ranking。目标范围为 0.05--0.8。

产物：

`checkpoints/pickxtimes_privileged_dense_softgate_seed260829_260827/`

### C. Mixed-objective gate-only diagnostic

仅允许 `gate_*` 参数更新，但 memory/state 与 gate objective 同时对 gate 反传。该组用于定位梯度冲突，不作为最终方案。

产物：

`checkpoints/pickxtimes_privileged_gate_probe_calibrated_seed260829_260827/`

### D. Pure privileged gate probe

冻结非 gate 参数，只优化特权 gate objective。目标范围校准为 0.005--0.08。

产物：

`checkpoints/pickxtimes_privileged_gate_probe_pure_seed260829_260827/`

### E. Scale-preserving pure gate probe

冻结非 gate 参数，只优化特权 gate objective；使用窄时间核和 0.007--0.03 目标，使目标平均值接近原 gate 尺度。

产物：

`checkpoints/pickxtimes_privileged_gate_probe_scale_preserving_seed260829_260827/`

## Dev 结果

百分数均为 strict state 指标。A/B 取各自按既定 checkpoint 规则选择的 best；D/E 同时列出最能回答诊断问题的末步或最佳步。

| 组别 | Step | Final | Transition | Hold | All-state | Event/far-hold gate | Mean gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 Pick checkpoint | 75 | 100.0 | 43.4 | -- | 41.6 | 约 0.8 | 约 0.014 |
| A: control best | 150 | 100.0 | 40.4 | 38.9 | 39.2 | 0.791 | 0.0137 |
| B: weak joint best | 175 | 100.0 | 40.4 | 39.1 | 39.3 | 0.799 | 0.0139 |
| C: mixed diagnostic best | 25 | 93.3 | 37.4 | 43.2 | 42.4 | 0.797 | 0.0132 |
| D: pure probe last | 300 | 0.0 | 0.0 | 28.2 | 24.2 | **1.268** | 0.0294 |
| E: scale-preserving best | 200 | 46.7 | 11.1 | 30.8 | 27.9 | 1.111 | 0.0116 |

## 结果解释

### 1. 低权重直接联合训练基本无效

B 相对 A 的最佳 all-state 只增加 `0.15` 个绝对百分点，transition 完全相同，gate ratio 仍小于 1。参数对比确认 gate 参数存在非零更新，因此不是梯度断路，而是小学习率和 canonical memory/state loss 将 gate 锁在原动态附近。

### 2. 特权标签确实能教出部分事件选择性

D 去掉 canonical loss 对 gate 的约束后，ratio 从约 0.8 上升到 1.27。这说明视觉 evidence 和 recurrent memory 中包含可用于事件判断的信号，环境特权标签有蒸馏价值。

但 1.27 仍低于预期的 2.0，表明当前 `mean(LN(memory)) + mean(LN(evidence))` 摘要并不是理想事件特征。12 帧内的运动差分、物体接触与交换方向在均值池化中损失较多。

### 3. 原 write gate 是更新步长，不是事件概率

D 将 mean gate 提高到 0.029 后 final 降到 0；E 将 mean gate 降到约 0.011，同样使 final 大幅下降。也就是说，当前模型依靠小步长反复积分 candidate，而 candidate 本身不是可以在事件处一次性写入的完整新状态。

因此不能要求原 gate 在事件处接近 1、稳定处接近 0。即使保持全局均值，改变时间分布也会改变 memory trajectory。

### 4. 直接扩大步数不是解决办法

三种直接监督方式已经覆盖弱联合、纯 gate 和尺度校准。继续用同一结构增加训练步数，只会在“维持旧 memory dynamics”和“提高事件选择性”之间移动，不会解除两种语义的结构耦合。

## 推荐结构：双 gate 分工

保留现有更新公式中的低幅度 gate，改名为 `update_amplitude`：

```text
M_candidate = Updater(M_t, E_t)
M_{t+1} = M_t + g_update * modulation(p_event) * (M_candidate - M_t)
```

新增独立的 `p_event` head：

- 输入使用 current evidence、memory summary、chunk 前后半段差分和 evidence-memory innovation；
- 训练时由 simulator privileged event/state transition 监督；
- 推理时只使用图像和当前 memory，不接收任何 privileged 字段；
- `p_event` 先作为独立可观测量训练，不立刻改变 memory。

建议使用有界、初始恒等的调制：

```text
modulation(p_event) = clip(exp(alpha * (p_event - p0)), 0.75, 1.25)
```

其中 `alpha` 从 0 开始，避免一接入就破坏原 checkpoint 的递归尺度。

## 下一轮训练流程

1. 冻结原 MEM，训练独立 event-confidence head，先报告 event AUPRC、F1、事件提前/滞后误差；不看 action。
2. 事件 head 达标后，令 `alpha: 0 -> 0.25` 逐步升高，联合优化 memory trajectory、final readout 和 privileged event loss。
3. 加入 base-memory anchor：限制新模型 memory 与原 checkpoint 在非事件 chunk 的漂移；事件 chunk 允许更大变化。
4. dev 门槛同时约束：final 不低于 93%、transition 高于原 checkpoint、hold 不下降超过 2 个百分点、event confidence ratio 至少 2。
5. 达标后移除训练期 privileged inputs，仅以视觉 recurrent rollout 进入 action smoke test。

## 当前判断

本轮不是“得到可接 action 的新 checkpoint”，而是完成了结构辨识：

- **支持**从 RoboMME 环境特权信息蒸馏视觉事件知识；
- **否定**把现有 interpolation gate 直接当事件分类 gate；
- 下一步应实现独立 event-confidence head，再以小幅、渐进方式调制原 recurrent update。

