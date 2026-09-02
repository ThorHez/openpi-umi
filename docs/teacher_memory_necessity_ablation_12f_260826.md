# Teacher memory 必要性：12 帧严格消融实验

日期：2026-08-26

## 结论

在完全相同的 12 帧输入、数据划分、随机种子、模型初始化、可训练参数和训练预算下，只增加 teacher latent-memory loss，验证集平均 stage accuracy 从 **45.0% 提升到 68.6%**，三次更新后的 final accuracy 从 **33.75% 提升到 50.0%**。

因此，在当前 recurrent MEM recipe 中，teacher memory 不是多余组件。它提供了仅靠离散 state CE 缺少的稠密状态几何监督，显著改善收敛和跨事件状态更新。这个结果证明的是当前训练方法中的工程必要性，而不是 teacher 在数学意义上不可替代。

## 严格对照设计

| 项目 | A：仅 GT state | B：GT state + teacher memory |
|---|---:|---:|
| student 输入 | 每个事件 12 帧 | 每个事件 12 帧 |
| event clips | `[18,30), [28,40), [38,50)` | 相同 |
| 初始参数 | 相同 | 相同 |
| 冻结 updater/readout | 相同 | 相同 |
| 可训练参数 | 2.424M visual encoder | 相同 |
| GT stage CE 权重 | 1.0 | 1.0 |
| teacher memory loss 权重 | 0.0 | 1.0 |
| 训练/验证 episode | 4500/500 | 相同 |
| steps / batch / seed | 1000 / 12 / 42 | 相同 |
| LR | warmup 50，`3e-4 -> 3e-5` | 相同 |

唯一改变的有效训练因素是 teacher memory loss。A 中仍计算 teacher 距离作为诊断指标，但权重为 0，不产生梯度。

Teacher 使用已验证的 symbolic event relation 生成目标 memory；student 推理路径只接收上一时刻 memory 和 12 帧连续视觉 clip，不接收 relation id、Qwen 输出或最终位置标签。

## 最终验证结果

最终验证使用两路完全相同的 20 个 batch，共 240 个固定验证样本。

| 指标 | A：仅 GT state | B：+ teacher memory | 绝对提升 |
|---|---:|---:|---:|
| stage accuracy | 45.00% | **68.61%** | **+23.61 pp** |
| final accuracy | 33.75% | **50.00%** | **+16.25 pp** |
| 第 1 次更新 | 65.00% | **81.67%** | +16.67 pp |
| 第 2 次更新 | 36.25% | **74.17%** | **+37.92 pp** |
| 第 3 次更新 | 33.75% | **50.00%** | +16.25 pp |
| stage CE | 0.9821 | **0.6822** | -30.53% |
| memory cosine distance | 1.0287 | **0.4167** | -59.49% |

A 的 final accuracy 仍接近三分类随机水平 33.33%；B 已达到 50%。最显著的收益发生在第二次 recurrent update，说明 teacher supervision 尤其有助于学习可组合的状态转移，而不只是第一事件的视觉分类。

训练中间阶段也揭示了作用机制：B 先降低 latent-memory distance、保持非退化 token state，随后在约 750--1000 step 将表示优势转化为 stage accuracy；A 到低学习率尾段才开始学习第一阶段，第二、第三阶段仍弱。

## 可以与不可以得出的结论

可以得出：

- 在当前固定 updater/readout、仅训练视觉 event encoder 的 recipe 中，teacher latent supervision 有显著价值，建议保留。
- teacher 的核心作用是稠密 representation/transition supervision，而不是为 student 推理提供额外输入。
- 只用可直接得到的 GT state 并不等价于 teacher memory：GT state 只有 3 类、每个事件一个 CE；teacher memory 对 128×64 的连续 recurrent state 提供逐 token 约束。

不可以得出：

- 不能声称 teacher 在数学意义上不可替代。更强的 end-to-end updater、辅助 transition loss 或多步 curriculum 可能不用 teacher 也能达到同等效果。
- 不能声称当前 teacher 已解决长时记忆。B 的 final accuracy 只有 50%，仍明显低于前两阶段。
- 这是 ShellGame、单 seed 的结构消融，不是 RoboMME 四任务上的最终统计结论。

## 对当前工程的建议

保留 teacher，但把它定位为 **训练期 latent-state regularizer**，而不是推理期必需模块。student 部署时仍完全移除 Qwen/teacher。

下一步最有价值的消融是：

1. 固定本实验配置跑 3 个 seed，报告均值和标准差。
2. 对第三次 update 加高权重或采用 1→2→3 step curriculum，判断 50% final 的瓶颈是不是误差累积。
3. 在四任务 RoboMME 数据上比较 `GT readout only` 与 `GT readout + teacher latent`，确认该收益是否跨任务成立。

## 复现资产

- 训练入口：`examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py`
- A checkpoint：`checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/teacher_necessity_12f_state_only_seed42_260826/999`
- B checkpoint：`checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/teacher_necessity_12f_state_plus_distill_seed42_260826/999`
- 机器可读结果：`evaluation/shellgame/teacher_memory_necessity_12f_260826/result.json`

训练使用 `/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv`。此前指定的 `/data2/hzl_workspace_for_pi/openpi-umi/.venv` 在本机是指向缺失 `/opt/conda/bin/python3.11` 的失效环境，因此本次使用当前工程内可用且能识别 8 张 A100 的环境。
