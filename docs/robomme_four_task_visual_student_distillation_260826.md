# RoboMME 四任务统一 Teacher → Visual Recurrent Student 蒸馏实验

日期：2026-08-26

## 结论

统一 canonical GT teacher 可以有效蒸馏到一个不接收 Qwen 输出、GT event 或 GT state 的直接视觉 recurrent student。最佳 checkpoint 为 step 2200；正式训练结束时的 held-out test field / full-state / final / full-sequence exact 分别为 91.8% / 57.2% / 46.7% / 26.7%。

当前结果证明了 teacher latent 作为视觉 MEM 的训练目标可行，但还不是最终在线系统：训练与测试使用 GT 定位的 12 帧 causal event window，线上 event trigger 的误差尚未计入。

## Student 推理契约

每个 episode 只在开始时注入统一 goal token：

- task id；
- 最多两个目标颜色；
- required count；
- queried ordinal；
- candidate region count。

每次 recurrent 更新只输入：

1. 上一步 student memory；
2. 当前 causal 12-frame visual window 的冻结 PaliGemma/SigLIP patch token；
3. event trigger 给出的 update mask。

Student 不输入 GT event type/argument、GT state、teacher memory、Qwen JSON 或 Qwen latent。Teacher 和 teacher readout 只在离线训练/评估时存在。

## 数据流水线

1. 从四任务统一 event manifest 选择与 teacher state transition 一一对应的 causal window。
2. 对每个 window 的 12 帧提取冻结视觉 backbone token。
3. 将 16×16 patch grid 平均池化为 4×4，缓存为 float16。
4. episode 内按时间顺序组织最多 12 个 event window；padding step 通过 step mask 严格保持 memory 不变。
5. 使用 canonical teacher cache 的 `[episode, state, 128, 64]` memory 作为蒸馏目标。`diagnostic_rollout_memory` 不参与 student 训练。

数据量：

| Split | Episodes | Event windows |
|---|---:|---:|
| train | 280 | 1183 |
| dev | 60 | 246 |
| test | 60 | 246 |

四任务在每个 split 中 episode 数严格平衡。

## 模型与损失

Student 使用一个共享 VisualWindowEncoder、一个共享 goal initializer、一个共享 RecurrentMemoryUpdater；没有 task-specific head。

总损失：

```text
L = 1.0 * L_memory + 0.5 * L_state

L_memory = token-aligned cosine + 0.1 * MSE
```

Canonical memory 的前 19 个 token 对应统一 state fields，因此这 19 个 token 在 memory loss 中使用 4× 权重，避免其余共享基底 token 稀释语义梯度。

`L_state` 使用冻结的 teacher shared readout 解码 student memory，再计算 masked state cross entropy。冻结 readout 的作用是要求 student memory 落入 teacher 已验证的可读 latent basis，而不是让一个可训练新 head 迁就任意 student latent。

## 正式训练

- steps：3000；
- batch size：8，四任务每 batch 平衡；
- optimizer：AdamW；
- peak LR：3e-4；
- best checkpoint：按 dev final → sequence → state exact 的字典序选择；
- best step：2200。

## Held-out test

| Task | Field | Full state | Final state | Full sequence |
|---|---:|---:|---:|---:|
| Overall | 91.8% | 57.2% | 46.7% | 26.7% |
| PickXtimes | 97.4% | 80.4% | 93.3% | 46.7% |
| VideoPlaceOrder | 92.5% | 67.3% | 46.7% | 40.0% |
| VideoUnmask | 91.1% | 46.7% | 20.0% | 20.0% |
| VideoUnmaskSwap | 83.8% | 28.7% | 26.7% | 0.0% |

Teacher canonical state/sequence/final 为 100%，因此表中差距全部来自视觉 student 的感知与递推，不是 teacher 标签噪声。

## 视觉依赖诊断

| Test input | Full state | Final state | Full sequence |
|---|---:|---:|---:|
| Normal video | 57.5% | 46.7% | 26.7% |
| Zero video | 27.1% | 11.7% | 11.7% |
| Reverse event-window order | 52.0% | 40.0% | 21.7% |
| Shuffle video across episodes | 51.0% | 33.3% | 23.3% |

清零视频使 final exact 从 46.7% 降到 11.7%，说明 student 确实使用视觉证据，而不是只依赖 goal/step count。倒序下降较小，暴露了当前数据中部分任务的最终状态对事件顺序不敏感，以及 temporal hard negatives 不足；这应作为下一轮优化重点。

## 局限与下一步

1. 将现有离线 GT event window 替换为滑动窗口 event trigger 产生的窗口，报告 trigger + memory 的端到端指标。
2. 为 VideoUnmaskSwap 加入 swap-order hard negatives、相反 swap pair 和跨 episode 匹配负例，强化时序/对象绑定。
3. 对 VideoUnmask 增加颜色/region counterfactual，避免目标覆盖阶段的确定性先验占主导。
4. 在不增加 task-specific head 的前提下，尝试更高空间分辨率（8×8）或多尺度 evidence，重点验证 VideoUnmaskSwap。

## 产物

- Student model：`src/openpi/tasks/robomme/unified_visual_student.py`
- Student sequence builder：`scripts/mem/build_robomme_four_task_visual_student_sequences.py`
- Frozen visual cache：`scripts/mem/cache_robomme_four_task_visual_features.py`
- Distillation trainer：`scripts/mem/train_robomme_four_task_visual_student_distillation.py`
- Visual dependence evaluator：`scripts/mem/eval_robomme_four_task_visual_student_distillation.py`
- Best checkpoint：`checkpoints/robomme_four_task_visual_student_distilled_v1_260826/best/params`
- Training result：`checkpoints/robomme_four_task_visual_student_distilled_v1_260826/result.json`
- Dependence result：`checkpoints/robomme_four_task_visual_student_distilled_v1_260826/test_visual_dependence.json`
