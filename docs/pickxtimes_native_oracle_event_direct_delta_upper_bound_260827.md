# PickXTimes 原生 simulator event + oracle correction + direct teacher delta 上界实验

日期：2026-08-27

## 实验问题

本实验用于拆解当前 recurrent MEM 与 GroundSG 之间的差距：

1. 如果事件时刻由 simulator 特权信息准确给出，MEM 是否仍然更新失败？
2. 在正确事件时刻直接监督 student 补偿 teacher memory delta，是否能显著提高 transition state 与完整记忆轨迹？
3. 收益究竟来自 privileged correction，还是普通的继续训练？

这是一项诊断性上界实验。oracle event mask 在部署时不可用，因此实验结果不能直接作为最终方法结果。

## 原生事件标签

此前 fixed-chunk 的 `state_change_mask` 来自
`robomme_qwen_unified_events_optimized_v2` 中采样视觉事件窗的最后一帧，并不是纯 simulator 时刻。

本实验改为从 RoboMME H5 的逐帧特权字段读取：

- `info/is_subgoal_boundary`
- 初始 boundary 表示第一个 subgoal 开始；
- 后续 boundary 分别表示 Pick、Place、Press canonical event 完成；
- 原生 boundary 数量必须与该 episode 的 canonical event 数完全一致。

全量审计结果：train/dev/test 的全部 PickXTimes episode 均通过数量和因果范围检查，且原生事件时刻均未超过已有视觉特征缓存的时间范围。

生成的数据目录：

`artifacts/robomme_four_task_fixed_chunk_sequences_pick_native_v1_260827`

## Correction 结构与损失

基础递归更新保持不变：

```text
m_base(t) = m(t-1) + g_write(t) * (candidate(t) - m(t-1))
```

新增零初始化 correction 分支：

```text
c(t) = MLP(LN([m(t-1), candidate(t), candidate(t)-m(t-1)]))
m(t) = m_base(t) + g_oracle(t) * c(t)
```

只在 simulator event chunk 上监督直接 teacher delta：

```text
target_delta(t) = stop_gradient(teacher_memory(t) - m_base(t))
L_delta = MSE(c(t), target_delta(t))
```

稳定化版本使用：

```text
L_total = L_canonical + L_delta
```

其中 `L_canonical` 保留原 memory trajectory、state readout、final、transition 与 hold consistency 约束。

## 对照设置

共同设置：

- PickXTimes 单任务；
- 12 帧固定非重叠 chunk；
- 从既有 PickXTimes 最佳 checkpoint 恢复；
- 400 steps，batch size 4；
- learning rate `3e-5 -> 3e-6`；
- 同一数据划分和随机种子；
- readout 与指标口径完全一致。

三组对照：

1. `Baseline`：恢复 checkpoint，step 0，不继续训练；
2. `Canonical control`：普通 canonical loss 继续训练，无 correction、无 oracle；
3. `Native oracle + direct delta`：canonical loss + simulator oracle correction + direct teacher delta。

## 主要结果

### 锁定 test split（15 episodes）

| 方法 | Final | Transition | Hold / no-change | All-state |
|---|---:|---:|---:|---:|
| Baseline，step 0 | 80.00% | 38.14% | 35.42% | 35.81% |
| Canonical control，step 400 | 73.33% | 36.08% | 44.07% | 42.94% |
| Native oracle + direct delta，step 400 | **80.00%** | 34.02% | **47.29%** | **45.41%** |

相对 Baseline，native oracle 组：

- Final：`+0.00 pp`；
- Transition：`-4.12 pp`；
- Hold：`+11.86 pp`；
- All-state：`+9.61 pp`。

相对同训练预算的 Canonical control，native oracle 组：

- Final：`+6.67 pp`；
- Transition：`-2.06 pp`；
- Hold：`+3.22 pp`；
- All-state：`+2.47 pp`。

### Dev split 的最终结果

| 方法 | Final | Transition | Hold / no-change | All-state |
|---|---:|---:|---:|---:|
| Baseline，step 0 | 100.00% | 41.41% | 38.61% | 39.01% |
| Canonical control，step 400 | 86.67% | 34.34% | 44.90% | 43.38% |
| Native oracle + direct delta，step 400 | 93.33% | 36.36% | **47.45%** | **45.85%** |

## 必要的负结果

只训练 correction 参数、只优化 `L_delta` 时，即使降低学习率也会递归崩溃：

| 设置 | Dev Final | Dev Transition | Dev All-state |
|---|---:|---:|---:|
| 初始 step 0 | 100.00% | 43.43% | 41.63% |
| Delta-only，LR `3e-4`，step 400 | 0.00% | 11.11% | 20.23% |
| Delta-only，LR `1e-4`，step 400 | 0.00% | 13.13% | 20.67% |

原因不是简单的学习率过高。Correction 会改写后续所有 recurrent states；即使只训练 correction 参数，也会让原 readout 面对新的 latent trajectory。缺少 canonical/readout 约束时，teacher latent 欧氏距离下降并不保证语义可解码。

## 结论

1. **特权 simulator 信息有用。** 相比严格 continuation control，oracle-delta 在 test 上提高 final、hold 和 all-state，证明环境特权信息能够向 recurrent MEM 提供有效监督。
2. **当前收益主要来自 persistence，而不是 event transition。** Hold 和 all-state 明显提高，但 transition 下降。因此当前流程不能解释 GroundSG 的高 transition 表现，也不能称为 GroundSG 上界已达到。
3. **递归结构不是天然无效。** Oracle correction 改善了跨时间状态保持，说明 recurrent state 能承载额外知识；问题更集中在事件瞬间的更新目标、latent 坐标兼容性和 loss 冲突。
4. **直接 full-latent delta 不是理想最终目标。** 稳定化训练结束时 prediction RMS 约 `0.0216`，target RMS 约 `0.7997`，correction 只实现了目标幅度的一小部分；canonical loss 在保护 readout 的同时强烈抑制了 direct delta。
5. **当前 checkpoint 不应直接接 action。** 它在推理时依赖 privileged oracle event mask，而且 transition 仍未改善。

## 下一步建议

下一项最有信息量的实验不是继续盲目加步数，而是把 correction 的训练目标从 full teacher latent 改为语义对齐目标：

- event chunk 上直接监督 `state readout` / teacher logits；
- 对 correction 加幅度 curriculum 或 trust region；
- transition loss 与 post-event hold loss分开加权；
- 先用 oracle mask 学会可解码 correction，再将 oracle mask 蒸馏给视觉 soft gate。

这可以验证：transition 瓶颈究竟来自 latent basis mismatch，还是视觉 evidence 本身不足。

## 产物

- 模型实现：`src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- 训练入口：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- 原生标签构建：`scripts/mem/build_robomme_four_task_fixed_chunk_sequences.py`
- 新结构兼容评估：`scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py`
- Native oracle run：`checkpoints/pickxtimes_native_oracle_event_correction_canonical_lr3e5_seed260831_260827`
- Native control run：`checkpoints/pickxtimes_native_canonical_control_lr3e5_seed260831_260827`

