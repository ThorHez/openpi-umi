# RoboMME 固定分块 + 可学习 Soft Write Gate 实验

日期：2026-08-26

## 结论

在保持 12 帧固定、不重叠 chunk 和原有 teacher/state loss 不变的条件下，仅加入一个端到端学习的标量 write gate，可明显缓解长序列中的 memory drift：held-out test 全状态准确率从 21.2% 提升到 46.0%，final 从 6.7% 提升到 13.3%。

该 gate 不是显式 event detector，不使用 event/state GT，也不丢弃窗口。它学习每个固定 chunk 应写入多少：

```text
candidate_t = F(memory_{t-1}, visual_chunk_t)
gate_t = sigmoid(MLP(pool(memory_{t-1}), pool(visual_chunk_t)))
memory_t = memory_{t-1} + gate_t * (candidate_t - memory_{t-1})
```

## 严格控制变量

- train/dev/test：280/60/60 episodes；
- chunk/stride：12/12 frames，无滑动窗口；
- 最大 96 个 recurrent steps；
- batch size：4，四任务平衡；
- 训练 2000 steps，peak LR 3e-4；
- 相同 canonical teacher memory、frozen teacher readout 和 state loss；
- 相同随机种子 260826；
- gate 初始 bias：-2，初始值 sigmoid(-2)=0.1192；
- gate 没有 BCE、event mask 或其它 gate supervision；
- 模型输入不包含 GT event/state，sequence mask 只标记变长序列 padding。

最佳 checkpoint 由 dev `(final, transition, sequence, all-state)` 字典序选择，出现在 step 1700。

原指定环境 `/data2/hzl_workspace_for_pi/openpi-umi/.venv` 的 Python 链接已失效，本实验实际使用 `/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv`（JAX 0.5.3、Flax 0.10.2）和 A100 GPU 6。

## Held-out Test 对照

| Method | Field | All state | Change state | No-change state | Final | Full sequence |
|---|---:|---:|---:|---:|---:|---:|
| Fixed chunk，无 gate | 79.5% | 21.2% | 6.7% | 23.5% | 6.7% | 0.0% |
| Fixed chunk，soft gate | **89.9%** | **46.0%** | **20.1%** | **50.0%** | **13.3%** | 0.0% |
| 绝对提升 | +10.5 pp | +24.8 pp | +13.4 pp | +26.5 pp | +6.6 pp | 0.0 pp |

结果支持 soft gate 对 memory drift 有显著作用，但仍不足以完整复原长事件序列。

## 分任务 Test

| Task | All state（无 gate → gate） | Change（无 gate → gate） | No-change（无 gate → gate） | Final（无 gate → gate） |
|---|---:|---:|---:|---:|
| PickXtimes | 18.6% → **46.3%** | 7.2% → **28.9%** | 20.5% → **49.2%** | 0.0% → 0.0% |
| VideoPlaceOrder | 19.9% → **52.7%** | 0.0% → **20.0%** | 21.1% → **54.7%** | 0.0% → **20.0%** |
| VideoUnmask | 53.5% → 53.5% | 20.0% → 20.0% | 67.6% → 67.6% | 20.0% → 20.0% |
| VideoUnmaskSwap | 18.8% → **20.2%** | 3.5% → **5.3%** | 24.4% → **25.6%** | 6.7% → **13.3%** |

PickXtimes 的中间状态保持和变化识别明显改善，但 final 仍为 0，说明 gate 抑制漂移后，计数/终态读出仍是独立瓶颈。VideoUnmask 已在原模型上达到相同结果，因此本次总体收益主要来自较长、更新次数更多的任务。

## Gate 是否真的学会了选择性写入

在最佳 checkpoint 的完整 held-out test 上：

| Chunk type | Mean gate |
|---|---:|
| State-change chunk | 0.02135 |
| State-hold chunk | 0.01737 |
| Relative difference | **+22.9%** |

初始时两者均为 0.1192。训练后 gate 首先把总体写入强度降低约一个数量级，随后学习到变化 chunk 比保持 chunk 更大的软写入量。因此它同时完成了：

1. 减少普通 chunk 的累计扰动；
2. 对包含状态变化的视觉证据提高相对写入强度。

这仍是 soft preference，不是可解释的硬 event switch。

## 视觉与时序诊断

| Input | All state | Change state | Final | Mean gate |
|---|---:|---:|---:|---:|
| Normal | 46.0% | 20.1% | 13.3% | 0.0179 |
| Zero video | 14.4% | 2.2% | 0.0% | 0.1392 |
| Reverse chunks | 43.5% | 19.2% | 13.3% | 0.0181 |
| Shuffle video across episodes | 42.5% | 18.8% | 11.7% | 0.0370 |

- 清零视觉后性能显著下降，确认模型使用视觉证据；
- 倒序下降很小，说明事件顺序建模仍不充分；
- zero-video 时 gate 异常增大，说明它不是通用的“低置信度关闭”机制，对分布外证据没有校准；
- 因而本次提升应解释为 write-rate control / drift suppression，而不是已解决 event detection。

## 结论与决策

应保留 soft write gate：它几乎不增加接口复杂度、不需要滑动窗口或显式 detector，并在严格对照中显著提升状态保持、变化状态和最终状态表现。

但当前 checkpoint 还不适合直接作为最终 action-memory 版本，因为：

1. full sequence 仍为 0；
2. PickXtimes final 仍为 0；
3. 倒序敏感性太弱；
4. gate 对 zero-video 分布外输入会错误打开。

## 产物

- 模型：`src/openpi/tasks/robomme/unified_fixed_chunk_student.py`
- 单元测试：`src/openpi/tasks/robomme/unified_fixed_chunk_student_test.py`
- 训练入口：`scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- 诊断入口：`scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py`
- 最佳 checkpoint：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_v1_260826/best/params`
- 完整结果：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_v1_260826/result.json`
- 训练日志：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_v1_260826/metrics.jsonl`
- 视觉/时序诊断：`checkpoints/robomme_four_task_fixed_chunk_soft_gate_v1_260826/test_visual_dependence.json`

