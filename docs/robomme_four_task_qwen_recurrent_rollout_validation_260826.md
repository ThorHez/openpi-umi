# RoboMME 四任务统一 Qwen recurrent rollout 验证（2026-08-26）

## 结论

当前统一 Qwen3-VL LoRA 可以稳定输出合法的四字段 JSON，但除 VideoUnmask 外，尚不能作为可靠的在线 recurrent memory updater，也不建议现在直接接入四任务 action 联训。

在 60 个锁定测试 episode（每任务 15 个）、470 个规范因果窗口上：

- JSON valid rate：100%。
- clip event accuracy：micro 61.06%，task macro 72.74%。
- clip exact accuracy：micro 50.21%，task macro 58.70%。
- 使用状态约束和重叠去重后，full-state sequence accuracy：micro 30.00%，task macro 38.41%。
- 最终完整状态准确率：30.00%。
- 最终任务答案准确率：35.00%。

VideoUnmask 已达到可继续做 memory-only/action 小规模接口实验的水平；PickXTimes、VideoPlaceOrder 和 VideoUnmaskSwap 仍需先优化事件边界与空间字段。

## 验证设置

- 模型：`checkpoints/qwen3vl_robomme_four_task_unified_optimized_v2_clean300_260826/final`
- 测试清单：`artifacts/robomme_four_task_qwen_unified_optimized_v2_mixture_seed260826/test.jsonl`
- 统一输出：`{"event": ..., "entity": ..., "region_a": ..., "region_b": ...}`
- 不使用任务 ID、任务专用 token 或任务专用 head。
- 每个 episode 按时间顺序输入：每个事件的一个规范 positive clip、对应 causal incomplete clip，以及 episode 的 hold/no-event clip。
- updater 使用统一事件合同，但保持任务状态约束：非法状态转移会被拒绝。
- overlap dedup：如果两个相互重叠的 clip 给出完全相同的 state-changing event signature，只提交一次。

先用 oracle event 执行了相同 rollout。60 个 episode 的完整状态序列、最终状态和答案均为 100%，说明验证器的窗口排序和状态更新规则是自洽的。

## 主要结果

| 任务 | clips | event acc | exact acc | full-state sequence | final state | final answer |
|---|---:|---:|---:|---:|---:|---:|
| PickXTimes | 224 | 39.73% | 39.73% | 25.45% | 13.33% | 13.33% |
| VideoPlaceOrder | 95 | 66.32% | 50.53% | 15.79% | 6.67% | 26.67% |
| VideoUnmask | 45 | 100.00% | 88.89% | 82.22% | 80.00% | 80.00% |
| VideoUnmaskSwap | 106 | 84.91% | 55.66% | 30.19% | 20.00% | 20.00% |

重叠去重相对于只做状态约束的 raw rollout：

- task-macro full-state sequence：35.10% → 38.41%。
- final state：25.00% → 30.00%。
- final answer：35.00% → 35.00%。
- 共抑制 30 次重叠提交：Pick 10、VideoPlaceOrder 14、VideoUnmaskSwap 6。

去重有效，但不足以解决主要错误。

## 错误拆解

### 1. PickXTimes：完成边界和 pick/place 类型都不稳定

- 97 个 incomplete clip 中有 67 个被错误预测为 state-changing event，false-commit rate 为 69.07%。
- 97 个真正 completed clip 中只有 42 个 event 正确，accuracy 为 43.30%。
- 97 组 incomplete→completed 边界中：clean 10，early duplicate 32，提前触发且 positive 又预测错 18，positive 漏检或类型错 37。
- 常见混淆是 `pick_complete ↔ place_complete`；状态机虽然会拒绝一部分非法转移，但无法修复合法却发生在错误时刻的转移。

特别需要注意：使用 oracle 完成窗口屏蔽所有提前触发后，最终状态反而是 0%。这说明当前 13.33% 的 final success 有一部分来自“提前误报恰好补偿后续漏检”，不具备在线鲁棒性。

### 2. VideoPlaceOrder：event 尚可，但 region_a 是主瓶颈

- 39 个 place positive 中，event 正确 27 个（69.23%），四字段 exact 仅 13 个（33.33%）。
- 所有 clip 的 `region_a` field accuracy 为 53.68%。
- incomplete-place false-commit rate 为 43.59%。
- 即使使用 oracle 完成时刻，最终完整状态仍只有 6.67%，最终查询答案为 33.33%。

这表明单纯调 event trigger 不能解决问题，必须加强统一 region grounding。

### 3. VideoUnmask：当前最稳定

- event accuracy 100%。
- `region_a` accuracy 88.89%。
- final state/final answer 均为 80%。
- oracle 完成时刻下仍为 80%，说明主要是少量空间格子识别错误，而不是累计触发污染。

### 4. VideoUnmaskSwap：swap event 可识别，但交换区域对不准

- swap positive event accuracy 为 70.37%，四字段 exact 只有 22.22%。
- 全部 clip 的 `region_a` / `region_b` accuracy 分别为 56.60% / 77.36%。
- incomplete-swap false-commit rate 为 29.63%。
- oracle 完成时刻下 final state/final answer 仍为 20%。

## 为什么单窗口分数不能直接代表 memory 可用性

单窗口验证只判断当前 clip 的回答；recurrent rollout 会将回答提交到持久状态。一次合法但错误的 event 或 region 会污染后续所有状态，因此 final state 会显著低于 clip-level event accuracy。这正是本次验证希望暴露的问题。

本实验使用的是每个真实事件周围的规范 causal windows，还不是 20 Hz dense online sliding windows，也不是 simulator action success。dense sliding inference 还会带来更多相邻窗口和重复触发，所以当前结果不能被解释为闭环成功率。

## 下一步建议

先做一轮仍保持“统一格式、统一模型、无任务专用 head”的数据优化，再接 action：

1. 使用同一 episode 的成对 boundary curriculum：`far incomplete → near incomplete → just completed → post-completion hold`，并增加 completion 前的时间间隔，避免相互重叠帧拥有冲突标签。
2. 对 PickXTimes 加强 pick/place 的相位可见性，按混淆矩阵做 hard-negative 重采样；目标是 positive event accuracy ≥80%，incomplete false-commit ≤15%。
3. 给所有需要 region 的任务统一叠加 row-major region overlay/anchor 标记；这不是任务专用 head，仍使用共享四字段输出。
4. 用 episode-sequence 采样训练或做 consistency regularization，约束相邻窗口只在可见完成边界发生一次状态改变。
5. 重新跑本 rollout；进入 action 联训的建议门槛为：各任务 final answer ≥80%，final state ≥70%，full-state sequence ≥70%，且 dense-window 重复触发测试通过。

## 产物

- rollout evaluator：`scripts/mem/eval_robomme_four_task_qwen_recurrent_rollout.py`
- diagnostics：`scripts/mem/analyze_robomme_four_task_qwen_recurrent_rollout.py`
- oracle rollout：`artifacts/robomme_four_task_qwen_unified_optimized_v2_rollout_seed260826/oracle.jsonl`
- Qwen rollout：`artifacts/robomme_four_task_qwen_unified_optimized_v2_rollout_seed260826/qwen_clean300.jsonl`
- summary：`artifacts/robomme_four_task_qwen_unified_optimized_v2_rollout_seed260826/qwen_clean300.summary.json`
- detailed diagnostics：`artifacts/robomme_four_task_qwen_unified_optimized_v2_rollout_seed260826/qwen_clean300.diagnostics.json`
