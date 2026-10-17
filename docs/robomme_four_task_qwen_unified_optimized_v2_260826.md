# RoboMME 四任务统一 Qwen 数据优化 V2（2026-08-26）

## 改动约束

本轮没有增加任务专用 head、task ID、特殊 task token 或任务路由器。四任务继续共享同一个 Qwen3-VL、LoRA、system prompt 和固定四字段输出。

## 数据修复

### Goal / focus 一致性

旧统一数据中：

- VideoUnmask 750 条训练样本有 428 条 goal/focus 冲突；
- VideoUnmaskSwap 611 条 focus 样本有 307 条冲突。

V2 保留原始 goal 作为 metadata；当辅助 focus 不在原始 instruction 中时，使用通用且 focus-consistent 的自然语言 goal。训练混合中的冲突数降为 0。统一 prompt 同时明确：存在 focus 时，focus 是本样本的查询实体，goal 只提供上下文。

### Candidate region

所有任务仍使用相同 `region_i`。prompt 增加通用 `candidate_region_count`，不提供任务名称或任务专用位置编码。

### PickXtimes 窗口

- positive 从 anchor 后 0--4 帧扩展到后 6--10 帧，必须看到稳定结果；
- incomplete 从 `anchor-1` 提前到 `anchor-4`；
- positive/incomplete frame-set Jaccard 从 46.7% 降至 16.7%；
- 两个窗口最后证据帧的间隔从 3 帧增至 12 帧；
- 旧 no-event 是 12 个重复的 frame 0；V2 改为初始 12 个不同帧和 press 完成后的 12 帧 hold；
- train no-event 从 70 条增加为 140 条，但仍使用所有任务相同的 event-temperature=0.5 采样规则。

## 训练

- 初始化与旧实验相同：`qwen3vl_videounmask_from_shellgame_replay25_B_260825/checkpoint-000300`
- 数据：四任务 + 10% 统一格式 ShellGame retention replay，共 5000 条
- 6 x A100，global batch 48，300 optimizer steps
- LR 1e-5，warmup 10，cosine decay
- 最终 val loss 0.0453，teacher-forced token accuracy 98.03%
- 18 个 contract/state/variable-length 测试全部通过

## 扩大 Dev 评估

每任务 48 条、共 192 条 event-balanced clips：

| 模型 | Exact | Event | Field | Valid |
|---|---:|---:|---:|---:|
| 旧统一模型直接应用 V2 prompt/data | 34.4% | 47.4% | 72.2% | 89.1% |
| V2 clean-300 | **59.9%** | **74.5%** | **87.8%** | **100%** |

V2 clean-300 逐任务：

| 场景 | Exact | Event | Valid |
|---|---:|---:|---:|
| VideoUnmask | 83.3% | 100% | 100% |
| VideoUnmaskSwap | 64.6% | 87.5% | 100% |
| VideoPlaceOrder | 39.6% | 58.3% | 100% |
| PickXtimes | 52.1% | 52.1% | 100% |

## 锁定 Test 评估

所有优化决策完成后才构建并运行 test。test 与 train/dev 各含每任务独立的 15 个 episode；评估每任务 48 条，共 192 条，seed 260829。

| 模型 | Exact | Event | Field | Valid |
|---|---:|---:|---:|---:|
| 旧统一控制模型 | 45.8% | 58.3% | 77.8% | 92.2% |
| V2 clean-300 | **61.5%** | **76.6%** | **87.5%** | **100%** |

V2 clean-300 test：

| 场景 | Exact | Event | Valid |
|---|---:|---:|---:|
| VideoUnmask | 85.4% | 100% | 100% |
| VideoUnmaskSwap | 50.0% | 83.3% | 100% |
| VideoPlaceOrder | 54.2% | 66.7% | 100% |
| PickXtimes | 56.3% | 56.3% | 100% |

PickXtimes test 的主要变化：

- no-event：0% -> 60.0%；
- pick：10.0% -> 40.0%；
- place：11.1% -> 55.6%；
- press：22.2% -> 77.8%；
- overall exact：25.0% -> 56.3%。

## 判断

锁定 test 支持之前的因果分析：统一格式本身不是主要退化原因，冲突 goal 与过度重叠/重复帧窗口才是关键。仅修改通用数据和 prompt 后，四任务都提升，且合法率达到 100%。

当前仍不能宣称四任务 teacher 已全部通过：

- VideoUnmask event=100%，已达到局部教师门槛；
- VideoUnmaskSwap event=83.3%，接近但略低于建议的 85%；
- VideoPlaceOrder event=66.7%，主要瓶颈仍是 region 参数与少量 hard swap label；
- PickXtimes event=56.3%，虽大幅提升，仍需提高 pick/incomplete 的边界稳定性。

因此暂不直接接四任务 action。下一阶段应先做 state rollout，而不是继续只看单 clip exact：把 Qwen 事件按 episode 顺序送入同一个 recurrent state machine，统计 final state、full state sequence 和 duplicate-event propagation。

## 产物

- 最终 adapter：`checkpoints/qwen3vl_robomme_four_task_unified_optimized_v2_clean300_260826/final`
- V2 Pick 数据：`artifacts/pickxtimes_qwen3vl_local_events_stable_v2_seed260826`
- V2 统一数据：`artifacts/robomme_qwen_unified_events_optimized_v2_seed260826`
- V2 混合数据：`artifacts/robomme_four_task_qwen_unified_optimized_v2_mixture_seed260826`
- Dev 48/task：`artifacts/robomme_four_task_qwen_unified_optimized_v2_eval_seed260826/clean300_n48.summary.json`
- Test 48/task：`artifacts/robomme_four_task_qwen_unified_optimized_v2_eval_seed260826/clean300_test_n48.summary.json`
