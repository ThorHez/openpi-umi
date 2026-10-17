# RoboMME 四任务 Qwen 统一 Contract 实验（2026-08-26）

## 目标

验证不增加任务专用 head、task ID、特殊 task token 的前提下，四个 RoboMME 场景能否共享：

- 一个 Qwen3-VL 与一个 LoRA；
- 一个 system prompt 和一个自然语言 goal 模板；
- 一个固定四字段 JSON 输出；
- 一个统一的局部事件集合与 candidate region 语义。

ShellGame 10% retention replay 也被转换到相同 contract；训练 batch 中不存在旧格式旁路。

## 统一接口

输入只包含视频、原始自然语言 goal 和可选的通用 `focus_entity`，不包含任务名称：

```text
Goal: {natural_language_goal}
Focus entity: {entity_or_none}
```

输出固定为：

```json
{"event":"swap_complete","entity":null,"region_a":"region_0","region_b":"region_2"}
```

四个字段始终存在。`region_i` 在所有场景中都表示候选空间位置，按屏幕 row-major 排列。VideoPlaceOrder hard episode 中出现的新空位置也进入同一 region 序列。

## 训练

### 第一阶段：统一格式适配

- 初始化：`qwen3vl_videounmask_from_shellgame_replay25_B_260825/checkpoint-000300`
- 四任务比例：15% / 25% / 30% / 20%，ShellGame replay 10%
- 6 x A100，global batch 48，100 optimizer updates，LR 1e-5
- 结果：val loss 0.0914，teacher-forced token accuracy 96.75%

### 第二阶段：通用事件温度采样

100-step 生成评估显示稀有事件被 `incomplete/place` 吞并。没有增加任务结构，而是对每个数据源使用相同的事件温度采样：

```text
p(event) proportional to count(event)^0.5
```

从第一阶段 final 继续 200 updates，LR 5e-6。最终 val loss 0.0566，token accuracy 97.39%。总训练量约 300 optimizer updates。

## Held-out 生成结果

每任务从未见 episode 中按 event round-robin 选择 24 条，共 96 条，greedy generation。

| 设置 | Overall exact | Overall event | Valid |
|---|---:|---:|---:|
| 旧 LoRA，对统一格式零样本 | 4.2% | 7.3% | 16.7% |
| 统一 100-step | 21.9% | 47.9% | 95.8% |
| 统一约 300-step + event temperature | **40.6%** | **63.5%** | **100%** |

最终逐任务结果：

| 场景 | Exact | Event accuracy | Field accuracy | Valid |
|---|---:|---:|---:|---:|
| VideoUnmask | 70.8% | 100% | 90.3% | 100% |
| VideoUnmaskSwap | 41.7% | 83.3% | 73.6% | 100% |
| VideoPlaceOrder | 33.3% | 54.2% | 80.6% | 100% |
| PickXtimes | 16.7% | 16.7% | 87.5% | 100% |

与上一版四套 contract 的 100-step 模型比较：

| 场景 | 多 contract exact | 统一 contract exact | 多 contract event | 统一 contract event |
|---|---:|---:|---:|---:|
| VideoUnmask | 87.5% | 70.8% | 95.8% | 100% |
| VideoUnmaskSwap | 12.5% | **41.7%** | 45.8% | **83.3%** |
| VideoPlaceOrder | 12.5% | **33.3%** | 25.0% | **54.2%** |
| PickXtimes | **50.0%** | 16.7% | **50.0%** | 16.7% |
| Overall | 40.6% | 40.6% | 54.2% | **63.5%** |

## 判断

统一 contract 的实验是正结果，但不是“四任务已解决”：

1. 不使用 task head/task ID，也能达到与多 contract 相同的 overall exact，同时 event accuracy 提升 9.3 个百分点、合法率达到 100%。
2. 最大收益出现在原先协议混淆最严重的 VideoUnmaskSwap 和 VideoPlaceOrder。
3. VideoUnmask 的事件判断保持 100%，但 region exact 下降，说明 candidate region 的视觉排序还需加强。
4. PickXtimes 明显退化。错误主要是 pick/place/incomplete/no-event/press 互相混淆，而不是 JSON 或 entity 字段错误；继续只优化格式没有意义。
5. 当前仍不满足四任务 Qwen teacher cache 的统一门槛，尤其不能直接据此接 action。

下一步保持统一模型结构，优先改进数据：增加 PickXtimes 边界附近的完整 event window、真正静态 no-event、button press 结果帧，并用同一个 event-temperature 规则重新训练。不得为 PickXtimes 添加专用 head。

## 产物

- 统一 contract：`src/openpi/tasks/robomme/qwen3vl_unified_event_contract.py`
- 转换器：`scripts/mem/build_robomme_qwen_unified_event_manifests.py`
- 混合清单构建器：`scripts/mem/build_robomme_four_task_qwen_unified_mixture.py`
- 统一数据：`artifacts/robomme_qwen_unified_events_seed260826`
- 温度平衡混合数据：`artifacts/robomme_four_task_qwen_unified_balanced_mixture_seed260826`
- 最终 adapter：`checkpoints/qwen3vl_robomme_four_task_unified_event_balanced_continue200_260826/final`
- 最终生成记录：`artifacts/robomme_four_task_qwen_unified_eval_seed260826/pilot300_balanced.jsonl`
- 最终汇总：`artifacts/robomme_four_task_qwen_unified_eval_seed260826/pilot300_balanced.summary.json`
