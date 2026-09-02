# RoboMME 四任务 Qwen 局部事件 Pilot（2026-08-26）

## 结论

四任务统一 LoRA 的方向可行，但当前 100-step pilot 只能证明 VideoUnmask 已稳定、PickXtimes 开始学到；VideoUnmaskSwap 和 VideoPlaceOrder 尚未达到可作为 recurrent MEM teacher 的水平，不能直接接 action。

本次是 Qwen 局部事件教师实验，不是 recurrent MEM 或 closed-loop action 成功率。

## 设置

- 初始化：`qwen3vl_videounmask_from_shellgame_replay25_B_260825/checkpoint-000300`
- 数据：5000 条，VideoUnmask / VideoUnmaskSwap / VideoPlaceOrder / PickXtimes / ShellGame = 15% / 25% / 30% / 20% / 10%
- 输入：每条 12 个因果帧；demo 长度由 H5 动态检测，不假设固定 60 帧
- 优化：6 x A100，global batch 48，100 optimizer steps，LoRA 19.08M 参数，LR 1e-5，10-step warmup
- 验证：每任务 24 条 held-out episode clip，并按 event 类别 round-robin 平衡抽样，共 96 条；greedy generation
- 最终训练指标：val loss 0.168，teacher-forced token accuracy 93.62%

## 生成结果

| 场景 | 旧 LoRA exact | 100-step exact | 100-step event | 100-step valid | 判断 |
|---|---:|---:|---:|---:|---|
| VideoUnmask | 87.5% | 87.5% | 95.8% | 100% | 可用，支持可变 demo 长度 |
| VideoUnmaskSwap | 4.2% | 12.5% | 45.8% | 50.0% | 学到事件类型，但参数/JSON 不稳定 |
| VideoPlaceOrder | 0.0% | 12.5% | 25.0% | 91.7% | place 有信号，cell 与 swap 仍弱 |
| PickXtimes | 20.8% | 50.0% | 50.0% | 100% | 明显可学，仍未过 teacher 门槛 |
| 四任务总体 | 28.1% | 40.6% | 54.2% | 85.4% | 多任务训练有效，但总体 token accuracy 明显虚高 |

逐事件观察：

- VideoUnmask：covered 8/8 exact，insufficient 8/8，visible 5/8 exact；event 为 7/8。
- VideoUnmaskSwap：visible event 6/6，covered event 3/6，swap event 2/6；slot 参数 exact 仍低。
- VideoPlaceOrder：place event 4/6，但 cell exact 1/6；incomplete 2/6，no-event 与 swap 均 0/6。
- PickXtimes：no-event 5/5，place 4/5，pick 2/5，incomplete 1/5，press 0/4。

样本仅 24/任务，因此这些数字用于 go/no-go pilot，不作为论文最终结果或置信区间充分的 benchmark 成绩。

## 实验中发现并修复的问题

1. PEFT 0.19.1 会导入当前 Transformers 4.57.1 不存在的 `EmbeddingParallel`。训练器现在仅在模型没有 tensor-parallel metadata 时跳过无用的 TP shard 步骤，不修改 LoRA 权重。
2. 原训练器在 gradient accumulation=4 时每个 micro-batch 都推进 scheduler，100-step cosine 在约 25 个真实更新内耗尽。现在只在 `accelerator.sync_gradients` 时推进一次，并通过 4:1 accumulation smoke test。
3. VideoPlaceOrder 的 H5 grounded 坐标为 `<y,x>`；旧 probe 使用了转置坐标。
4. 初版 VideoPlaceOrder local label 错把执行指令的目标颜色附到每个示范 placement。现改为 `place_complete(target_cell)`，ordinal 由 recurrent `written_count+1` 累积，目标颜色只进入 goal/action。

修正第 4 点后，相同第 50 步的混合 val loss 从错误标签版 0.463 降到 0.286；生成评估中的 VideoPlaceOrder event accuracy 从 8.3% 提升到 25.0%。

## 下一步门槛

暂不生成四任务 Qwen pseudo-cache，也不接 action。下一轮优先处理两个弱项：

1. VideoUnmaskSwap 将自由 JSON 生成拆成 event 分类、slot-a 分类、slot-b 分类，或使用约束解码；先把 valid rate 从 50% 提到 95% 以上。
2. VideoPlaceOrder 对 place cell 改为候选 target 分类，并补齐全部 hard swap pair；当前仅 14 个 target-relevant hard episode 有 swap label，不足以训练完整 updater。
3. 对 PickXtimes 增加 press/incomplete/no-event 平衡采样；不重复已经完成的旧 cumulative-count 实验。
4. 弱项修好后再跑 300--400 step 多任务 LoRA。进入 recurrent MEM teacher cache 的最低建议门槛为 event F1 >= 85%、valid >= 95%，state rollout final accuracy 另行统计。

## 产物

- 训练 adapter：`checkpoints/qwen3vl_robomme_four_task_local_event_corrected_pilot_260826/final`
- 训练日志：`checkpoints/qwen3vl_robomme_four_task_local_event_corrected_pilot_260826/metrics.jsonl`
- 平衡生成记录：`artifacts/robomme_four_task_qwen_pilot_eval_seed260826/corrected_pilot100.jsonl`
- 汇总：`artifacts/robomme_four_task_qwen_pilot_eval_seed260826/corrected_pilot100.summary.json`
- 四任务混合清单：`artifacts/robomme_four_task_qwen_mixture_seed260826`
