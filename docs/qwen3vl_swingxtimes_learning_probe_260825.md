# Qwen3-VL SwingXtimes 可学习性实验（2026-08-25）

## 结论

Qwen3-VL-4B 经过 400 step LoRA 后，能够从 RoboMME SwingXtimes 的稀疏因果视频前缀中学习往返进度：20 个独立 held-out episode、全部 3 种采样变体上，累计计数准确率从 zero-shot 的 0% 提升到 73.86%，最终完成状态准确率为 70%。这说明该视觉语义能力是可学习的。

但它还不适合直接决定动作或停止：左右到达事件准确率为 60.98%，严格要求整条 episode 所有进度状态都正确时只有 10%。此外，本轮 SwingXtimes-only 微调造成了明显的旧任务遗忘。因此该 checkpoint 应视为任务可学习性 probe，不应替代原来的多任务 B checkpoint。

下一版更合理的结构是：Qwen 只识别局部语义事件（`right_arrival` / `left_arrival` / `no_event`），由 recurrent MEM 或显式有限状态机累积计数；同时加入 ShellGame 和 VideoUnmask replay。

## 任务与数据

- 原始数据：`data/robomme_extracted/record_dataset_SwingXtimes.h5`
- 有效 episode：100 条，按目标往返次数 1/2/3 分布为 23/31/46 条
- episode 长度：296–608 帧，均值 434.19 帧
- 任务过程：抓取指定颜色方块，依次到达右端和左端；一次 `right -> left` 构成一轮，目标为 1–3 轮；随后放下方块并按停止按钮
- 所有 100 条 episode 的 `simple_subgoal` 转移顺序均与上述状态机一致
- 数据没有独立的 `is_video_demo=True` 演示段；本实验输入均是当前机器人执行轨迹的因果前缀，不使用未来帧

### 切分

固定随机种子 260825，按目标轮数分层切分：

- train：80 episode，1512 样本
- validation：20 个完全独立 episode，372 样本

validation 中 372 个样本包含同一进度点的多个采样变体，因此样本级指标用于测试采样鲁棒性；episode 级指标只在 20 条独立轨迹上计算。

### 视频输入

每条输入最多 12 帧：

- 每个已经完成的端点到达事件保留 2 个边界附近帧
- 目标最多 3 轮，即最多 6 次端点到达，正好占 12 帧
- 剩余位置由事件发生前的历史帧补齐
- 所有帧时间戳都不晚于当前监督事件，避免 future leakage

### 监督协议

因果前缀输出紧凑 JSON：

```json
{"event":"right_arrival","right_count":2,"left_count":1,"completed_round_trips":1,"ready_to_stop":false}
```

并加入两类负样本：

- 端点事件尚未完成：`{"event":"no_completed_arrival"}`
- 只给局部到达片段、缺少累计历史：`{"event":"insufficient_history"}`

训练样本组成：1074 个 causal prefix、278 个 local-only、160 个 no-event。

## 训练设置

- 基座：`Qwen3-VL-4B-Instruct`
- 初始 adapter：VideoUnmask + 25% ShellGame replay 的 B/checkpoint-000300
- LoRA：rank 16，alpha 32，dropout 0.05
- batch size：4
- 学习率：2e-5，40 step warmup，cosine decay
- 优化步数：400
- 环境：`/data1/conda_envs/qwen3vl_shellgame`

teacher-forcing validation：

| Step | Val loss | Token accuracy |
|---:|---:|---:|
| 100 | 0.06776 | 97.17% |
| 200 | 0.04041 | 98.13% |
| 300 | 0.02592 | 98.56% |
| 400 | 0.02620 | 98.47% |

由于真实自回归生成指标在 step 400 略好，本实验选择 step 400，而不是只按 teacher-forcing loss 选择 step 300。

## SwingXtimes 生成结果

### Zero-shot 与微调后

| 指标 | B zero-shot | Swing step 400 |
|---|---:|---:|
| Causal-prefix 累计计数准确率 | 0.00% | 73.86% |
| 当前左右事件准确率 | 0.00% | 60.98% |
| Causal-prefix 完整 JSON exact | 0.00% | 60.23% |
| Local-only 拒答准确率 | 0.00% | 97.06% |
| No-event 准确率 | 5.00% | 90.00% |
| 最终 count + ready_to_stop | 0.00% | 70.00% |
| 整条进度序列全部正确 | 0.00% | 10.00% |

step 400 结果使用全部 3 种采样变体；累计计数按目标轮数拆分：

- 1 轮：80.00%
- 2 轮：70.83%
- 3 轮：74.07%

从 0% 到约 74% 的提升不能由输出格式记忆解释，因为验证 episode 与训练 episode 完全分离，而且模型还必须区分局部无历史、未完成到达和不同累计阶段。不过严格序列只有 10%，表明单个状态的错误仍会在长序列中累积。

## 旧任务保持性

本轮没有加入旧任务 replay，遗忘结果符合预期，但幅度较大：

### ShellGame

- reveal / no-event / incomplete-event：均为 20/20
- swap：0/60，全部退化为 `incomplete_event`
- 完整 reveal + 三次 swap：0/20

而 Swing 微调前的 B checkpoint 在相同验证协议上为 120/120，因此 step 400 不能作为通用 checkpoint。

### VideoUnmask

- paired-memory exact：61.67%，nearest-container：91.67%
- visible-grounding exact：38.33%，nearest-container：51.67%
- masked-only 拒答：100%

Swing 微调前 B 的 paired exact 为 80%、nearest 为 98.3%，visible exact 为 81.7%、nearest 为 100%。尤其 visible grounding 发生了严重退化。

## 建议的下一版

1. 将 Qwen 目标收缩为局部 event detector，不要求它从压缩视频重复推导全局累计数。
2. recurrent MEM 保存 `(last_side, right_count, left_count, completed_round_trips)`，只在可靠事件触发时更新；状态转移约束为 `right -> left -> right`。
3. 训练混合建议从 70% SwingXtimes、15% VideoUnmask、15% ShellGame 开始，并按各任务 validation 独立早停。
4. 对 Swing 增加 endpoint 附近 hard negatives、连续帧去重和 event hysteresis，优先把局部事件准确率从 61% 提升到 85% 以上。
5. 达到局部事件稳定后再接入 action；当前 checkpoint 适合验证 perception/memory contract，不适合直接闭环控制。

## 产物

- 数据构建摘要：`artifacts/swingxtimes_qwen3vl_sft_seed260825/summary.json`
- 训练配置与曲线：`checkpoints/qwen3vl_swingxtimes_from_multitask_B_260825/training_config.json`、`metrics.jsonl`
- 推荐任务 probe：`checkpoints/qwen3vl_swingxtimes_from_multitask_B_260825/checkpoint-000400`
- zero-shot 结果：`evaluation/robomme/qwen3vl_swingxtimes_B_zeroshot_val20.summary.json`
- Swing 全变体结果：`evaluation/robomme/qwen3vl_swingxtimes_step400_val20_all_variants.summary.json`
- ShellGame 保持性：`evaluation/shellgame/qwen3vl_swingxtimes_step400_shellgame_val20.summary.json`
- VideoUnmask 保持性：`evaluation/robomme/qwen3vl_swingxtimes_step400_videounmask_val20.summary.json`

