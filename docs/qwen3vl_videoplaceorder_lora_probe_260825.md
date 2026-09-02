# Qwen3-VL VideoPlaceOrder LoRA 可学习性实验（2026-08-25）

## 结论

Qwen3-VL-4B 可以从 VideoPlaceOrder 演示中学到一部分“第 N 次放置目标”的顺序记忆与空间定位能力，但目前仍不足以单独为 action 提供可靠目标。

在 20 条完全独立的 held-out episode 上，使用完全相同的 variant-0 输入比较：

- 8x8 目标格 exact：zero-shot 5% -> LoRA 25%
- 允许一个网格的误差：zero-shot 20% -> LoRA 55%
- 截断演示拒答：0% -> 100%
- 只有局部放置片段时拒答：0% -> 95%

step 400 在三种演示采样变体上的最终结果为 28.33% exact、60% within-one-cell。该提升说明模型学到了任务信息，但精度仍低于闭环动作所需水平。

困难档包含 target swap。困难档三变体 exact 为 38.89%、within-one-cell 为 61.11%。在验证集中真正跨 8x8 cell 移动的 4 条 episode（12 个采样变体）上，exact 为 33.33%；错误预测中有明显回到交换前位置的倾向，说明 swap 后的目标身份跟踪是主要瓶颈。

## 数据协议

- 原始数据：`data/robomme_extracted/record_dataset_VideoPlaceOrder.h5`
- 有效数据：100 episode
- 切分：80 train / 20 validation，按 difficulty、颜色、ordinal 和演示目标数分层
- 训练样本：640
  - 480 条 full-demo
  - 80 条 truncated-demo
  - 80 条 local-only
- 验证样本：100
  - 60 条 full-demo（20 episode x 3 采样变体）
  - 20 条 truncated-demo
  - 20 条 local-only

输入严格只使用 `is_video_demo=True` 的演示帧；所有帧索引都早于机器人执行段。执行段仅用于离线生成监督标签，没有作为模型输入。

### 12 帧压缩方式

演示长度约为 900–1100 帧，而 Qwen 输入固定为 12 帧：

1. 每次演示放置选择 2 个接近完成时刻的帧，用于保持放置顺序。
2. 剩余帧在演示末尾的 static/swap 段均匀采样，用于观察目标交换后的最终位置。
3. 若演示中有 4 次放置，则 8 帧用于放置事件，只剩 4 帧表示 target swap。

监督输出为：

```json
{"event":"ordinal_target_grounded","target_color":"green","ordinal":3,"target_cell":"r3_c4"}
```

负样本在证据不足时输出：

```json
{"event":"insufficient_evidence"}
```

## 数据审计

- 100 条 episode 均有独立演示前缀和执行段。
- easy/medium 中，演示目标与最终目标坐标平均仅相差 1.24/0.42 像素。
- hard 中平均相差 37.39 像素、最大 90.14 像素，证明 hard split 确实要求跟踪 target swap，而不能只复述初始放置位置。
- 目标监督采用 camera-relative 8x8 网格，每格 32x32 像素。

## 训练设置

- 基座：`Qwen3-VL-4B-Instruct`
- 初始 adapter：VideoUnmask + 25% ShellGame replay 的 B/checkpoint-000300
- LoRA rank 16，alpha 32，dropout 0.05
- batch size 4
- learning rate 2e-5，40-step warmup，cosine decay
- 400 optimizer steps
- 训练环境：`/data1/conda_envs/qwen3vl_shellgame`

teacher-forcing validation：

| Step | Val loss | Token accuracy |
|---:|---:|---:|
| 100 | 0.10919 | 95.64% |
| 200 | 0.08897 | 96.71% |
| 300 | 0.07993 | 97.35% |
| 400 | 0.07834 | 97.47% |

## 自回归生成结果

### Variant-0 checkpoint 对比

| Checkpoint | Full exact | Within 1 cell | Truncated reject | Local-only reject |
|---|---:|---:|---:|---:|
| B zero-shot | 5% | 20% | 0% | 0% |
| Step 100 | 5% | 35% | 60% | 80% |
| Step 200 | 25% | 55% | 60% | 80% |
| Step 300 | 20% | 55% | 100% | 90% |
| Step 400 | 25% | 55% | 100% | 95% |

step 400 的 full-demo exact 与 step 200 相同，但拒答校准和 teacher-forcing validation 更好，因此选 step 400 作为本次任务 probe。

### Step 400 全部三种采样变体

| Split | Exact | Within 1 cell |
|---|---:|---:|
| Overall full-demo | 28.33% | 60.00% |
| Easy | 18.18% | 54.55% |
| Medium | 44.44% | 77.78% |
| Hard / target swap | 38.89% | 61.11% |

validation 中 ordinal 3 和 4 各只有 1 条独立 episode，因此其分项百分比不能用于得出可靠结论。

## 对 MEM 设计的含义

当前单次 12 帧压缩同时承担两个困难目标：

1. 记住哪一个 target 是第 N 次放置目标；
2. 在后续 swap 中持续跟踪该 target 的身份和最终位置。

更合适的方案是两阶段事件记忆：

1. 每次 `place_complete` 触发 Qwen，将目标视觉 token 按 ordinal 写入 MEM。
2. swap 段使用滑动窗口持续更新目标 token，而不是用 4 个稀疏帧一次性表示整个交换。
3. 最终由 action 查询第 N 个 memory slot，并在当前画面中对候选 target 做匹配。

这样 Qwen 负责局部语义与目标身份，MEM 负责顺序和跨事件跟踪；也避免让语言模型直接生成精确坐标。下一轮训练目标建议改为“候选 target 分类 + memory token 对比损失”，而不是只有 8x8 cell 文本生成。

另外，本轮为单任务可学习性 probe，没有旧任务 replay；不应覆盖原来的通用 B checkpoint。正式多任务训练仍应加入 VideoUnmask 和 ShellGame replay。

## 产物

- 数据摘要：`artifacts/videoplaceorder_qwen3vl_sft_seed260825/summary.json`
- 推荐任务 probe：`checkpoints/qwen3vl_videoplaceorder_from_multitask_B_260825/checkpoint-000400`
- zero-shot：`evaluation/robomme/qwen3vl_videoplaceorder_B_zeroshot_val20.summary.json`
- step 400 全变体：`evaluation/robomme/qwen3vl_videoplaceorder_step400_val20_all_variants.summary.json`

