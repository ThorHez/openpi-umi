# 真实杯子交换数据微调 Qwen3-VL：实验日志与跨会话交接

日期：2026-08-26  
工程：`/data2/hzl_workspace_for_pi_mem/openpi-umi`  
状态：真实数据分支的三组实验已完成；下一位实验者可以直接从本文列出的 checkpoint 和 manifest 继续。

## 1. 目标与当前结论

目标是让已有的 ShellGame Qwen3-VL LoRA 适应真实杯子交换视频，读取三个中间交换事件，并为后续 recurrent MEM 生成结构化事件标签。

目前得到的核心结论：

1. 仿真 ShellGame LoRA 不能零样本迁移到真实杯子视频：469 个真实滑窗中预测出 0 个 swap。
2. 使用真实 GT 中心的 12 帧单事件窗口微调后，验证集交换对准确率从 35.0% 提升到 80.0%，三次交换全部正确从 5.0% 提升到 55.0%。
3. 使用不切事件的 36 帧完整观察上下文微调后，初始杯位置能达到 95%，但交换平均准确率最高只有 51.7%，三次交换全部正确只有 15%。
4. 因此当前最可靠的模型是“局部事件读取器”，而不是完整视频的一次性长时序推理器。
5. 当前 12 帧局部评测使用 GT 给出的事件中心/边界，并没有解决真实滑动窗口中的事件发现、去重和 `no_event` / `incomplete_event` 判断。

推荐用于后续局部事件实验的 checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/qwen3vl_real_cup_gt_sequence_balanced_lora_v2_260826/checkpoint-000060
```

推荐用于“完整上下文 -> 渐进裁剪”课程学习起点的 checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826/checkpoint-000150
```

## 2. 环境与关键路径

Qwen 专用环境：

```text
/data1/conda_envs/qwen3vl_shellgame/bin/python
/data1/conda_envs/qwen3vl_shellgame/bin/accelerate
```

基础模型：

```text
/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct
```

起始 ShellGame LoRA：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375
```

真实数据：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/data/cup_replay_buffer.zip
/data2/hzl_workspace_for_pi_mem/openpi-umi/data/cup_replay_buffer/replay_buffer.zarr
/data2/hzl_workspace_for_pi_mem/openpi-umi/data/cup_replay_buffer/labels.jsonl
```

注意：Qwen 环境中没有用于预处理的 `zarr/numcodecs` 依赖。数据构建应使用工程 Python 环境，Qwen 训练和生成评测才使用上述 Qwen 环境。

## 3. 数据审计与标签定义

数据共有 100 个 episode、49,326 帧 RGB，原始视频约为 20 Hz。RGB 字段为 `camera0_rgb`，分辨率为 `224 x 224`。

`labels.jsonl` 每条包含：

- `episode_id`
- `initial_cup`
- `moves`：三次交换的杯位置对
- `final_cup`
- `frames_per_move=80`
- `n_observe_frames=241`
- `grasp_start_frame=241`
- `n_frames`

标签位置索引使用 `0/1/2`，训练时统一转换为相机坐标：

```text
0 -> screen_left_cup
1 -> screen_middle_cup
2 -> screen_right_cup
```

由于标签时间轴与 Zarr 原始帧数不同，使用以下线性映射：

```text
raw_index = round(label_index * (raw_length - 1) / (label_n_frames - 1))
```

已完成一致性检查：

- 100 条标签与 100 个 Zarr episode 一一对应。
- 从 `initial_cup` 顺序执行三个 `moves` 后，100/100 都等于 `final_cup`。
- 红球颜色检测器可覆盖 100/100 个初始画面，与人工标签一致 82/100；该检测器只用于审计，从未覆盖人工 GT。

## 4. 固定训练/验证划分

所有真实数据实验使用相同的 episode-disjoint 划分：80 train / 20 validation。

固定验证 episode：

```text
[1, 10, 16, 24, 26, 27, 31, 32, 34, 38, 40, 45, 50, 56, 58, 60, 73, 82, 84, 93]
```

后续实验不要重新随机划分，否则不能与现有结果直接对比。

## 5. 输出契约

统一实现：

```text
src/openpi/tasks/shellgame/real_cup_qwen3vl_sft_contract.py
```

局部交换只输出一个交换对：

```json
{"screen_pair":["screen_left_cup","screen_middle_cup"]}
```

初始或最终位置分别只输出一个事实：

```json
{"initial_cup":"screen_middle_cup"}
{"final_cup":"screen_left_cup"}
```

完整 sequence 旧合同输出：

```json
{"initial_cup":"screen_middle_cup","moves":[["screen_left_cup","screen_middle_cup"],["screen_left_cup","screen_right_cup"],["screen_middle_cup","screen_right_cup"]],"final_cup":"screen_left_cup"}
```

杯对必须按照画面从左到右排序。

## 6. 实验 0：真实域零样本滑窗探测

模型使用原始 ShellGame checkpoint-375。主设置是 20 输入帧、原视频 stride 4、窗口起点 stride 10，在 10 个真实 episode 上共生成 469 个滑窗。

结果：

| 输出 | 数量 |
|---|---:|
| `swap` | 0 |
| `incomplete_event` | 363 |
| `no_event` | 106 |
| 合法 JSON | 469/469 |

结论：模型学会了 JSON/拒答格式，但没有迁移真实杯子交换语义。不能用这批输出作为 recurrent MEM 的伪标签。

详细记录：

```text
docs/real_cup_qwen3vl_pseudo_annotation_probe_260826.md
evaluation/shellgame/real_cup_qwen3vl_step375_probe/confirm10_frames20_stride4.summary.json
```

## 7. 实验 1：12 帧局部事件 + 12 帧完整 sequence

### 7.1 样本构造

每个 episode 生成：

- 3 个 `local_swap`：每个样本为均匀覆盖一个 GT 交换区间的 12 帧，目标只包含该交换对。
- 1 个 `sequence`：12 帧均匀覆盖完整观察阶段，目标包含初始位置、三个交换对和最终位置。

原始 manifest：

| split | local swap | sequence | total |
|---|---:|---:|---:|
| train | 240 | 80 | 320 |
| val | 60 | 20 | 80 |

平衡版 `train_balanced.jsonl` 将 sequence 重复三次，得到 240 local + 240 sequence，共 480 个训练样本。

数据文件：

```text
artifacts/real_cup_qwen3vl_gt_sft_v1_260826/train.jsonl
artifacts/real_cup_qwen3vl_gt_sft_v1_260826/train_balanced.jsonl
artifacts/real_cup_qwen3vl_gt_sft_v1_260826/val.jsonl
artifacts/real_cup_qwen3vl_gt_sft_v1_260826/summary.json
```

### 7.2 Stage 1 训练

- 初始化：ShellGame checkpoint-375
- 6 张 A100
- 每卡 batch 2，梯度累积 1，全局 batch 12
- 120 optimizer steps
- LR `1e-5`，cosine，warmup 10
- bf16，LoRA rank 16 / alpha 32 / dropout 0.05
- 每 30 步保存和验证

输出：

```text
checkpoints/qwen3vl_real_cup_gt_sequence_lora_v1_260826/
```

### 7.3 Stage 2 平衡继续训练

- 初始化：Stage 1 checkpoint-000120
- 使用 `train_balanced.jsonl`
- 6 张 A100，全局 batch 12
- 90 optimizer steps
- LR `5e-6`，cosine，warmup 5
- 每 30 步保存和验证

输出：

```text
checkpoints/qwen3vl_real_cup_gt_sequence_balanced_lora_v2_260826/
```

### 7.4 Held-out 生成结果

| 模型 | JSON 合法 | 局部交换对 | 三次局部交换全对 | 局部递推 final | 完整 clip initial | 完整 clip final | 完整 JSON 全对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 checkpoint-375 | 77.5% | 35.0% | 5.0% | 65.0% | 10.0% | 5.0% | 0.0% |
| Stage 1 step 90 | 100% | 71.7% | 35.0% | 45.0% | 55.0% | 50.0% | 0.0% |
| Stage 1 step 120 | 98.8% | 71.7% | 35.0% | 45.0% | 60.0% | **65.0%** | 0.0% |
| Balanced step 30 | 100% | 78.3% | **55.0%** | 60.0% | 80.0% | 50.0% | 0.0% |
| Balanced step 60 | 100% | **80.0%** | **55.0%** | 60.0% | **85.0%** | 30.0% | 0.0% |
| Balanced step 90 | 100% | **80.0%** | **55.0%** | 60.0% | 80.0% | 45.0% | 0.0% |

主结果计数：局部交换 48/60，三次交换全对 11/20。

### 7.5 如何正确解释结果

- 局部样本彼此独立。第三次交换输入只有第三次交换的 12 帧，既没有前两次 GT，也没有外部 memory。
- “局部递推 final”评测从 GT `initial_cup` 开始，把 Qwen 对三个局部窗口的预测依次送入符号 updater。
- endpoint 指标会被错误抵消污染。原始 checkpoint 虽然只有 1/20 三次交换全对，却有 13/20 的递推 final 正确，所以不能用 final accuracy 代替事件路径准确率。
- 完整 sequence SFT 使用自回归 teacher forcing。第三个 move 的 loss 会看到前两个 GT target token，训练与一次性自由生成存在差距，因此完整 clip final=65% 只能作为辅助结果。

## 8. 实验 2：36 帧完整上下文优先

### 8.1 为什么改合同

为了测试“先学完整过程，再切成局部窗口”，避免旧 `sequence` 合同中的 target-side teacher forcing，同一个 36 帧完整视频被拆成五个彼此独立的问题：

1. initial cup
2. first swap
3. second swap
4. third swap
5. final cup

每个答案只包含当前事实。第三次交换问题的 target 中不含前两次交换，final 问题的 target 中不含任何 GT move。

### 8.2 输入和训练

- 每个 episode 从 reveal 到 decision 均匀采 36 帧。
- 不输入事件边界。
- 排除 decision 之后的 grasp/action 后缀。
- train 400 条：80 initial + 240 swap + 80 final。
- val 100 条：20 initial + 60 swap + 20 final。
- 从 ShellGame checkpoint-375 初始化，不从局部真实数据 checkpoint 初始化。
- 6 张 A100，每卡 batch 1，全局 batch 6。
- 200 optimizer steps，LR `1e-5`，warmup 15。
- 每 50 步保存与验证。

输出：

```text
checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826/
```

### 8.3 Held-out 生成结果

| checkpoint | Initial | Swap 平均 | Swap 1 | Swap 2 | Swap 3 | 三次全对 | Direct final | Recurrent final | 五事实全对 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 step-375 | 50% | 35.0% | 40% | 30% | 35% | 5% | **35%** | 40% | 0% |
| Full step 50 | 65% | 40.0% | 30% | 55% | 35% | 5% | 30% | **45%** | 0% |
| Full step 100 | **95%** | 45.0% | 45% | 50% | 40% | 5% | 30% | 40% | 0% |
| Full step 150 | 90% | **51.7%** | **65%** | **50%** | **40%** | **15%** | 25% | **45%** | **5%** |
| Full step 200 | 90% | 50.0% | **65%** | 45% | **40%** | 10% | 30% | 40% | **5%** |

所有 checkpoint 的 JSON 合法率均为 100%。

最优 step 150 对第三次交换的预测有明显塌缩：20 条中 16 条预测为左右杯交换。标签本身大致平衡，因此更像长时序 ordinal/event binding 失败，而不是类别不均衡。

## 9. 当前 checkpoint 选择

不同 checkpoint 的用途不同，不要只按 final accuracy 选择：

| 用途 | checkpoint | 原因 |
|---|---|---|
| 局部真实交换读取器 | `balanced_lora_v2/checkpoint-000060` | 局部 pair 80%，三次全对 55%，JSON 100% |
| 直接预测完整 clip 的 final | `gt_sequence_lora_v1/checkpoint-000120` | direct final 65%；但不适合作为事件 teacher |
| 完整上下文课程学习起点 | `full_context36_lora_v1/checkpoint-000150` | 完整上下文 swap 平均和三次全对最好 |

## 10. 建议的下一组实验

### 10.1 首选：完整上下文到局部窗口的渐进裁剪课程

验证完整上下文预训练是否真的能帮助局部事件学习，而不是只学习初始场景外观：

1. 初始化 `full_context36 ... checkpoint-000150`。
2. 先训练覆盖单个事件的 24 帧 clip。
3. 再训练现有 12 帧单事件 clip。
4. 两阶段都保留 20% 的 36 帧 full-context replay，防止完整场景能力快速遗忘。
5. 使用完全相同的 20 个验证 episode。
6. 与“直接从 ShellGame checkpoint-375 做局部训练”的现有结果公平比较。

判定标准：必须超过当前 48/60 的局部 pair 和 11/20 的三次交换全对，才能证明 full-context-first 有实际价值。

### 10.2 第二步：真实滑窗事件发现

当前局部结果依赖 GT event boundary。要生成可用于 recurrent MEM 的自动 JSON，还需要：

- 从完整轨迹采真实 `no_event` 负样本；
- 从交换起始/结束附近采 `incomplete_event` 负样本；
- 对正样本做边界抖动，避免只识别固定中心裁剪；
- 训练或校准 event confidence；
- 在滑窗推理中做相邻窗口去重/NMS；
- 分开报告 event precision/recall/F1、matched-event pair accuracy、三事件路径准确率。

在事件 precision/recall 合格之前，不要把自动输出当作 MEM 的训练 GT。

## 11. 等价复现命令

以下命令根据已保存的 `training_config.json` 整理。现有 artifact/checkpoint 已存在；若重跑，建议换一个新的输出目录，避免覆盖结果。

先进入工程：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
```

构建 12 帧局部/sequence manifest（预处理使用工程环境）：

```bash
/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python \
  scripts/mem/build_real_cup_qwen3vl_sft_manifest.py \
  --output-dir artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO

/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python \
  scripts/mem/balance_real_cup_qwen3vl_manifest.py \
  --input artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/train.jsonl \
  --output artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/train_balanced.jsonl \
  --sequence-repeat 3
```

Stage 1 训练：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
/data1/conda_envs/qwen3vl_shellgame/bin/accelerate launch \
  --multi_gpu --num_processes 6 \
  scripts/mem/train_shellgame_qwen3vl_lora.py \
  --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct \
  --initial-adapter checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375 \
  --train-manifest artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/train.jsonl \
  --val-manifest artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/val.jsonl \
  --output-dir checkpoints/qwen3vl_real_cup_gt_sequence_lora_v1_260826_REPRO \
  --max-steps 120 --per-device-batch-size 2 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --warmup-steps 10 --num-workers 1 \
  --logging-steps 5 --eval-steps 30 --eval-batches 7 --save-steps 30
```

Stage 2 平衡训练：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
/data1/conda_envs/qwen3vl_shellgame/bin/accelerate launch \
  --multi_gpu --num_processes 6 \
  scripts/mem/train_shellgame_qwen3vl_lora.py \
  --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct \
  --initial-adapter checkpoints/qwen3vl_real_cup_gt_sequence_lora_v1_260826_REPRO/checkpoint-000120 \
  --train-manifest artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/train_balanced.jsonl \
  --val-manifest artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/val.jsonl \
  --output-dir checkpoints/qwen3vl_real_cup_gt_sequence_balanced_lora_v2_260826_REPRO \
  --max-steps 90 --per-device-batch-size 2 --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 --warmup-steps 5 --num-workers 1 \
  --logging-steps 5 --eval-steps 30 --eval-batches 7 --save-steps 30
```

构建 36 帧完整上下文 manifest：

```bash
/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python \
  scripts/mem/build_real_cup_qwen3vl_full_context_manifest.py \
  --split-summary artifacts/real_cup_qwen3vl_gt_sft_v1_260826_REPRO/summary.json \
  --output-dir artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826_REPRO
```

36 帧完整上下文训练：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
/data1/conda_envs/qwen3vl_shellgame/bin/accelerate launch \
  --multi_gpu --num_processes 6 \
  scripts/mem/train_shellgame_qwen3vl_lora.py \
  --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct \
  --initial-adapter checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375 \
  --train-manifest artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826_REPRO/train.jsonl \
  --val-manifest artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826_REPRO/val.jsonl \
  --output-dir checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826_REPRO \
  --max-steps 200 --per-device-batch-size 1 --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 --warmup-steps 15 --num-workers 1 \
  --logging-steps 10 --eval-steps 50 --eval-batches 17 --save-steps 50
```

局部模型生成评测示例：

```bash
/data1/conda_envs/qwen3vl_shellgame/bin/python \
  scripts/mem/eval_real_cup_qwen3vl_lora.py \
  --adapter-path checkpoints/qwen3vl_real_cup_gt_sequence_balanced_lora_v2_260826/checkpoint-000060 \
  --output evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/balanced_step60_recheck.jsonl \
  --device cuda:1
```

完整上下文生成评测示例：

```bash
/data1/conda_envs/qwen3vl_shellgame/bin/python \
  scripts/mem/eval_real_cup_qwen3vl_full_context.py \
  --adapter-path checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826/checkpoint-000150 \
  --output evaluation/shellgame/real_cup_qwen3vl_full_context_v1/step150_recheck.jsonl \
  --device cuda:1
```

## 12. 代码与结果索引

数据/合同/训练：

```text
scripts/mem/export_cup_replay_buffer_episodes.py
scripts/mem/build_real_cup_qwen3vl_sft_manifest.py
scripts/mem/balance_real_cup_qwen3vl_manifest.py
scripts/mem/build_real_cup_qwen3vl_full_context_manifest.py
scripts/mem/train_shellgame_qwen3vl_lora.py
src/openpi/tasks/shellgame/real_cup_qwen3vl_sft_contract.py
```

评测：

```text
scripts/mem/eval_real_cup_qwen3vl_events.py
scripts/mem/eval_real_cup_qwen3vl_lora.py
scripts/mem/eval_real_cup_qwen3vl_full_context.py
```

主结果：

```text
evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/balanced_step60.summary.json
evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/step120.summary.json
evaluation/shellgame/real_cup_qwen3vl_full_context_v1/step150.summary.json
```

原始详细报告：

```text
docs/real_cup_qwen3vl_pseudo_annotation_probe_260826.md
docs/real_cup_qwen3vl_gt_finetune_results_260826.md
docs/real_cup_qwen3vl_full_context36_results_260826.md
```

## 13. 新会话可以直接使用的任务描述

```text
请阅读 docs/real_cup_qwen3vl_finetune_handoff_260826.md，并在不改变固定验证 episode 的前提下，继续真实杯子数据上的“36 帧完整上下文 -> 24 帧事件窗口 -> 12 帧事件窗口”课程学习实验。以 full_context36 checkpoint-000150 初始化，保留 20% 完整上下文 replay，并与现有 local-first balanced checkpoint-000060 的 80.0% 局部 pair / 55.0% 三事件全对结果比较。不要用 final endpoint accuracy 代替事件路径准确率。
```
