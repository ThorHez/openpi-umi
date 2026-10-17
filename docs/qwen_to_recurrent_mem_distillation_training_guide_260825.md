# Qwen3-VL 蒸馏 Recurrent MEM 训练手册

> 更新日期：2026-08-25
> 适用范围：ShellGame 已验证 recipe，以及把该方法迁移到其他长时序任务
> 已验证结果：学生 MEM 在 5,000 个 episode 上的最终杯位准确率为 **94.6%**

## 1. 目标与关键结论

本 recipe 用 Qwen3-VL 提供高层视觉事件知识，把它蒸馏到一个低计算量、可递归更新的视觉 MEM。蒸馏完成后：

- 学生 MEM 输入是一小段连续图像和上一步 memory embedding；
- 学生 MEM 输出是下一步固定容量的 memory embedding；
- 部署时不需要 Qwen、relation label、最终目标 label 或仿真 metadata；
- action model 直接读取学生 memory，而不是读取 Qwen 文本。

ShellGame 中的目标是：看到小球初始所在杯子，跟踪三次换位，并在遮挡后记住最终杯位。

### 1.1 当前实现不是 Qwen hidden-state 蒸馏

已验证方案的 teacher 路径是：

```text
短视频 clip
  -> Qwen3-VL LoRA
  -> 结构化 reveal / exchange 事件
  -> 已验证 symbolic recurrent teacher
  -> 每个事件后的 teacher memory [128,64]
  -> 监督 direct-visual recurrent MEM
```

蒸馏 target 是 symbolic teacher 的 recurrent memory，不是 Qwen token hidden state。Qwen 负责把视频解析为结构化事件；symbolic teacher 把离散事件转换为连续 memory state。

### 1.2 teacher relation 不能泄漏到学生

训练参数 `teacher_relation_ids` 只用于构造 teacher target，不能进入 `direct_visual_segment_encoder`、`shared_visual_memory_updater` 或 action 读取路径。学生部署接口必须保持为：

```text
(previous_memory, continuous_images) -> next_memory
```

当前 ShellGame JAX 实验使用仿真 GT 作为已验证 Qwen reveal/relation 输出的 clean cached proxy，并没有逐 episode 在线运行 Qwen。合作实验报告必须保留这个边界。

## 2. 模型结构

### 2.1 Qwen 教师

```text
10 帧连续视频，224x224
  -> Qwen3-VL-4B-Instruct + LoRA
  -> 紧凑 JSON
       reveal:           {"screen_cup":"screen_left_cup"}
       exchange:         {"screen_pair":["screen_left_cup","screen_middle_cup"]}
       no event:         {"event":"no_event"}
       incomplete event: {"event":"incomplete_event"}
  -> camera/world adapter
  -> 初始容器 ID 或交换对 ID
```

Qwen 只预测摄像机相对事实，不直接预测最终杯位，也不输出机器人动作。

Qwen 微调采用 LoRA，目标模块覆盖视觉侧 `qkv/proj/linear_fc1/linear_fc2` 和轻量语言侧 `q_proj/k_proj/v_proj/o_proj`。训练使用 bf16、gradient checkpointing，并且只在 assistant 的紧凑 JSON token 上计算 loss。当前 4.46B 模型中约 19.08M 参数可训练（约 0.43%）；6 卡、每卡 batch 2、梯度累积 4 时 global batch 为 48。

### 2.2 Symbolic recurrent teacher

```text
Qwen reveal -> initial teacher memory [128,64]
Qwen event 1 + previous teacher memory -> teacher memory 1 [128,64]
Qwen event 2 + previous teacher memory -> teacher memory 2 [128,64]
Qwen event 3 + previous teacher memory -> teacher memory 3 [128,64]
```

已验证 teacher checkpoint：

```text
checkpoints/shellgame_stage_slot_only_relation_recurrent_probe/
  stage_slot_only_random_relation_frozen_memory_1k_260821/500/params
```

teacher 在蒸馏期间完全冻结，teacher memory 使用 `stop_gradient`。

### 2.3 Direct-visual recurrent MEM 学生

```text
60 帧历史 base_rgb
  -> SigLIP 浅层 patch embedding
  -> 每帧 16x16 = 256 patches，width=1152
  -> 固定 2x2 pooling
  -> 每帧 8x8 = 64 tokens

三个 10 帧 clip
  -> 共享 DirectVisualSegmentEncoder
       width=256, depth=2, heads=8
       factorized temporal/spatial attention
  -> 每个 clip 得到 [10*64,64] = [640,64] visual evidence

(previous memory [128,64], visual evidence [640,64])
  -> 共享 RecurrentMemoryUpdater
       width=64, depth=2, heads=4
  -> next memory [128,64]
```

当前 ShellGame 固定切片：

```python
HISTORY_FRAMES = 60
SWAP_SLICES = ((20, 30), (30, 40), (40, 50))
SWAP_SEGMENT_SIZE = 10
SPATIAL_TOKENS = 64
```

`SingleHistoryReadAdapter + SharedMemoryTokenReadout` 只用于监督和评估 memory 是否可读，不是学生 updater 的 relation 输入。

### 2.4 训练与推理数据流

```text
训练：
metadata/Qwen event -> frozen symbolic teacher -> teacher memory_t
                                                    |
images_t -> visual compressor -> recurrent updater -> student memory_t
                                                    |
                                                    +-> diagnostic slot readout

推理：
initial detector -> initial memory
image clip_t + memory_(t-1) -> student memory_t -> action model
```

推理路径中没有 teacher relation、teacher memory 或最终杯位 GT。

## 3. 训练数据结构

### 3.1 原始数据

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/
  shellgame_absolute_eef_phase_instruction_dataset/
    episode_000000/
      metadata.json
      vla_trajectory.npz
    ...
```

`vla_trajectory.npz` 至少需要 `third_person_images` 和稳定帧索引。当前 `metadata.json` 需要 `initial_ball_cup`、`final_ball_cup`、`swaps` 和 `phase_ranges`。

```text
reveal  0..9
cover   10..19
swap_0  20..29
swap_1  30..39
swap_2  40..49
settle  50..59
```

### 3.2 Qwen SFT manifest

manifest 不复制图片，每行只保存轨迹路径、10 个帧索引和紧凑 target：

```json
{
  "schema_version": 1,
  "episode_index": 0,
  "sample_type": "swap",
  "event_index": 0,
  "trajectory_path": "/absolute/path/episode_000000/vla_trajectory.npz",
  "frame_indices": [20,21,22,23,24,25,26,27,28,29],
  "target": "{\"screen_pair\":[\"screen_left_cup\",\"screen_right_cup\"]}"
}
```

每个 episode 产生 6 条样本：

| 类型 | 数量 | 作用 |
|---|---:|---|
| `reveal` | 1 | 找到露出小球的杯子 |
| `swap` | 3 | 找到完成换位的两个杯子 |
| `no_event` | 1 | 拒绝静止窗口 |
| `incomplete_event` | 1 | 拒绝缺少完整 before/after 证据的窗口 |

当前 5,000 episode 数据使用 `split_seed=42`、`val_ratio=0.1`：train 为 4,500 episodes / 27,000 rows，validation 为 500 episodes / 3,000 rows。必须按 episode 分割，不能按 clip row 分割。

### 3.3 MEM 使用的 LeRobot 数据

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/
  shellgame_lerobot_absolute_eef_raw7/
```

memory loader 配置：

```text
num_frames = 61
frame_stride = 1
video_layout = fixed_prefix_current
fixed_prefix_frames = 60
min_frame_index = 59
max_frame_index = 59
tokenize_prompt = false
```

每个 episode 只取一条“60 帧固定历史 + 1 帧 current”样本。Standalone memory objective 忽略 action tensor。

### 3.4 ShellGame label table

```text
[initial_slot, relation_1, relation_2, relation_3,
 slot_after_1, slot_after_2, slot_after_3]

slot:     0=left, 1=middle, 2=right
relation: 0=(left,middle), 1=(left,right), 2=(middle,right)
```

`load_episode_label_table()` 会校验 raw 和 LeRobot metadata 的 initial/final cup 一致性。

### 3.5 推荐的通用 teacher cache

迁移到真机或其他任务时，推荐离线运行 Qwen 并缓存：

```text
episode_index:       [N]        int32
clip_frame_indices:  [N,S,W]    int32
event_valid:         [N,S]      bool
event_type:          [N,S]      string/int
event_arguments:     [N,S,...]  task-adapter output
teacher_memory:      [N,S,M,D]  float32/bfloat16
semantic_label:      [N,S,...]  optional
teacher_confidence:  [N,S]      float32
```

`teacher_memory` 是主 target；`semantic_label` 只是可选诊断监督。Qwen 输出非法、低置信度或事件不完整时设置 `event_valid=false`，不能用 GT 静默替换后仍声称是 Qwen pseudo-label。

## 4. Loss 与冻结策略

```text
L_memory = mean(1 - cosine(student_memory, stopgrad(teacher_memory)))
           + 0.1 * MSE(student_memory, stopgrad(teacher_memory))

L_stage = CE(Readout(student_memory_t), slot_after_t)

L_total = 1.0 * L_memory + 0.25 * L_stage
```

`L_stage` 用于确认 memory 包含可读任务状态，不是学生输入。迁移时可换成 entity state、subgoal progress、contrastive state 或 action 可读性 loss。

已验证冻结方式：

- 冻结 Qwen、symbolic teacher、recurrent updater、read adapter/readout 和 PaliGemma/SigLIP 主干；
- 只训练 `DirectVisualSegmentEncoder`；
- 可训练约 2.42M 参数，冻结约 3.51B 参数。

这样做让视觉压缩器映射到既有 recurrent memory state basis，避免同时改变 teacher 坐标系。

## 5. 完整训练流程

下面命令保留了已验证实验名，便于核对历史产物。启动新的合作实验时，必须把 Qwen 的 `--output-dir` 和 MEM 的 `--exp-name` 改成新的唯一名称；不要用 `--overwrite` 覆盖本手册列出的已验证 checkpoint。

### A. 准备 Qwen 独立环境

Qwen 不使用 OpenPI `.venv`。当前环境入口：

```text
/data1/conda_envs/qwen3vl_shellgame/bin/python
```

依赖文件为 `scripts/mem/qwen3vl_shellgame_requirements.txt`。如果环境不存在，请创建独立环境，不要把 Transformers/Qwen 依赖混入 OpenPI JAX 环境。

### B. 生成 Qwen SFT manifest

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
.venv/bin/python scripts/mem/build_shellgame_qwen3vl_sft_manifest.py --raw-root /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset --output-dir artifacts/shellgame_qwen3vl_gt_event_sft_v1 --split-seed 42 --val-ratio 0.1 --overwrite
```

检查 `summary.json` 中 train/val episode 不重叠、`future_frame_leakage=false`、`copied_video_bytes=0`，正负样本齐全。

### C. LoRA 微调 Qwen3-VL

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /data1/conda_envs/qwen3vl_shellgame/bin/accelerate launch --num_processes 6 scripts/mem/train_shellgame_qwen3vl_lora.py --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct --train-manifest artifacts/shellgame_qwen3vl_gt_event_sft_v1/train.jsonl --val-manifest artifacts/shellgame_qwen3vl_gt_event_sft_v1/val.jsonl --output-dir checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825 --max-steps 1800 --per-device-batch-size 2 --gradient-accumulation-steps 4 --learning-rate 2e-5 --warmup-steps 100 --lora-rank 16 --lora-alpha 32 --eval-steps 125 --eval-batches 8 --save-steps 375 --num-workers 2 --overwrite
```

当前后续蒸馏使用 `checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375`。该 checkpoint 在 held-out 数据上达到 100% reveal/swap 和完整三事件序列准确率。选 checkpoint 时应看结构化 event/sequence 指标，不能只看 token loss。

### D. 验证 Qwen

固定 clip：

```bash
CUDA_VISIBLE_DEVICES=0 /data1/conda_envs/qwen3vl_shellgame/bin/python scripts/mem/eval_shellgame_qwen3vl_lora.py --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct --adapter-path checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375 --manifest artifacts/shellgame_qwen3vl_gt_event_sft_v1/val.jsonl --output evaluation/shellgame/qwen3vl_lora_step375_recheck.jsonl --num-episodes 100 --batch-size 2 --overwrite
```

滑窗 event trigger：

```bash
CUDA_VISIBLE_DEVICES=0 /data1/conda_envs/qwen3vl_shellgame/bin/python scripts/mem/eval_shellgame_qwen3vl_sliding_trigger.py --model-path /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct --adapter-path checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375 --manifest artifacts/shellgame_qwen3vl_gt_event_sft_v1/val.jsonl --output evaluation/shellgame/qwen3vl_step375_sliding_recheck.jsonl --num-episodes 20 --window-size 10 --window-stride 1 --cluster-gap 3 --overwrite
```

已知边界：正常 10 帧交换为 60/60 正确；快速 6 帧交换显著下降。真机前必须测试不同速度、窗口偏移、帧率、丢帧和半事件窗口。

### E. 生成 teacher target

有两种模式：

1. **当前高效 ShellGame 模式**：JAX 训练用 metadata relation 作为已验证 Qwen 输出的 clean cached proxy，再由 frozen symbolic teacher 生成 memory target。优点是快；限制是不能测量 Qwen 错误传播。
2. **通用真实蒸馏模式**：离线运行 frozen Qwen，校验和去重事件，丢弃低置信度事件，再递归生成并缓存每个事件后的 teacher memory。若报告声称 Qwen pseudo-label 蒸馏，应使用此模式并报告 pseudo-label 覆盖率和准确率。

### F. 学生图结构 self-test

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py --exp-name qwen_distill_self_test --self-test-only
```

期望输出：

```text
Qwen direct-visual distillation self-test passed:
student=(2, 3, 8, 16), teacher=(2, 3, 8, 16), student_relation_params=0
```

### G. 分段训练学生 MEM

step 0--249：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py --exp-name qwen_distilled_direct_visual_memory250_260825 --teacher-checkpoint checkpoints/shellgame_stage_slot_only_relation_recurrent_probe/stage_slot_only_random_relation_frozen_memory_1k_260821/500/params --steps 250 --warmup-steps 50 --peak-lr 3e-4 --memory-distill-weight 1.0 --stage-slot-weight 0.25 --batch-size 12 --fsdp-devices 6 --eval-interval 50 --eval-batches 10 --save-interval 250 --num-workers 0
```

step 250--499：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py --exp-name qwen_distilled_direct_visual_memory250_260825 --steps 500 --warmup-steps 0 --peak-lr 6e-5 --memory-distill-weight 1.0 --stage-slot-weight 0.25 --batch-size 12 --fsdp-devices 6 --eval-interval 50 --eval-batches 10 --save-interval 250 --num-workers 0 --resume
```

step 500--999：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py --exp-name qwen_distilled_direct_visual_memory250_260825 --steps 1000 --warmup-steps 0 --peak-lr 4e-5 --decay-lr 4e-6 --memory-distill-weight 1.0 --stage-slot-weight 0.25 --batch-size 12 --fsdp-devices 6 --eval-interval 100 --eval-batches 20 --save-interval 500 --keep-period 499 --num-workers 0 --resume
```

`--steps` 是总 global step，不是“再训练这么多步”。从 step499 到 step999 使用 `--steps 1000 --resume`。

## 6. 评估与验收

### 6.1 必看指标

- `memory_distill_loss`、`memory_cosine_loss`、`memory_mse_loss`；
- `stage_memory_loss`；
- `slot_0_accuracy`、`slot_1_accuracy`、`slot_2_accuracy`；
- `stage_memory_accuracy`、`final_memory_accuracy`；
- student/teacher memory token variance。

不能只看总 loss。Memory loss 下降但 final accuracy 仍约 33% 时，说明对齐尚未进入可读语义区域。

### 6.2 已验证曲线

| checkpoint | stage 1 | stage 2 | final | stage mean | memory cosine |
|---:|---:|---:|---:|---:|---:|
| 249 | 53.3% | 42.5% | 38.3% | 44.7% | 0.464 |
| 499 | 75.8% | 68.3% | 52.5% | 65.6% | 0.396 |
| 599 | 90.8% | 85.8% | 69.2% | 81.9% | 0.293 |
| 699 | 95.4% | 94.2% | 82.5% | 90.7% | 0.216 |
| 899 | 98.3% | 97.9% | 94.2% | 96.8% | 0.138 |
| 999 | 100.0% | 98.3% | 94.2% | 97.5% | 0.122 |

step499 的 52.5% 是欠优化，不是容量上限；小学习率继续到 step999 后验证指标同步提升。

### 6.3 导出全量 memory cache

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 .venv/bin/python scripts/mem/cache_shellgame_qwen_distilled_visual_memory.py --exp-name qwen_distilled_direct_visual_memory250_260825 --all-episodes --batch-size 12 --fsdp-devices 6 --overwrite
```

产物 `artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_all5000_260825.npz` 包含：

```text
episode_index          [5000]
initial_slot           [5000]
final_label            [5000]
final_prediction       [5000]
final_memory           [5000,128,64]
memory_templates       [5000,128,64]
episode_template_index [5000]
```

全 5,000 episode 准确率为 **94.6%**。

表中的 step999 是 `num_train_steps=1000` 对应的零起始 checkpoint 编号；held-out 固定评估在 step999 的 final accuracy 为 94.2%，全 5,000 episode cache 的总体准确率为 94.6%，两者统计范围不同。

### 6.4 必做消融

1. 正常历史；
2. swap clip 置零；
3. 跨 episode 打乱 swap clip；
4. `--shuffle-teacher-targets`；
5. zero-memory/chance baseline；
6. episode-held-out 验证。

正常历史必须显著优于打乱和置零历史，否则高准确率可能来自 initial label、episode identity 或数据泄漏。

## 7. 接入 action model

action model 应读取学生 `final_memory [128,64]`，不是 Qwen JSON 或 teacher relation。已验证 compatibility：

| memory | 适配前 action loss | 适配 250 steps 后 |
|---|---:|---:|
| direct-visual student | 0.04539 | 0.03399 |
| symbolic teacher | 0.03125 | 0.02631 |
| wrong episode | 0.09550 | 0.09757 |
| zero memory | 0.10582 | 0.07321 |

推荐先冻结 MEM，只训练 action memory adapter；闭环有效后再用小学习率联合微调。

## 8. 迁移到其他任务

ShellGame 的固定三段 swap 只是验证实现。通用接口应保持：

```text
Qwen: short video + context -> event/state delta + confidence
Teacher: previous state + event -> next teacher memory
Student: previous memory + compressed images -> next student memory
```

迁移时需要替换 cup-specific schema、camera/world adapter、固定 `SWAP_SLICES`、stage-slot classifier 和固定三次 update。推荐改为重叠滑窗 + event trigger + 去重，每个被接受事件只更新一次 memory，并用 `event_valid` mask 忽略不完整/低置信度窗口。

## 9. 常见问题

### 长期停留在 33%

检查 visual encoder 是否真的可训练、freeze filter、teacher 参数映射、`episode_index`、左右坐标转换、完整事件窗口和 teacher memory variance。

### Memory cosine 降但准确率不升

通常是学习率过早衰减、只对齐公共 token、read adapter 坐标不一致或第三次 update 累积误差。当前 recipe 已证明 250/500 steps 不足，不能过早归因于 memory 容量。

### 训练高、验证低

必须按 episode 分割。固定 60 帧 prefix 不能同时出现在 train/val，并应用 wrong-episode history 排除 identity shortcut。

### Qwen 固定 clip 很准，真机触发失败

固定 GT 阶段准确率不等于滑窗触发能力。还要测试跨边界、半事件、重复/漏触发、不同速度、帧率、丢帧和视角变化。

## 10. 代码与产物索引

- Qwen manifest：`scripts/mem/build_shellgame_qwen3vl_sft_manifest.py`
- Qwen LoRA：`scripts/mem/train_shellgame_qwen3vl_lora.py`
- Qwen 固定 clip eval：`scripts/mem/eval_shellgame_qwen3vl_lora.py`
- Qwen 滑窗 eval：`scripts/mem/eval_shellgame_qwen3vl_sliding_trigger.py`
- Qwen 输出协议：`src/openpi/tasks/shellgame/qwen3vl_sft_contract.py`
- 坐标/事件 adapter：`src/openpi/tasks/shellgame/qwenvl_event_adapter.py`
- recurrent core：`src/openpi/models/siglip_mem_semantic.py`
- ShellGame memory：`src/openpi/tasks/shellgame/semantic_memory.py`
- 蒸馏入口：`examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py`
- 通用 memory trainer：`scripts/mem/train_semantic_memory.py`
- memory cache：`scripts/mem/cache_shellgame_qwen_distilled_visual_memory.py`

已验证 checkpoints：

```text
Qwen:
checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375

Symbolic teacher:
checkpoints/shellgame_stage_slot_only_relation_recurrent_probe/
  stage_slot_only_random_relation_frozen_memory_1k_260821/500/params

Student MEM:
checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/
  qwen_distilled_direct_visual_memory250_260825/999
```

相关报告：

- `evaluation/shellgame/qwen_distilled_direct_visual_recurrent_memory_260825/README.md`
- `docs/qwen3vl_sliding_event_trigger_generalization_260825.md`
- `evaluation/shellgame/qwen_distilled_direct_visual_memory_action_compatibility_260825.json`
- `evaluation/shellgame/qwen_distilled_direct_visual_memory_action_compatibility_after_action250_260825.json`

## 11. 合作者最小执行清单

1. 检查 raw 图像、时序和 metadata。
2. 构建 Qwen manifest，并按 episode 分 train/val。
3. 在独立环境训练并验证 Qwen LoRA。
4. 离线生成 teacher event/memory，或明确声明 clean metadata proxy。
5. 运行 student self-test，确保 relation 不进入学生。
6. 分三段训练 MEM 到语义指标收敛。
7. 做 zero/shuffle/wrong-teacher 消融。
8. 导出全量 memory cache 并核对准确率。
9. 冻结 MEM 接入 action model，先做 compatibility，再做闭环。

核心思想：Qwen 负责把视频解释成事件，recurrent teacher 把事件变成连续状态轨迹，低计算量学生 MEM 再仅凭图像和 previous memory 复现这条轨迹。
