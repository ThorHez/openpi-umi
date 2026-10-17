# ShellGame 通用 Semantic Memory 框架训练步骤

> 本文只描述代码整理后的通用训练接口。<br>
> 不再使用 `examples/shellgame/train_*probe.py`。<br>
> 不展开 V6/V9 数据生成过程。

## 1. 使用的代码

通用模型组件：

```text
src/openpi/models/siglip_mem_semantic.py
src/openpi/models/pi0_mem_semantic_action.py
```

统一训练入口：

```text
scripts/mem/train_semantic_memory.py
scripts/mem/train_mem.py
```

ShellGame只在task adapter和recipe中定义杯子类别、三段交换、数据路径和EEF动作语义：

```text
src/openpi/tasks/shellgame/semantic_memory.py
src/openpi/tasks/shellgame/pi0_mem_semantic_action.py
src/openpi/training/mem/recipes/shellgame_semantic_memory_pretrain.py
src/openpi/training/mem/recipes/shellgame_semantic_action.py
```

整体训练由两个正式阶段组成：

```text
阶段1：Semantic memory监督预训练
  train_semantic_memory.py
  ↓ 输出完整模型checkpoint

阶段2：Memory-conditioned Pi0.5 action训练
  train_mem.py
  ↓ 显式加载阶段1 checkpoint

阶段3：固定协议闭环测试和checkpoint选择
```

旧版M1～M4的初始杯、交换关系和递归memory实验，现在合并到阶段1的一个联合目标中；EEF action训练由阶段2统一完成。memory-to-action接口会进入阶段2前向，但当前注册recipe默认冻结该接口，具体边界见第7节。

## 2. 数据准备

当前两个阶段都使用同一份absolute-EEF LeRobot数据：

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7
```

阶段1还需要与LeRobot episode index一一对应的原始metadata：

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset
```

已核对的数据规模：

```text
episodes: 5000
semantic labels per episode: 7
  - initial cup: 1
  - swap relation: 3
  - cup slot after each swap: 3
```

训练前检查：

```bash
test -d /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7
test -d /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_absolute_eef_phase_instruction_dataset
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7/meta/episodes.jsonl
df -h /data2
```

建议至少保留100 GiB可用空间。

## 3. 统一模型与输入契约

```text
history frames: 0..59，共60帧，stride=1
current frame: 动态当前帧
total input frames: 61
video layout: fixed_prefix_current

semantic memory: [B, 128, 64]
action queries: [B, 16, 1024]
action horizon: 16

EEF state: xyz(3) + rot6d(6) + gripper(1) = 10维
EEF action: xyz(3) + rotation-vector(3) + gripper(1) = 7维
```

模型pipeline：

```text
60帧历史
  → SigLIP patch embedding
  → 固定2×2空间pooling
  → factorized spatial/temporal encoder
  → 初始杯和三段交换关系
  → recurrent compact memory [128,64]
  → 16个learned queries读取memory
  → action-token cross-attention
  → Pi0.5 flow action expert
  → 16×7 absolute-EEF action chunk
```

## 4. 环境设置

所有命令从OpenPI根目录运行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_semantic_memory
export HF_HOME=/data2/hzl_workspace_for_pi_mem/.cache/huggingface
export HF_DATASETS_CACHE=/data2/hzl_workspace_for_pi_mem/.cache/huggingface/datasets
```

六卡训练统一使用：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
--fsdp-devices 6
```

## 5. 阶段1：Semantic memory监督预训练

### 5.1 目的

一次训练同时完成：

1. 从frame 0识别初始球杯；
2. 从三段交换视频识别三次交换关系；
3. 使用共享recurrent updater连续更新compact memory；
4. 从每个stage memory预测交换后的球杯；
5. 在episode-held-out验证集上检查最终memory是否能正确表示球的位置。

### 5.2 数据采样

- 每个episode只取frame 59对应的一行；
- 每行输入固定frame `0..59`历史；用于构造61帧张量的当前帧固定为frame 59；
- prompt不参与memory预训练；
- 按episode划分90%训练、10%验证，避免同一episode泄漏到两侧。

标签从原始metadata自动构造，不需要额外生成分类标注文件。

阶段1的recurrent update采用teacher forcing：ground-truth初始杯和三段交换关系负责驱动memory更新；初始杯logits和交换关系logits仍分别接受交叉熵监督。因而`final_memory_accuracy`表示在正确语义事件输入下的memory组合能力，必须同时结合held-out `initial_accuracy`和`relation_accuracy`判断，不能单独把它当成完整全视觉准确率。

### 5.3 初始化与训练模块

基础权重来自：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
```

加载时会重新初始化并训练：

```text
HistoryFrame0InitialCupClassifier
HistoryThreeSwapVisualRelationMemoryTracker
```

冻结：

- PaliGemma/SigLIP主干；
- Pi0.5 action expert；
- memory-to-action query resampler；
- action memory cross-attention。

### 5.4 启动命令

把 `<TAG>` 替换为唯一实验名：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
nohup uv run python scripts/mem/train_semantic_memory.py \
  shellgame_semantic_memory_pretrain \
  --exp-name semantic_memory_<TAG> \
  --num-train-steps 6000 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --val-ratio 0.1 \
  --eval-interval 250 \
  --eval-batches 20 \
  --log-interval 10 \
  --save-interval 500 \
  --keep-period 1000 \
  --lr-schedule.warmup-steps 300 \
  --lr-schedule.peak-lr 3e-4 \
  --lr-schedule.decay-steps 6000 \
  --lr-schedule.decay-lr 3e-5 \
  --initial-loss-weight 1.0 \
  --relation-loss-weight 1.0 \
  --stage-memory-loss-weight 1.0 \
  --no-memory-train-augmentation \
  --no-wandb-enabled \
  > train_semantic_memory_<TAG>.log 2>&1 &
```

输出目录：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/
  shellgame_semantic_memory_pretrain/
  semantic_memory_<TAG>/
```

最终参数通常位于：

```text
checkpoints/shellgame_semantic_memory_pretrain/semantic_memory_<TAG>/5999/params
```

### 5.5 验收指标

日志中关注：

```text
val/initial_accuracy
val/relation_accuracy
val/stage_memory_accuracy
val/final_memory_accuracy
val/initial_loss
val/relation_loss
val/stage_memory_loss
val/loss
```

进入阶段2前要求：

- 四个held-out accuracy均稳定接近100%；
- train和validation趋势一致；
- `relation_accuracy`不能只在训练集达到100%；
- 日志确认只有memory相关模块具有非零梯度；
- final checkpoint和完整训练日志存在。

### 5.6 续训

只有确实需要增加步数时才续训，并复用相同experiment name：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
nohup uv run python scripts/mem/train_semantic_memory.py \
  shellgame_semantic_memory_pretrain \
  --exp-name semantic_memory_<TAG> \
  --resume \
  --num-train-steps 8000 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  > continue_semantic_memory_<TAG>.log 2>&1 &
```

续训时会恢复model、optimizer和step count；不能只把旧params当成新初始化。

## 6. 阶段2：Memory-conditioned action训练

### 6.1 目的

使用阶段1训练好的semantic memory预测absolute-EEF action chunk。

前向路径会调用：

```text
RawMemoryQueryResampler
ActionMemoryCrossAttention
Pi0.5 action expert
```

当前已注册recipe冻结semantic memory和基础视觉主干，只训练：

- Pi0.5 action expert；
- `action_in_proj`；
- `action_out_proj`；
- `time_mlp_in`；
- `time_mlp_out`。

### 6.2 数据采样

- 数据：`shellgame_lerobot_absolute_eef_raw7`；
- 固定历史：frame `0..59`；
- 当前/action observation：frame `59..153`；
- episode-held-out validation：10%；
- action horizon：16；
- gripper loss weight：4；
- episode末端超出frame154的future action自动mask。

### 6.3 必须正确衔接阶段1 checkpoint

这是最容易出错的地方。

`pi0_mem_semantic_action_shellgame_eef7` recipe的默认loader仍指向旧V6 checkpoint。执行多阶段训练时，必须显式覆盖：

```text
--weight-loader.params-path=<阶段1 checkpoint>/params
```

如果漏掉这个参数，阶段2不会使用刚完成的semantic-memory预训练结果。

### 6.4 启动命令

```bash
MEMORY_PARAMS=/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/shellgame_semantic_memory_pretrain/semantic_memory_<TAG>/5999/params

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
nohup uv run python scripts/mem/train_mem.py \
  pi0_mem_semantic_action_shellgame_eef7 \
  --exp-name semantic_action_<TAG> \
  --weight-loader.params-path "${MEMORY_PARAMS}" \
  --num-train-steps 6000 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --val-ratio 0.1 \
  --eval-interval 250 \
  --eval-batches 20 \
  --log-interval 10 \
  --save-interval 500 \
  --keep-period 1000 \
  --lr-schedule.warmup-steps 300 \
  --lr-schedule.peak-lr 3e-5 \
  --lr-schedule.decay-steps 6000 \
  --lr-schedule.decay-lr 3e-6 \
  --model.gripper-loss-weight 4.0 \
  --no-wandb-enabled \
  > train_semantic_action_<TAG>.log 2>&1 &
```

输出目录：

```text
checkpoints/pi0_mem_semantic_action_shellgame_eef7/semantic_action_<TAG>/
```

### 6.5 启动后的检查

日志中确认：

1. `CheckpointWeightLoader`加载的是阶段1路径；
2. 固定历史为60帧、总输入为61帧；
3. action维度为32，但只有前7维参与loss；
4. gripper index为6，权重为4；
5. memory和SigLIP梯度为0；
6. action expert和action/time projection梯度非0；
7. `val/action_loss`稳定下降且没有NaN/Inf。

### 6.6 Checkpoint选择

保存并测试：

```text
step 500
step 1000
step 1500
step 2000
后续每500～1000步checkpoint
最终step 5999
```

最终选择不能只看最低`val/action_loss`，必须运行固定seed闭环测试，并分别统计：

- 正确选杯率；
- 到达目标杯上方的比例；
- 成功下降比例；
- 正确闭爪比例；
- 成功抬升比例。

## 7. 当前通用recipe的边界

### 7.1 当前action recipe不是旧V10的60/30/10混合训练

当前注册的：

```text
pi0_mem_semantic_action_shellgame_eef7
```

只使用nominal absolute-EEF数据。它不会自动混合V6 replay和V9 timing数据，也没有迁移旧版V10的动态采样器。

因此本文命令是“通用semantic memory + nominal EEF action”的正式基线，不应把结果命名为V10复现。

若后续需要V10，应在：

```text
src/openpi/training/mem/recipes/
```

新增一个通用mixed-action recipe，并继续使用同一个`train_mem.py`入口；不应重新调用旧`examples/shellgame`训练脚本。

### 7.2 当前action recipe默认不更新memory-action interface

`Pi0MemSemanticActionConfig`已经提供：

```text
get_freeze_filter_memory_interface_finetune()
```

但当前注册recipe使用的是：

```text
get_freeze_filter_action_finetune()
```

因此`HistoryRawMemoryQueryResampler`和`ActionMemoryCrossAttention`参与前向，但保持冻结。

如果实验目标是重新对齐新训练的memory和action token，应该新增一个独立的interface-finetune recipe，训练：

- memory query resampler；
- action memory cross-attention；
- action expert和action/time projection；

同时继续冻结semantic memory和视觉主干。当前命令行不能直接覆盖`freeze_filter`，不要假装加一个CLI参数就能完成这个阶段。

## 8. 阶段3：闭环验收

闭环测试必须保持：

```text
seed: 260813
fixed history: frame 0..59
num_frames: 61
frame_stride: 1
action mode: absolute EEF raw7
osc input: absolute
rotation convention: openpi
max policy steps: 150
```

推荐先进行20 episode checkpoint筛选，再对最佳checkpoint执行100 episode测试。

如果选杯正确但抓取失败，说明memory已经影响approach方向，问题主要在下降、持续XY纠偏、闭爪或抬升；不应直接重新训练memory。

## 9. 合作方执行顺序

```text
1. 检查LeRobot数据和raw metadata
2. 运行阶段1 semantic-memory训练
3. 检查四个held-out memory accuracy
4. 记录阶段1最终params路径
5. 在阶段2命令中显式覆盖weight-loader.params-path
6. 运行action训练并保存多个中间checkpoint
7. 固定seed做20 episode筛选
8. 对最佳checkpoint运行100 episode闭环测试
9. 交付命令、日志、checkpoint、result.json和视频
```

## 10. 每次实验需要记录

```text
experiment name:
code revision / patch:
memory checkpoint:
action initialization:
dataset root:
full command:
visible GPUs:
FSDP devices:
trainable modules:
frozen modules:
validation metrics:
closed-loop selection rate:
closed-loop lift success rate:
result.json:
video directory:
conclusion:
```
