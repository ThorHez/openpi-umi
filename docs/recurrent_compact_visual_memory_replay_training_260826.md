# 递归紧凑视觉记忆模型：结构、训练与验证说明

> 日期：2026-08-26  
> 版本：v1.0（正式版）  
> 状态：ShellGame 记忆跟踪结构已完成可行性验证  
> 主要实现：`examples/shellgame/train_replay_unrolled_clip6_memory_probe.py`

## 1. 文档目的

本文说明目前已经验证可行的递归紧凑视觉记忆模型，包括：

- 为什么不需要一次把完整 60 帧送入大规模时空 Transformer；
- 如何把完整 episode 拆成连续、不重叠的 6 帧 clip；
- 如何用上一时刻的紧凑 memory 和当前 clip 持续更新状态；
- 如何通过完整 recurrent unroll，让最终监督反向传播到最早的 clip；
- carry-biased soft gate、损失函数、初始化方式和训练命令；
- 已完成实验、因果消融结果、当前能力边界和 action 集成接口。

这里记录的是已经由实验验证的版本，不是最早的无门控 replay 版本，也不是依赖显式 relation token 的三阶段 tracker。

## 2. 已验证结论

该结构能够只依赖：

```text
当前 6 帧图像 + 上一次 compact memory
```

连续处理一条 60 帧观察序列，并通过 10 次共享 recurrent update 跟踪最终球位。Updater 的输入不包含：

- Qwen 输出；
- swap relation ID 或 relation probability；
- event/phase ID；
- 当前是第几次交换的标识。

在 500 个 held-out episode、每个 episode 的全部 6 种 clip offset 上，共 3000 条验证序列中，step 1499 checkpoint 的最终球位准确率为 **77.23%**。不同 offset 的准确率为 74.6%--79.6%，说明模型并不依赖某一种严格的 clip/交换边界对齐。

这个结果证明以下训练机制是成立的：

1. 完整 episode 可以作为 replay item 保存；
2. 训练时可用当前学生重新展开 recurrent memory；
3. 最终 loss 可以穿过 10 次 memory update 反传到最早的 clip；
4. 非重叠小 clip 可以替代一次性处理完整长视频；
5. 软门控的递归更新可以在无显式 event 输入时学习有效的状态保持与写入。

但当前结果不能解释为“完全无标签、从原始图像端到端达到 77.23%”。本实验仍使用 GT 初始球位初始化 memory，并使用已知交换结束帧构造训练目标。详细边界见第 12 节。

## 3. 模型整体结构

```text
60 帧历史 RGB
    │
    ▼
冻结的 SigLIP / PaliGemma image encoder
每帧得到 256 × 1152 patch tokens
    │
    ▼
固定 2×2 空间平均池化
16×16 patches → 8×8 patches
每帧 64 × 1152 tokens
    │
    ▼
按时间拆成 10 个连续、不重叠的 6 帧 clip
[B, 10, 6, 64, 1152]
    │
    ▼
共享 DirectVisualSegmentEncoder
LN → 1152→256 投影 → 2 层 factorized space-time Transformer
→ 256→64 投影
每个 clip 得到 6×64=384 个 evidence tokens
[B, 10, 384, 64]
    │
    ▼
初始化 compact memory
128 memory tokens × 64 dim
第 0 个 token 注入初始球位 one-hot
    │
    ▼
共享 ReplayRecurrentMemoryUpdater，连续执行 10 次
candidate_t = UpdateTransformer(memory_{t-1}, evidence_t)
gate_t = sigmoid(MLP(summary(memory_{t-1}), summary(evidence_t)))
memory_t = memory_{t-1} + gate_t × (candidate_t - memory_{t-1})
    │
    ▼
每一步 memory 通过共享 adapter/readout 解码球位
3 类：left / middle / right
```

### 3.1 主要张量形状

| 张量 | 形状 | 说明 |
|---|---|---|
| 历史图像 | `[B, 60, H, W, 3]` | 只读取观察阶段 frame 0--59 |
| SigLIP patch | `[B, 60, 256, 1152]` | 冻结视觉主干的带位置编码输出 |
| 空间池化后 patch | `[B, 60, 64, 1152]` | 16×16 网格池化为 8×8 |
| clip 网格 | `[B, 10, 6, 64, 1152]` | 10 个非重叠 clip |
| clip evidence | `[B, 10, 384, 64]` | 每个 clip 保留完整时空 token，而非压成一个向量 |
| recurrent memory | `[B, 10, 128, 64]` | 每次更新后的全部 memory state |
| soft gate | `[B, 10]` | 每个 clip 一个标量写入强度 |
| slot logits | `[B, 10, 3]` | 每次更新后的球位预测 |

## 4. 各模块设计

### 4.1 冻结视觉主干

历史图像通过 Pi/PaliGemma 中已有的 SigLIP image encoder 提取 patch token：

```python
_, encoder_out = self.PaliGemma.img(history, train=False)
patches = encoder_out["with_posemb"][:, :60]
```

训练本实验时，基础 Pi 模型和视觉主干被冻结，只训练 replay tracker。这样能降低训练成本，也避免在小规模 ShellGame 数据上破坏通用视觉表征。

### 4.2 拓扑保持的空间池化

每帧原始 256 个 patch 对应 16×16 网格。使用固定 2×2 平均池化得到 8×8，即 64 个 patch：

```text
256 patches/frame → 64 patches/frame
```

该操作降低后续时空 Transformer 的空间 token 数量，同时保留二维位置关系。它不同于对一帧直接做 global pooling；小球与杯子的相对空间位置仍然能够保留。

### 4.3 轻量 clip encoder

每个 6 帧 clip 使用同一个 `DirectVisualSegmentEncoder`：

1. LayerNorm；
2. 将 patch width 从 1152 投影到 256；
3. 加入 6 帧相对时间位置编码；
4. 运行 2 层 factorized space-time Transformer；
5. 投影到 memory width 64。

Factorized block 先在每个空间位置上做 temporal attention，再在每一帧内部做 spatial attention，最后运行 MLP：

```text
temporal attention per spatial cell
    → spatial attention per frame
    → MLP
```

因此后续 Transformer 的计算对象是池化后的 6×64 token，而不是完整 60×256 patch 序列。

### 4.4 Persistent compact memory

Memory 的固定形状为：

```text
M = 128 tokens
D = 64 channels
```

它由可学习 `base_memory` 初始化。ShellGame 实验中，第 0 个 memory token 的前 3 个通道加上初始球位 one-hot：

```python
memory[:, 0, :3] += one_hot(initial_slot)
```

初始球位只是本任务的条件输入方式，不是通用 memory core 的结构限制。其他任务可以换成初始状态 encoder、语言条件、机器人状态或任意任务 adapter。

### 4.5 Shared recurrent updater

10 个 clip 共享完全相同的 updater 参数。每次更新包含 2 个 `MemoryUpdateBlock`，每个 block 依次运行：

1. memory 对当前 evidence 的 cross-attention；
2. memory token 之间的 self-attention；
3. MLP residual update。

对应过程为：

\[
\tilde{M}_t = F_\theta(M_{t-1}, E_t)
\]

其中 \(E_t\) 是第 \(t\) 个 6 帧 clip 的视觉 evidence，\(F_\theta\) 在全部 10 次更新中共享。

### 4.6 Carry-biased soft gate

无门控 updater 会在没有完成交换、只看到局部交换证据时反复覆盖 memory。最终采用一个由旧 memory 和当前 evidence 共同预测的连续标量 gate：

\[
g_t = \sigma\left(\mathrm{MLP}\left([
\operatorname{mean}(\operatorname{LN}(M_{t-1})),
\operatorname{mean}(\operatorname{LN}(E_t))]
\right)\right)
\]

\[
M_t = M_{t-1} + g_t(\tilde{M}_t-M_{t-1})
\]

Gate 输出层使用零 kernel 和 bias `-2` 初始化：

\[
g_0=\sigma(-2)=0.1192
\]

因此训练开始时更接近“保留旧 memory”，而不是每个 clip 都完全覆盖状态。Gate 不接收 GT event 或 phase。

正式模型的平均 gate 为：

- transition clip：0.1866；
- hold clip：0.1659。

二者有软区分，但不是硬开关。因果消融证明，将 gate 变成硬 event mask 会明显破坏性能。

### 4.7 Memory readout

为了监督每一次 recurrent state，模型使用共享 read adapter 和 readout：

1. 一个可学习 query 从 128×64 memory 中读取语义；
2. 将结果投影为 1152 维 residual，并加到 256 个 diagnostic current token；
3. attention pooling；
4. 输出 left/middle/right 三分类 logits。

该 readout 是训练与诊断接口。接入 action model 时，应该把最终或持续更新后的原始 memory token 交给 action-memory cross-attention，而不是只把三分类概率当作 memory。

## 5. Replay 数据与标签

### 5.1 Replay item

每个完整 60 帧观察 episode 是一条 replay item。当前数据读取配置为：

```text
num_frames = 61
history_frames = 60
frame_stride = 1
video_layout = fixed_prefix_current
fixed_prefix_frames = 60
min_frame_index = max_frame_index = 59
```

实际 tracker 只读取 frame 0--59。Frame 60 已经进入 `robot_approach`，会通过专家手臂运动泄漏正确杯位，因此不得进入 memory 训练输入。

### 5.2 Episode 标签表

原始 ShellGame metadata 被转换为：

```text
episode_index → [initial_slot,
                 relation_1, relation_2, relation_3,
                 slot_after_swap_1,
                 slot_after_swap_2,
                 slot_after_swap_3]
```

本 replay 模型只使用：

```text
initial_slot
slot_after_swap_1
slot_after_swap_2
slot_after_swap_3
```

三个 relation label 不进入 updater，也不计算 relation loss。

### 5.3 随机 offset 与零填充

训练时，每条 replay item 独立采样：

```text
offset ∈ {0, 1, 2, 3, 4, 5}
```

从该 offset 开始选取历史帧，末尾不足的部分使用零 patch 填充，使窗口长度保持 60：

```text
offset = 0: frame 0..59
offset = 5: frame 5..59 + 5 个 zero frames
```

然后固定切成 10 个不重叠的 6 帧 clip。每个真实帧最多出现一次，不使用重叠滑窗。随机 offset 会让交换过程落在不同 clip 内部或跨越相邻 clip，从而训练模型适应边界变化。

这里使用 patch-space zero padding 是为了严格避免读取 frame 60 之后的机器人动作，而不是一种必须沿用到所有任务的通用 padding 方案。在线部署时应使用有效 clip mask 或只在收齐真实帧后更新。

## 6. Recurrent unroll 与监督构造

### 6.1 完整学生 unroll

一次训练前向会使用当前学生模型重新计算完整链条：

```text
M0
 → updater(M0, clip0) → M1
 → updater(M1, clip1) → M2
 ...
 → updater(M9, clip9) → M10
```

默认 `detach_between_clips=False`，因此不会在 clip 之间执行 `stop_gradient`。最终 loss 对早期 clip encoder 和 updater 参数存在真实梯度路径。Self-test 已验证 final logit 对第一个 clip 的梯度非零。

这与“提前把旧模型算出的 memory 存进 replay buffer 并直接训练下一步”不同：buffer 保存的是原始完整 episode，memory state 每次由当前学生重新 rollout，避免 stale hidden state。

### 6.2 每个 clip 的状态标签

ShellGame 中三次交换的结束帧为：

```text
29, 39, 49
```

对每个 clip，根据 clip 结束帧已经越过几个交换结束点，选择对应的 committed slot：

```text
尚未完成 swap 1 → initial_slot
完成 swap 1       → slot_after_swap_1
完成 swap 2       → slot_after_swap_2
完成 swap 3       → slot_after_swap_3
```

这些 GT 时间点仅用于构造 loss label 和指标 mask，不作为模型输入。对于真实任务，应由数据标注、状态变化标签或可自动计算的弱监督产生 committed-state target；不应把 GT event mask直接作为推理 gate。

### 6.3 平衡后的损失函数

最终有效 recipe 使用：

\[
L = L_{final} + L_{transition} + L_{hold}
\]

其中：

- \(L_{final}\)：最后一个 recurrent state 的最终球位交叉熵；
- \(L_{transition}\)：刚跨过交换结束点的 clip 上，更新后球位的交叉熵；
- \(L_{hold}\)：前 9 个 clip 中没有新交换完成时，对当前 committed state 的交叉熵。

三个 loss group 的权重均为 1。先分别在各自 mask 内求平均，再相加，避免数量更多的 hold clip 主导训练。

正式训练参数为：

```text
final_slot_weight       = 1
intermediate_slot_weight = 0
transition_slot_weight  = 1
hold_slot_weight        = 1
```

训练目标中显式关闭了旧语义 tracker 的其他 loss：

```text
initial_loss_weight = 0
relation_loss_weight = 0
```

注意：initial slot 仍然作为条件注入，只是不训练 initial classifier。

## 7. 初始化与冻结策略

### 7.1 推荐：warm initialization

推荐从已经完成 Qwen 教师蒸馏的 direct-visual recurrent tracker 初始化：

```text
checkpoints/
shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/
qwen_distilled_direct_visual_memory250_260825/999/params
```

Warm loader 会：

- 复制已有 visual segment encoder、memory updater、adapter 和 readout 权重；
- 将原 10 帧 temporal position embedding 截取为前 6 帧；
- 对新加入的 8 个 gate parameter leaves 使用当前初始化；
- 保持基础 Pi 参数与来源 checkpoint 一致。

Qwen 不参与该 replay 模型的推理。它只通过 warm checkpoint 提供更好的视觉转换初始化。

### 7.2 Scratch control

Scratch 模式保留相同的 Pi base，但把整个 replay tracker 随机初始化。1000 steps 后 scratch 的最终准确率只有 45.0%，显著低于 warm 的 75.0%。因此在当前数据量和训练预算下，warm initialization 对样本效率非常重要。

### 7.3 可训练参数

当前 freeze filter 只解冻 `HistoryReplayUnrolledVisualMemoryTracker`，主要包括：

- `DirectVisualSegmentEncoder`；
- `base_memory`；
- recurrent updater；
- carry gate；
- memory read adapter；
- diagnostic current tokens 和 slot readout。

SigLIP/PaliGemma 和 Pi action backbone 均被冻结。

## 8. 推荐训练流程

所有命令均在仓库根目录执行：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi
```

### 8.1 Self-test

先检查张量形状、gate 初始化、无重复真实帧，以及 final loss 到第一段 clip 的梯度路径：

```bash
.venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_self_test \
  --init-mode=warm \
  --carry-gate --carry-gate-bias=-2 \
  --self-test-only
```

### 8.2 第一阶段：无 gate 的 warm 迁移基线

这一步是本次模型演化中使用过的迁移基线。若已有对应 checkpoint，可以直接进入 8.3；新任务迁移时建议保留同类对照。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_clean60_warm_bptt_1k_260825 \
  --init-mode=warm \
  --steps=1000 --batch-size=8 --num-workers=4 \
  --fsdp-devices=4 --eval-interval=100 --eval-batches=20 \
  --save-interval=500 --keep-period=1000 --overwrite
```

该基线最终准确率只有 36.88%，其用途是证明仅有完整 BPTT 仍不足以稳定更新 memory。

### 8.3 第二阶段：carry-gated replay/BPTT 正式训练

从无 gate 的 step 999 初始化，新 gate 使用 carry-biased 初始化：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_gated_clean60_warm_bptt_1k_260825 \
  --init-mode=warm \
  --warm-checkpoint=checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_clean60_warm_bptt_1k_260825/999/params \
  --steps=1000 --warmup-steps=100 \
  --peak-lr=1e-4 --decay-lr=1e-5 \
  --final-slot-weight=1 --intermediate-slot-weight=0 \
  --transition-slot-weight=1 --hold-slot-weight=1 \
  --carry-gate --carry-gate-bias=-2 \
  --batch-size=8 --num-workers=4 --fsdp-devices=4 \
  --eval-interval=100 --eval-batches=20 \
  --save-interval=500 --keep-period=1000 --overwrite
```

1000 steps 后的 held-out 指标：

| 指标 | 结果 |
|---|---:|
| final cup accuracy | 75.00% |
| all-clip state accuracy | 88.75% |
| transition endpoint accuracy | 85.62% |
| partial-swap hold accuracy | 91.76% |
| final CE | 0.5635 |

### 8.4 可选：低学习率续训 500 steps

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_gated_clean60_warm_bptt_1k_260825 \
  --init-mode=warm --steps=1500 --warmup-steps=100 \
  --peak-lr=1e-5 --decay-lr=2e-6 \
  --final-slot-weight=1 --intermediate-slot-weight=0 \
  --transition-slot-weight=1 --hold-slot-weight=1 \
  --carry-gate --carry-gate-bias=-2 \
  --batch-size=8 --num-workers=4 --fsdp-devices=4 \
  --eval-interval=100 --eval-batches=20 \
  --save-interval=250 --keep-period=999 --resume
```

小规模 eval 中最终准确率从 75.00% 提升至 78.12%；完整 3000-sequence 评估为 77.23%。增益有限，因此继续增加训练步数不是弥补剩余差距的主要手段。

### 8.5 Checkpoint

推荐用于复现实验的 checkpoint：

```text
checkpoints/shellgame_replay_unrolled_clip6_memory_probe/
replay_clip6_gated_clean60_warm_bptt_1k_260825/1499/params
```

## 9. 完整验证与因果消融

### 9.1 评估命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/eval_replay_clip6_gate_causal_ablation.py \
  --batch-size=4 --fsdp-devices=4 --eval-batches=125 \
  --oracle-open-gate=1.0 \
  --output=evaluation/shellgame/replay_clip6_gate_causal_ablation_step1499_260825.json
```

评估覆盖：

```text
500 held-out episodes × 6 offsets = 3000 sequences / condition
```

### 9.2 结果

| 条件 | final | swap 1 endpoint | swap 2 endpoint | swap 3 endpoint | partial hold |
|---|---:|---:|---:|---:|---:|
| normal learned gate | **77.23%** | **97.93%** | **91.60%** | **76.83%** | **89.41%** |
| third swap 后冻结 memory | 76.83% | 97.93% | 91.60% | 76.83% | 89.41% |
| 仅 GT swap-overlap clip 允许 learned gate | 47.43% | 50.43% | 45.53% | 47.43% | 62.50% |
| GT swap-overlap clip 强制 gate=1 | 36.70% | 49.73% | 42.10% | 36.70% | 52.70% |

### 9.3 消融解释

1. **尾部覆盖不是主要瓶颈。** 第三次交换后冻结 memory 只下降 0.40 个百分点。
2. **主要误差来自多次 transition 累积。** 准确率从 swap 1 的 97.93%，下降到 swap 2 的 91.60%，再下降到 swap 3 的 76.83%。
3. **不能用 GT event 作为硬门控。** 只允许交换窗口更新会降到 47.43%。模型需要在非交换、跨边界和局部交换 clip 上逐步积累上下文与未完成证据。
4. **candidate 不能完全覆盖 memory。** 强制 gate=1 只有 36.70%，说明 updater 已经与小幅 residual integration 共同校准。
5. **不存在单一 offset 灾难。** 六种 offset 的 final accuracy 为 74.6%、79.0%、79.6%、77.4%、78.2%、74.6%。
6. **边缘杯仍较困难。** center 为 87.75%，left 为 73.07%，right 为 70.76%。

因此下一步应该增强第二、第三次递归转换的状态鲁棒性和边缘目标的视觉证据，而不是增加硬 event trigger 或简单冻结尾部。

## 10. 训练时应重点监控的指标

| 指标 | 含义 | 异常信号 |
|---|---|---|
| `final_memory_accuracy` | 最终状态是否正确 | 长期接近 33% 表示没有学会跟踪 |
| `transition_endpoint_accuracy` | 新状态提交后是否正确 | 高 hold、低 transition 表示不会更新 |
| `partial_swap_hold_accuracy` | 交换尚未完成时能否保持旧状态 | 低值表示过早写入或中间证据破坏 memory |
| `clip_slot_accuracy` | 所有 clip 的平均状态准确率 | 需要与 final/transition 一起分析 |
| `transition_gate_mean` | transition 区域平均写入量 | 不要求接近 1；当前有效值约 0.19 |
| `hold_gate_mean` | hold 区域平均写入量 | 不要求为 0；当前有效值约 0.17 |
| `memory_token_variance` | memory 是否发生表示坍缩 | scratch 失败时曾降至约 `7.86e-6` |
| `memory_step_delta` | 相邻 recurrent state 的变化量 | 极小可能完全不写，极大可能持续覆盖 |
| `offset_k_final_accuracy` | 对 clip 边界偏移的鲁棒性 | 单一 offset 过低通常意味着边界依赖 |

不能只看总 loss。尤其需要同时满足：transition 能更新、partial clip 能保持、最终状态正确。

## 11. 在线推理方式

概念上的在线接口为：

```python
memory = initialize_memory(initial_condition)

while task_is_running:
    clip = collect_next_6_non_overlapping_frames()
    patches = frozen_visual_encoder(clip)
    evidence = clip_encoder(spatial_pool(patches))
    candidate = recurrent_updater(memory, evidence)
    gate = gate_predictor(memory, evidence)
    memory = memory + gate * (candidate - memory)
    action = action_policy(current_observation, memory)
```

推理时不需要 Qwen，也不需要重新输入之前 60 帧。历史被持续压缩在固定大小的 128×64 memory 中，因此 episode 变长时 memory 存储量保持不变。

当前实验代码为了训练效率，仍然一次接收 60 帧并在单个 JAX graph 内部展开 10 次；它在数学上对应上述在线过程，但尚未把 `memory` 暴露为跨 policy call 的持久 runtime state。正式接入机器人闭环前，需要把以下接口独立出来：

```text
encode_clip(frames_6) -> evidence
update_memory(previous_memory, evidence) -> next_memory
read_memory(next_memory, current_observation) -> action condition
reset_memory(initial_condition)
```

## 12. 当前能力边界

### 12.1 已经证明的部分

- 6 帧非重叠 clip 可以进行连续 recurrent update；
- full-unroll BPTT 的梯度链路成立；
- 128×64 compact memory 容量足以学到明显高于随机的三次交换跟踪；
- soft carry gate 明显优于无 gate；
- 不需要把 relation/event/Qwen 输出作为 updater 输入；
- random offset 能提供一定的时间边界鲁棒性；
- Qwen 可以只作为离线教师/初始化来源，推理时不需要 Qwen。

### 12.2 尚未证明或仍依赖任务条件的部分

- 当前训练和验证向 memory 注入 **GT initial slot**；尚未把 frame-0 initial classifier 的错误纳入 77.23% 指标；
- transition/hold label 依赖 ShellGame 的已知交换结束帧 29/39/49；虽然时间标签不进入模型，但 loss 构造仍是任务特定的；
- 当前最终验证是球位分类，不是 action policy 闭环成功率；
- 当前实现固定为 60 帧、6 帧 clip 和 10 次 update；更长、变速或异步真实视频仍需验证；
- edge cup 和连续多次 transition 的误差仍明显；
- 当前代码尚未实现跨 policy call 的 stateful runtime memory API。

因此准确的结论是：**模型结构和 replay-unrolled 训练机制可行，但仍需要完成 initial-state 感知、通用监督构造、stateful action 接口以及闭环 action 验证。**

## 13. 迁移到其他长时记忆任务的训练 recipe

通用部分可以保持不变：

```text
短视觉 clip encoder
    + 固定容量 memory tokens
    + shared recurrent updater
    + carry-biased soft gate
    + current-student full unroll
    + 跨 clip BPTT
```

任务侧只需要提供：

1. 连续 episode replay；
2. 初始状态条件或可学习 initial encoder；
3. 每个时间点应保持的 committed state，或等价的自监督/教师 embedding target；
4. final、state-change、state-hold 三类平衡监督；
5. episode-level held-out split。

推荐迁移步骤：

1. 先训练或蒸馏短 clip visual encoder，使单次状态变化可识别；
2. 用完整 episode replay、随机 clip offset 和当前学生 full unroll 训练；
3. 初期使用 carry bias `-2`，避免随机 updater 连续覆盖 memory；
4. 分组平衡 final/change/hold loss，而不是直接平均所有时间 token；
5. 对更新次数、clip offset、速度和 episode 长度做 held-out 消融；
6. 最后再冻结或低学习率微调 memory，并接入 action expert。

其他任务不应照搬“三个 swap”或固定 29/39/49 帧。更通用的监督可以来自：

- 环境状态变化；
- 人工或自动生成的 subgoal completion 标签；
- Qwen/VLM 离线教师 embedding；
- 相邻 clip 的一致性与未来状态预测；
- action/observation 可预测性；
- success-conditioned temporal contrastive loss。

核心原则是让模型学会“何时逐步整合新证据、何时保持已有状态”，而不是在推理时依赖 GT event 触发器。

## 14. 代码与实验文件

- 训练与模型实验实现：`examples/shellgame/train_replay_unrolled_clip6_memory_probe.py`
- Gate 因果消融：`examples/shellgame/eval_replay_clip6_gate_causal_ablation.py`
- 通用视觉 memory building blocks：`src/openpi/models/siglip_mem_semantic.py`
- Direct visual clip encoder 来源：`examples/shellgame/train_direct_visual_recurrent_stage_slot_probe.py`
- ShellGame label recipe：`src/openpi/training/mem/recipes/shellgame_semantic_memory_pretrain.py`
- 训练入口复用：`scripts/mem/train_semantic_memory.py`
- 完整实验记录：`docs/shellgame_replay_clip6_bptt_probe_260825.md`
- 正式因果消融结果：`evaluation/shellgame/replay_clip6_gate_causal_ablation_step1499_260825.json`

## 15. 最终建议

后续开发应以 step 1499 的 gated replay tracker 为结构基线，不再回退到：

- 无 gate 的十段连续覆盖；
- 只在 GT event 窗口更新；
- gate 强制等于 1；
- 把旧模型预计算的 stale memory 当作固定 replay state；
- 读取 frame 60 之后包含专家动作的泄漏数据。

下一阶段最合理的顺序是：

1. 将 updater 改造成真正 stateful 的单 clip runtime 接口；
2. 保持原始 memory token 接入 action-memory cross-attention；
3. 优先提升连续多次 transition 和 edge-object tracking；
4. 加入可学习 initial-state encoder，移除 GT initial slot；
5. 进行 memory classification 与 action closed-loop 的联合 held-out 评估。
