# Semantic Memory + Multi-Hypothesis Future Belief 设计方案

## 1. 文档目的

本文档用于为后续开发会话提供完整上下文，目标是在当前 history-only semantic memory 模型上，引入训练期 future frames，并学习多个可能的未来状态表示（multi-hypothesis future belief）。

核心目标如下：

1. 模型既学习“过去发生了什么”，也学习“未来可能发生什么”。
2. past 和 future 尽量共用 temporal encoder / memory updater 权重。
3. 真实 future frames 只在训练期间作为 teacher 使用。
4. 推理时不需要 future frames，只根据 history、current、state 和 language 预测 future belief。
5. 不把不确定的未来压缩成一个平均 latent，而是显式表示多个可能的未来及其概率。
6. 最终验证 future、multi-hypothesis 和 past/future 权重共享是否真正提高动作成功率和扰动恢复能力。

---

## 2. 当前仓库与模型背景

工作区：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi
```

当前 ShellGame semantic action 模型主要位于：

```text
src/openpi/tasks/shellgame/pi0_mem_semantic_action.py
src/openpi/tasks/shellgame/semantic_memory.py
src/openpi/models/siglip_mem_semantic.py
src/openpi/models/pi0_mem_semantic_action.py
src/openpi/training/mem/recipes/shellgame_semantic_action.py
src/openpi/training/mem/video_dataset.py
```

仓库中还保留了以下旧版 Past-Future 实验实现：

```text
src/openpi/models/pi0_mem_pf.py
src/openpi/models/pi0_mem_pf_safe.py
src/openpi/models/siglip_pf.py
```

但这些文件建立在老版 `pi0_mem` 的视觉 memory 数据流、模型接口和 checkpoint 约定之上。当前 `pi0_mem_semantic` 已经发生较大变化，因此：

1. **不能从 `pi0_mem_pf.Pi0MemPF` 继承新模型**；
2. **不能把 `siglip_pf` 作为当前 semantic 模型的视觉 backbone 替换进去**；
3. **不能假设旧 PF 的 tensor layout、freeze filter、loss hook 或 checkpoint remap 对当前模型仍然适用**；
4. **不能以修改旧 PF 为本任务的实现路线**；
5. 旧 PF 最多只能帮助理解 prior/posterior、训练期 future teacher 和推理期 causal prior 这些抽象概念，具体代码应重新基于当前 semantic 实现设计。

本项目的唯一实现基线应是当前：

```text
Pi0MemSemanticAction
ThreeSwapVisualRelationMemoryTracker
FactorizedSpaceTimeEncoder
SharedSegmentMemoryUpdater
RawMemoryQueryResampler
ActionMemoryCrossAttention
```

未来路径应沿着当前 semantic memory 的“视觉片段 -> 语义事件 -> recurrent memory update -> action-memory interface”数据流扩展，而不是沿旧 PF 的 UTR/GTCA 数据流扩展。

### 2.1 当前 semantic 模型的关键限制

当前实现存在以下约束：

1. 输入被强制为固定的 60 帧 history 加 1 帧 current：

   ```text
   [frame 0, ..., frame 59, current_t]
   ```

2. `Pi0MemSemanticAction._embed_current_prefix()` 使用 `video[:, -1]` 选择当前帧。添加 future 后，最后一帧将不再是 current，因此必须改成显式 `current_frame_index`。

3. `FixedPrefixCurrentVideoDataset` 当前明确禁止 `num_future_frames != 0`，需要扩展数据集。

4. semantic memory 在进入 action interface 前调用了：

   ```python
   jax.lax.stop_gradient(tracked["stage_memories"][:, -1])
   ```

   因而 action loss 无法反向训练 semantic memory。

5. 当前训练 freeze filter 主要训练 action expert，semantic memory 本身通常被冻结。

6. 当前 semantic tracker 的核心信息是初始杯位和三段 swap relation。完成 swap 后，目标杯位置基本不再变化。如果 future 只预测“球在哪个杯子”，新增信息有限。

因此 future belief 应重点覆盖机器人交互过程中的动态状态，例如 approach、alignment、contact、grasp、lift、碰杯、抓偏和恢复，而不应只重复最终杯位。

---

## 3. 核心概念

### 3.1 单一 future belief 的问题

普通 deterministic future prior 为：

```text
M_future_prior = P(M_history, current, robot_state, language)
```

但是同一个 history/current 下可能存在多种未来：

- 正确对准并成功抓取；
- XY 有偏差，随后校正；
- 碰动杯子；
- gripper timing 错误导致失败；
- 受到扰动后恢复或未恢复。

如果用 MSE 将这些未来拟合成一个单一 memory，模型容易学到不存在的“平均未来”。

### 3.2 Multi-hypothesis future belief

模型输出 `K` 个候选 future semantic memory 和对应概率：

```text
{(M_future_prior_1, p_1), ..., (M_future_prior_K, p_K)}
```

其中：

- `M_future_prior_k` 表示第 `k` 种可能未来的 semantic memory；
- `p_k` 表示该未来的预测概率；
- 初始建议 `K=4`。

注意：`K` 是模型输出的候选数量，并不意味着每条普通训练样本必须同时包含 `K` 条真实 future。最小版本中，每个样本只需要一条观测到的真实 future，模型通过跨样本的 winner/soft assignment 逐渐形成不同 future modes。

---

## 4. 推荐模型结构

整体结构完全沿当前 semantic tracker 展开：

```text
训练期

history segments
  -> PaliGemma.img
  -> shared FactorizedSpaceTimeEncoder
  -> history relation tokens
  -> shared SharedSegmentMemoryUpdater
  -> M_history
       │
       ├─ current + state + language
       │    -> K组 predicted future event tokens
       │    -> 同一个 SharedSegmentMemoryUpdater
       │    -> {M_future_prior_k, p_k}
       │                         └─> Action prior branch
       │
future segments
  -> PaliGemma.img
  -> 同一个 FactorizedSpaceTimeEncoder
  -> observed future event tokens
  -> 同一个 SharedSegmentMemoryUpdater(M_history, ...)
  -> M_future_post (train-only teacher)
                                 └─> Action posterior branch


推理期

history/current/state/language
  -> M_history
  -> predicted future event tokens
  -> {M_future_prior_k, p_k}
  -> Action

不读取 future frames
```

这里的 posterior/prior 指的是当前 semantic memory 在未来事件更新后的状态，不是旧 `pi0_mem_pf` 中的视觉 bottleneck latent。

### 4.1 Past/future 权重共享策略

推荐“当前 semantic 核心共享、任务角色分离”，而不是完全无差别共享。

共享部分：

- 当前 `PaliGemma.img` 图像特征提取；
- `FactorizedSpaceTimeEncoder` 中的 temporal/spatial attention core；
- `SharedSegmentMemoryUpdater`；
- memory width、token topology 和 recurrent update contract；
- 下游 `RawMemoryQueryResampler` / `ActionMemoryCrossAttention` 接口。

方向特有部分：

- history/future role embedding；
- history swap-relation head 与 future interaction-state head；
- history/future segment position embedding；
- future hypothesis query/token generator；
- 可选独立轻量输入/输出 projection 和 LayerNorm。

当前 history 路径可抽象为：

```text
history patches
  -> FactorizedSpaceTimeEncoder
  -> history relation/event tokens
  -> SharedSegmentMemoryUpdater
  -> M_history
```

推荐新增的 future posterior 路径为：

```text
future patches
  -> 同一个 FactorizedSpaceTimeEncoder core
  -> future interaction/event tokens + future_role
  -> 同一个 SharedSegmentMemoryUpdater
  -> M_future_post
```

其中 `FrozenSwapRelationClassifier` 的三分类输出只适用于历史 swap relation，不能直接复用于 future interaction。应把它拆成“共享 segment visual encoder + history relation head”，然后新增 future phase/contact/error/outcome heads。

当前 history relation 路径还使用 `argmax -> one_hot`，这会切断从 memory/action loss 到 visual relation encoder 的梯度。若要让共享 encoder 真正同时从 history 和 future 学习，需要选择以下一种策略：

1. 训练时使用 `relation_mode="probabilities"` 或可微 logits，推理时再选择 hard relation；
2. 保留 hard history relation，但通过显式 history relation auxiliary loss 更新共享 encoder；
3. 第一阶段冻结原 history encoder，仅训练 future heads/updater；第二阶段用很小学习率和 history replay supervision 解冻共享 encoder。

推荐先采用第 3 种保证旧 history 能力不退化，再尝试第 1 种端到端联合训练。

另一个形状约束是：当前 `FactorizedSpaceTimeEncoder` 和 `SharedSegmentMemoryUpdater` 的 history segment 长度为 10，且 temporal position embedding 参数形状与 `segment_size` 绑定。为了第一版能够真正调用同一个模块实例并共享参数，future segment 也应先设为 10 帧。若未来需要可变长度，再把 temporal position 改成可切片的最大长度参数或固定 sin/cos 表示。

更强的 multi-hypothesis prior 不直接凭空预测一个旧 PF 风格 latent，而是预测 `K` 组未来语义事件 token，然后让每组 token 继续通过当前共享 recurrent updater：

```text
M_history + current + state + language
  -> FutureEventPrior
  -> {predicted_future_events_k}_{k=1..K}
  -> SharedSegmentMemoryUpdater
  -> {M_future_prior_k}_{k=1..K}
```

这种设计最符合“future 也走与 history 相似的路径，并共用训练权重”的目标，同时保持当前 semantic memory 的模型语义和 action interface。

### 4.2 Future posterior teacher

真实 future frames 经当前 semantic 路径得到：

```text
future_event_tokens = FutureEventHead(
    SharedFactorizedSpaceTimeEncoder(future_segments)
)

M_future_post = SharedSegmentMemoryUpdater(
    M_history,
    future_event_tokens + future_role
)
```

建议把 current representation 加入 future posterior，使其更像：

```text
q(M_future | M_history, current, future_events)
```

这样 `M_future_post` 表示从当前 semantic memory 出发，在真实未来事件作用下更新后的状态，而不是记忆 future 图像的绝对像素。

`M_future_post` 只在训练时存在，推理绝不能读取它。

### 4.3 Multi-hypothesis future prior

prior 只使用推理时可用的信息：

```text
{future_event_tokens_k}, logits =
    FutureEventPrior(M_history, current_tokens, robot_state, language)

M_future_prior_k = SharedSegmentMemoryUpdater(
    M_history,
    future_event_tokens_k
)

p = softmax(logits)
```

第一版建议：

```text
K = 4
future_event_tokens_per_hypothesis = 10 × spatial/event tokens
```

不要让 future event head 直接输出高维原始视觉 token。应投影到当前 `semantic_memory_width=64` 的 compact event tokens，保持与 `SharedSegmentMemoryUpdater` 的输入契约一致。

### 4.4 Action expert 如何使用多个 future

可以分阶段实现。

第一版最简单方式：

```text
k_top = argmax(p)
ActionExpert(M_history, M_future_prior_k_top, current, state)
```

训练时用与真实 `M_future_post` 最匹配的 hypothesis 对应的 action branch。

为了最大程度保留当前 action-memory interface，不建议引入旧 PF 的双 GTCA 分支。可以把预测未来表达为当前 memory 上的 gated residual：

```text
DeltaM_prior_k = M_future_prior_k - M_history
M_action_prior_k = M_history + gate_future * DeltaM_prior_k

DeltaM_post = M_future_post - M_history
M_action_post = M_history + gate_future * DeltaM_post
```

然后继续走当前已经验证的路径：

```text
M_action
  -> HistoryRawMemoryQueryResampler / 后续重命名的 SemanticMemoryQueryResampler
  -> ActionMemoryCrossAttention
  -> action expert
```

`gate_future` 应小值或近零初始化，使新模型在初始化时接近现有 history-only policy。这样无需改变 `RawMemoryQueryResampler` 所要求的 `[B, 128, 64]` memory shape，也不需要替换当前 action expert。

后续可以尝试：

1. 对每个 hypothesis 分别生成 action，再做 risk-aware selection；
2. 根据 `p_k` 对 action/value 进行加权；
3. 让 value/risk head 评估每个 hypothesis 下的失败概率；
4. 选择在多个未来下都比较安全的动作。

不建议简单平均 latent：

```text
sum_k p_k * M_future_prior_k
```

因为它可能重新产生“平均未来”。若需要加权，优先在 value 或 action distribution 层面组合。

### 4.5 Action-conditioned future belief

未来状态通常依赖机器人将要执行的动作。更完整的模型应学习：

```text
p(M_future | M_history, current, state, candidate_action)
```

但这会形成 action 和 future belief 的耦合，工程复杂度较高。

建议分两阶段：

1. 第一版 prior 不读取 action，学习当前策略分布下的可能结果；
2. 第二版把 noisy action tokens 或候选 action chunk 输入 future prior，实现 action-conditioned belief/value prediction。

---

## 5. 训练数据构造

### 5.1 单条普通轨迹样本

对 episode 中的当前时刻 `t`，构造：

```text
D_t = (
    history_frames,
    current_frame,
    robot_state,
    language,
    target_action_chunk,
    future_frames,
    future_semantic_labels,
)
```

ShellGame 第一版建议：

```text
history = frame 0 ... 59
current = frame t
actions = action t ... t+15
future  = frame t+1, t+2, ..., t+10
```

这里优先使用 10 帧 future，是为了与当前三个 swap segment 的 `SWAP_SEGMENT_SIZE=10` 完全一致，从而真正复用同一个 `FactorizedSpaceTimeEncoder` 和 `SharedSegmentMemoryUpdater` 参数。action target 仍然可以保持 horizon 16。

即：

```text
num_future_frames = 10
future_frame_stride = 1
action_horizon = 16
current_frame_index = 60
```

模型输入布局：

```text
[frame 0, ..., frame 59, current_t, future_t+1, ..., future_t+10]
```

有效 `t` 范围：

```text
59 <= t <= episode_last_frame - 10
```

如果使用可变 future mask，也可以保留更靠近 episode 末尾的样本，但第一版建议直接过滤，避免 padding future 污染 teacher。

### 5.2 数据集代码改造

当前 `FixedPrefixCurrentVideoDataset` 不支持 future，需要增加类似：

```text
FixedPrefixCurrentFutureVideoDataset
```

采样索引：

```python
prefix_indices = [episode_start + i for i in range(60)]
current_index = source_index
future_indices = [
    current_index + (j + 1) * future_stride
    for j in range(num_future_frames)
]
target_indices = prefix_indices + [current_index] + future_indices
```

必须检查：

- future index 未越过 episode 边界；
- current frame 在 tensor 中的位置固定；
- `video_frame_valid_mask` 正确覆盖 history/current/future；
- transform 中 `num_frames` 使用总帧数；
- inference 数据仍可只提供 history + current；
- 模型不要再使用 `video[:, -1]` 选 current。

### 5.3 Future semantic labels

建议从 robosuite simulator state 自动生成以下标签：

```text
target_slot              左/中/右杯
future_phase             approach/align/descend/grasp/lift/recovery
eef_target_delta         dx, dy, dz
gripper_state            open/closing/closed
contact                  bool
grasp_success            bool
cup_displacement         dx, dy
error_type               none/xy_offset/collision/missed_grasp/slip
recovery_required        bool
recovery_success         bool
```

这些标签只作为训练辅助监督，不作为推理输入。

对每个 future timestep 可以保留 dense label，也可以只使用 horizon endpoint label。推荐：

- phase、relative pose 使用多 timestep；
- success、collision、recovery 使用整个 future window 的聚合标签。

### 5.4 普通轨迹切片是否足够

普通离线轨迹中，每个 `(history, current)` 通常只有一条真实 future。它足以实现第一版，并验证 future teacher 是否有价值。

但它不一定能可靠学习真正的多模态未来，因为模型可能把不同结果归因于当前状态细微差异，而不是认识到同一状态存在多种可能性。

因此建议第二阶段增加 branched rollout 数据。

### 5.5 同状态分叉 rollout

在 robosuite 的时刻 `t` 保存完整 simulator state，然后恢复相同 state 多次，生成不同分支：

```text
same simulator state at t
├── branch 1: nominal policy                       -> success
├── branch 2: small XY action noise                -> correction/success
├── branch 3: cup or EEF disturbance               -> collision/recovery
└── branch 4: gripper timing/orientation disturbance -> grasp failure
```

推荐保存：

```python
{
    "branch_group_id": "episode_14_t_87",
    "source_episode": 14,
    "source_frame": 87,
    "branch_id": 2,
    "intervention_type": "eef_pos_x_plus_20mm",
    "intervention_probability": 0.1,
    "history_frames": ...,
    "current_frame": ...,
    "actions": ...,
    "future_frames": ...,
    "future_labels": ...,
}
```

如果同一个样本同时加载多条 future branch，可以使用 set matching/Hungarian matching，将 `K` 个 hypothesis 与 `M` 个真实 future 分支匹配。

如果训练 loader 每次只加载一条 branch，也可以继续使用 soft assignment；相同 `branch_group_id` 的不同 outcome 会在多个 batch 中推动不同 hypothesis 专门化。

### 5.6 数据概率与校准

如果人为均衡 success/failure/recovery 数据，模型学习到的 `p_k` 反映的是训练采样分布，不再是真实发生概率。

建议：

1. 表示学习阶段可以平衡 outcome，先确保不同 hypothesis 都能形成；
2. 记录原始采样概率和 intervention probability；
3. 后续使用自然分布 validation set 或 importance weighting 校准 probability head；
4. 未校准前把 `p_k` 称为 mixture weight/confidence，不要直接解释为真实失败概率。

---

## 6. Multi-hypothesis 匹配与损失

### 6.1 Posterior teacher

真实 future 先得到 event tokens，再从 `M_history` 继续执行当前 recurrent update：

```text
E_future_post = FutureEventHead(
    SharedFactorizedSpaceTimeEncoder(future_frames)
)

M_future_post = SharedSegmentMemoryUpdater(
    M_history,
    E_future_post
)
```

prior 输出：

```text
E_future_prior: [B, K, segment_size, spatial_tokens, memory_width]
M_future_prior: [B, K, memory_tokens, memory_width]
logits: [B, K]
p = softmax(logits)
```

### 6.2 Hypothesis 距离

对每个 hypothesis：

```text
d_k = D(M_future_prior_k, stop_gradient(M_future_post))
```

推荐使用 normalized cosine distance，而不是未经归一化的 MSE：

```text
d_k = mean(
    1 - cosine(
        normalize(M_future_prior_k),
        normalize(M_future_post),
    )
)
```

原因：直接 MSE 加 norm regularization 容易出现 latent 全部趋近于零的退化解。这里应在当前 semantic future tokens/memory 上独立实现 cosine alignment 和 variance floor；不要从 `pi0_mem_pf_safe.py` 继承模型或训练逻辑。

### 6.3 Hard winner-takes-all

```text
k* = argmin_k d_k
L_match = d_k*
L_prob = -log p_k*
```

优点是模式容易分离，缺点是早期 winner 不稳定、dead hypothesis 风险较高。

### 6.4 Soft assignment

第一版更推荐：

```text
r_k = softmax(-d_k / temperature)
L_match = sum_k stop_gradient(r_k) * d_k
L_prob  = -sum_k stop_gradient(r_k) * log(p_k)
```

训练过程中逐渐降低 temperature：

```text
1.0 -> 0.5 -> 0.2 -> 0.1
```

后期接近 hard winner，但早期梯度更稳定。

### 6.5 Future semantic loss

每个 `M_future_prior_k` 和 `M_future_post` 都通过共享或轻量 prediction heads 输出：

```text
phase_logits
success_logits
error_type_logits
eef_delta
contact_logits
```

posterior 直接对真实标签监督；prior 按 responsibility `r_k` 加权监督，或只监督匹配 hypothesis。

语义 loss 可以同时参与匹配 cost：

```text
cost_k = latent_distance_k
       + beta_phase * phase_loss_k
       + beta_pose * pose_loss_k
       + beta_outcome * outcome_loss_k
```

这样不同 hypothesis 会形成更可解释的行为模式，而不是任意 latent 聚类。

### 6.6 Diversity 与 non-collapse

防止所有 hypotheses 相同：

```text
L_diversity = mean_{i != j} max(
    0,
    margin - distance(M_future_prior_i, M_future_prior_j),
)
```

防止 latent 各维度塌缩：

```text
std_d = std(flatten(M_future_prior), axis=batch_and_token)
L_noncollapse = mean(relu(variance_target - std_d)^2)
```

可以增加很小的 batch usage loss 防止 dead heads：

```text
usage = mean(r, axis=batch)
L_usage = KL(usage || uniform)
```

注意 `L_usage` 权重必须很小。真实 outcome 本来可能不均衡，不能强迫各 hypothesis 永远等概率使用。

### 6.7 Action loss

部署分支的 action loss 必须始终是主损失：

```text
L_action_prior
```

posterior teacher 也计算 action loss：

```text
L_action_post
```

推荐完整目标：

```text
L = lambda_prior * L_action_prior
  + lambda_post  * L_action_post
  + lambda_match * L_match
  + lambda_prob  * L_prob
  + lambda_sem   * L_future_semantic
  + lambda_div   * L_diversity
  + lambda_nc    * L_noncollapse
  + lambda_usage * L_usage
```

初始参考权重，不应直接视为最终值：

```text
lambda_prior = 1.0
lambda_post  = 0.5 ~ 1.0
lambda_match = 0.01 ~ 0.1
lambda_prob  = 0.01
lambda_sem   = 0.1
lambda_div   = 0.001 ~ 0.01
lambda_nc    = 0.001
lambda_usage = 0.0001 ~ 0.001
```

建议 alignment/match loss warmup，避免随机初始化的 future branch 在训练初期破坏 pretrained policy。

---

## 7. 推荐训练阶段

### Stage 0：数据与标签验证

目标：确认 future 数据没有跨 episode、current index 正确、标签可信。

检查：

- 随机可视化完整输入 clip；
- 检查 frame 60 是否始终为 current；
- 检查 future 与 action chunk 时间对齐；
- 检查 episode 末尾过滤；
- 统计 success/error/phase 分布；
- 检查固定 history 是否与当前 episode 一致。

### Stage 1：单一 future latent

先设置：

```text
K = 1
```

验证：

- `M_future_post` 是否包含 future semantic；
- posterior action branch 是否优于 history-only；
- prior 是否能逼近 posterior；
- inference-only prior 是否不退化。

如果 `K=1` 都没有收益，不应直接增加 `K`。

### Stage 2：K=4 soft assignment

启用：

- `K=4`；
- soft responsibility；
- future semantic heads；
- diversity/non-collapse；
- hypothesis usage 监控。

先使用现有轨迹切片。

### Stage 3：Branched rollout

从同一 simulator state 采集多种 intervention/outcome，验证 hypotheses 是否对应不同可解释未来。

### Stage 4：Risk-aware action selection

在 action generation 之外增加 value/risk head：

```text
V_k = P(success | M_history, M_future_prior_k, candidate_action)
```

根据 `p_k` 和 `V_k` 选择更鲁棒动作，而不是只使用 top-1 future。

---

## 8. 代码改造建议

建议优先复用当前 semantic 通用模块，不要把 ShellGame task-specific 逻辑写回 `src/openpi/models/`。

### 8.1 通用模型层

可以新增：

```text
src/openpi/models/multi_hypothesis_future.py
```

包含任务无关、且与当前 semantic memory tensor contract 一致的模块：

```text
MultiHypothesisFutureEventPrior
FutureHypothesisMatcher
HypothesisDiversityLoss
FutureBeliefActionInterface
```

这些模块应组合 `siglip_mem_semantic.FactorizedSpaceTimeEncoder`、`SharedSegmentMemoryUpdater` 和当前 action-memory interface。不要扩展或继承 `pi0_mem_pf.py`。

### 8.2 ShellGame task adapter

建议新增而不是直接覆盖当前 validated 模型：

```text
src/openpi/tasks/shellgame/pi0_mem_semantic_future.py
src/openpi/tasks/shellgame/future_semantics.py
```

职责：

- 固定 history/current/future layout；
- ShellGame phase/outcome/error vocabulary；
- semantic label heads；
- 与当前 recurrent memory 的连接；
- train/inference 分支选择；
- action loss 和 temporal mask。

### 8.3 数据层

建议在：

```text
src/openpi/training/mem/video_dataset.py
```

增加：

```text
FixedPrefixCurrentFutureVideoDataset
```

同时扩展：

```text
VideoFrameConfig
```

明确保存：

```text
fixed_prefix_frames
current_frame_index
num_future_frames
future_frame_stride
require_full_future
```

不建议改变现有 `FixedPrefixCurrentVideoDataset` 的行为，避免影响当前训练 recipe。

### 8.4 训练 recipe

建议新增独立 recipe：

```text
src/openpi/training/mem/recipes/shellgame_semantic_future.py
```

第一版参数：

```text
fixed_prefix_frames = 60
current_frame_index = 60
num_future_frames = 10
future_frame_stride = 1
future_segment_size = 10
future_event_width = 64
future_hypotheses = 1  # MVP 验证后再改为 4
```

### 8.5 梯度与冻结策略

必须明确处理当前 `stop_gradient(history_mem)`：

- 如果目标是联合训练 semantic memory，需要移除或配置化 stop-gradient；
- 如果希望先保护已验证的 history tracker，可以先冻结 tracker，仅训练 future prior/posterior/interface；
- 后期再以更小学习率解冻 shared temporal core；
- 建议在当前 `ActionMemoryCrossAttention` 或新增 future-belief residual 上使用小 gate、warmup 和分组学习率；不要复制旧 `pi0_mem_pf_safe.py` 的 freeze pattern，因为其参数路径不对应当前 semantic 模型。

推荐顺序：

```text
阶段 A：冻结 PaliGemma、history tracker、action backbone
        同时冻结共享 FactorizedSpaceTimeEncoder 和 SharedSegmentMemoryUpdater
        仅训练 future event heads、FutureEventPrior、matcher 和 belief gate

阶段 B：解冻 action-memory interface 和 action expert

阶段 C：低学习率解冻共享 FactorizedSpaceTimeEncoder/SharedSegmentMemoryUpdater
        同时加入 history relation/stage auxiliary loss 或 frozen-teacher distillation
        防止共享参数学习 future 后遗忘原 history tracking 能力
```

---

## 9. 评估与消融实验

必须至少包含：

1. History-only baseline。
2. History + single future semantic memory，`K=1`。
3. History + multi-hypothesis，`K=4`。
4. `K=4` + branched rollout。
5. Past/future 不共享 encoder。
6. Past/future 共享 core、分离 role/query/output。
7. 移除 semantic future labels。
8. 移除 diversity/non-collapse。
9. 推理时仅 prior，确认没有 future leakage。

关键指标：

```text
closed-loop task success rate
grasp success rate
disturbance recovery rate
cup collision rate
final target-slot accuracy
future phase accuracy
future outcome/error accuracy
prior action loss
posterior action loss
prior-posterior loss gap
M_future_prior-M_future_post alignment
hypothesis usage histogram
hypothesis pairwise distance
probability calibration / ECE
```

最重要的因果判断：

- posterior 明显优于 baseline：future 中确实包含对动作有用的信息；
- prior 接近 posterior：future 信息可以从推理期条件中预测；
- `K=4` 优于 `K=1`：多模态表示确实有价值；
- branched rollout 进一步提高：同状态多结局数据确实帮助模式分离；
- shared core 优于 non-shared 或参数更少而性能相当：权重共享有效。

---

## 10. 主要风险

### 10.1 Future leakage

风险：posterior 分支直接看到动作结果，训练 loss 很低，但 prior 推理性能不提升。

对策：

- `L_action_prior` 始终为主损失；
- 所有最终评估只使用 prior；
- 记录 prior/posterior gap；
- 对 future teacher 使用 bottleneck；
- 不允许 future frame 进入 current prefix 或 KV cache。

### 10.2 不可预测噪声

future encoder 可能编码光照、微小纹理、随机扰动等 prior 无法预测的信息。

对策：

- 减小 future event/memory token 容量；
- future posterior 条件化 current；
- 使用 semantic auxiliary supervision；
- 使用 feature normalization；
- future augmentations；
- 只对齐 task-relevant projected semantic memory。

### 10.3 Hypothesis collapse

风险：所有 `M_future_prior_k` 相同，或永远只有一个 head 被使用。

对策：

- soft assignment warmup；
- temperature annealing；
- diversity loss；
- variance floor；
- 小权重 usage balancing；
- branched rollout；
- 监控每个 head 的 assignment 和语义分布。

### 10.4 人为扰动分布失真

风险：为了制造失败而过度采样，导致概率头不校准。

对策：

- 保存 intervention probability；
- 分离 representation training 与 calibration；
- 在自然分布 validation 上重新校准。

### 10.5 Future 只重复 final slot

风险：ShellGame 的目标杯在 swap 后不再变化，future memory 仅重复现有 semantic memory。

对策：

- future labels 聚焦机器人交互与恢复；
- 评估 disturbance/recovery；
- 采集 contact、collision、misalignment、grasp failure 分支；
- 证明 future belief 对动作结果而不只是 target identity 有增益。

---

## 11. 推荐的最小可行版本（MVP）

第一版不要同时实现所有功能。建议范围：

```text
数据：
  fixed history frame 0..59
  explicit current frame t
  10 future frames, stride 1
  full-future samples only

模型：
  reuse current semantic FactorizedSpaceTimeEncoder core
  reuse current SharedSegmentMemoryUpdater
  K = 1 先验证
  M_future_post from real future semantic events + shared recurrent updater
  M_future_prior from predicted future events + shared recurrent updater
  prior/posterior action loss
  cosine alignment + variance floor

语义监督：
  phase
  eef_target_delta
  contact
  grasp_success

训练：
  freeze pretrained backbone initially
  small learned gates
  alignment warmup

评估：
  history-only vs K=1 future-aware
  prior-only closed-loop evaluation
```

只有在 `K=1` 证明 future teacher 有用之后，再升级：

```text
K=4 + soft assignment + diversity + branched rollouts
```

---

## 12. 最终研究假设

本方向需要验证的核心假设是：

```text
过去发生了什么
    -> 当前应该相信什么
    -> 未来可能出现哪些状态
    -> 哪个动作在这些未来下更可靠
```

相比纯 history memory，创新点不只是“多输入几张未来图像”，而是：

1. 训练期 future posterior teacher；
2. 推理期 causal future prior；
3. past/future shared temporal core；
4. semantic predictive state；
5. multi-hypothesis belief；
6. branched counterfactual rollout；
7. risk-aware action selection。

预期合理的开发顺序：

```text
future data contract
-> K=1 prior/posterior
-> semantic future supervision
-> K=4 multi-hypothesis
-> branched rollout
-> risk-aware control
```

不要从最终完整系统一步到位；每一阶段都应有清晰的 history-only 对照和 prior-only closed-loop 评估。
