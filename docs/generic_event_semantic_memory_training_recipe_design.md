# 通用滑动窗口 Event-Driven Semantic Memory 训练 Recipe 设计

本文说明如何将当前验证过的“短滑动窗口 + event trigger + recurrent memory”方案迁移到 ShellGame 以外的任务。目标是让新任务只实现自己的标签、事件语义和评估逻辑，而复用相同的视觉编码、因果触发、记忆更新、训练循环和 action 接口。

## 1. 设计目标

该方案适合具有以下特征的任务：

- 完整历史很长，但真正改变任务状态的事件比较稀疏；
- 策略需要长期记住物体身份、位置、所有权、完成状态或交互结果；
- 单个事件可以由一个较短的局部视频窗口识别；
- 推理时不能预先知道事件发生在哪一帧，也不能依赖固定阶段边界。

典型任务包括：

- 物体经过遮挡、移动或交换后的身份跟踪；
- 多步骤装配中零件状态和已完成步骤的记忆；
- 抽屉、柜门、按钮等环境状态的长期跟踪；
- 导航过程中地标、访问状态或任务进度的记忆；
- 人机协作中物体交接、指令变更和角色状态的记忆。

该结构不适合单独承担高频连续动力学建模。对于持续变化的速度、接触力或精细控制误差，仍应由当前视觉、机器人 state 和 action model 直接处理。

## 2. 总体结构

```text
最近 W 帧视觉窗口
    ↓
冻结或低学习率的视觉 patch embedding
    ↓
拓扑保持的空间 pooling
    ↓
短窗口时空编码器
    ├── event-presence logit：是否发生了完整、可写入的事件
    └── event-type logits / event embedding：事件发生了什么
            ↓
因果 trigger：threshold + rising edge / hysteresis
            ↓ 仅 trigger 时写入
persistent recurrent memory
            ↓
memory readout / memory resampler
            ↓
分类、状态预测或 action model
```

在线推理只需要保存：

```text
最近 W 帧 ring buffer
+ persistent memory
+ event_active（event gate 的迟滞状态）
```

不需要一直保存或重复处理完整历史视频。

## 3. 代码分层

### 3.1 可以直接复用的通用模块

- `src/openpi/models/siglip_mem_semantic.py`
  - patch-grid pooling；
  - factorized spatial/temporal Transformer；
  - recurrent memory updater；
  - memory read adapter。
- `src/openpi/models/siglip_mem_semantic_event.py`
  - 重叠滑动窗口构造；
  - event-presence 和可配置 event-type head；
  - rising-edge / hysteresis 因果 trigger；
  - event-triggered recurrent update；
  - 流式状态 `event_active`。
- `src/openpi/models/pi0_mem_semantic_action.py`
  - raw memory query resampler；
  - action-memory cross-attention。
- `src/openpi/models/pi0_mem_semantic_event_action.py`
  - event evidence、memory 更新和 action conditioning 的组合接口。
- `scripts/mem/train_semantic_memory.py`
  - 数据加载、episode split、FSDP、优化器、EMA、日志、eval 和 checkpoint 循环。
- `scripts/mem/train_event_semantic_memory.py`
  - event-memory 专用训练入口。

### 3.2 每个新任务必须实现的部分

建议每个任务新增以下文件：

```text
src/openpi/tasks/<task_name>/semantic_memory_event.py
src/openpi/tasks/<task_name>/pi0_mem_semantic_event_memory.py
src/openpi/training/mem/recipes/<task_name>_event_semantic_memory_pretrain.py
```

它们分别负责：

1. 定义任务事件及 memory 状态语义；
2. 把通用模型封装为该任务的数据接口；
3. 定义数据路径、标签读取、采样策略、loss 和 eval 指标。

通用模型中不应出现以下任务假设：

- 固定事件数量；
- 固定事件帧范围；
- 固定类别数；
- ShellGame 的杯子、交换关系或三阶段逻辑；
- 特定机器人的 action schema。

## 4. 数据契约

### 4.1 基础数据

每个 episode 至少需要：

- 按时间排序的视觉帧；
- `episode_index`；
- 可选机器人 state；
- 可选 action trajectory；
- 任务目标或 prompt；
- 用于构造 memory 监督的 episode metadata。

推荐按 episode 划分 train/validation，不能按单帧随机划分，否则同一段视频可能同时出现在训练集和验证集，造成严重泄漏。

### 4.2 Memory 标签

一个通用 episode 标签可以表示为：

```python
{
    "initial_state": ...,          # 可选：事件发生前的离散或连续状态
    "events": [
        {
            "start": int,
            "end": int,
            "event_type": ...,    # 离散类别或连续 embedding target
            "state_after": ...,   # 该事件完成后的 memory 状态
        },
        ...
    ],
    "final_state": ...,
}
```

不是所有任务都需要 `initial_state` 或离散 `event_type`。但至少要有一种监督能够回答：

- 当前窗口是否包含一个完整事件；
- memory 更新后应该表达什么状态。

### 4.3 Event 标签定义

`event=1` 不应简单表示“窗口里有运动”。它应表示：

> 该窗口包含足够完整、语义明确的状态转换，可以安全写入长期 memory。

推荐窗口标签：

| 窗口类型 | Event target | 作用 |
|---|---:|---|
| 完整覆盖一次状态转换 | 1 | 正样本 |
| 静止或无关运动 | 0 | 普通负样本 |
| 只包含事件开头或结尾 | 0 | partial hard negative |
| 同时跨越两个事件或两个阶段 | 0 | mixed/cross-boundary hard negative |
| 严重遮挡、无法判断结果 | 0 或 ignore | 取决于任务定义 |

跨边界窗口非常重要。若不显式训练这些负样本，重叠滑窗会在同一事件附近多次写入，或者把两个事件混成一个错误更新。

## 5. 无法精确标注事件边界时怎么办

真机数据通常没有精确的事件起止帧，可以使用以下弱监督方式：

### 5.1 根据状态变化自动生成

如果能获得物体位姿、接触、开关状态、抓取状态或机器人控制信号，可以从状态变化点生成粗边界，再对边界做时间扩张。

### 5.2 正样本使用安全内区间

不要把靠近粗边界的窗口直接作为正样本。只选择能确认完整包含事件的窗口作为正样本，把边界附近窗口作为 partial negative 或 ignore。

### 5.3 Temporal jitter

对正窗口起点做随机偏移，例如：

```text
nominal start ± 1～3 帧
```

偏移范围必须保证窗口仍包含完整事件。这样可以避免模型记住固定帧号。

### 5.4 多尺度窗口

如果事件持续时间变化明显，可以同时训练多个窗口长度，例如 6、10、16 帧，共享 event-type head 和 memory updater。不要仅用一个大窗口覆盖所有变化，否则会重新引入高计算量和混合事件问题。

## 6. 推荐采样策略

一个 batch 中不要只随机抽窗口。负样本远多于正样本，纯随机采样容易让 event head 学成始终输出 no-event。

推荐每个 episode 样本包含：

```text
完整事件正样本                 35%～45%
partial / cross-boundary 负样本 25%～35%
静止或无关负样本              25%～35%
```

ShellGame 当前使用的 8 个窗口只是一个示例：

```text
3 个完整事件正样本
2 个跨边界 hard negative
3 个静止/不完整负样本
```

其他任务不应照搬“3 个事件”，但可以保留相近的正负比例。

当 event 类别不平衡时，还要对 `event_type` 做类别平衡或 weighted sampling。

## 7. Loss 设计

通用形式为：

\[
L = \lambda_{init}L_{init}
  + \lambda_{event}L_{event}
  + \lambda_{type}L_{event-type}
  + \lambda_{memory}L_{memory-state}
  + \lambda_{action}L_{action}
\]

### 7.1 `L_event`

event-presence 的 binary cross entropy。建议分别计算正、负样本均值后再平均：

\[
L_{event}=\frac{1}{2}(L_{positive}+L_{negative})
\]

这样不会因为负窗口数量更多而压制正样本 recall。

### 7.2 `L_event-type`

事件类型监督。离散事件使用 cross entropy，连续事件可以使用 embedding regression、contrastive loss 或下一状态预测。

该 loss 回答“发生了什么”，event loss 只回答“什么时候写 memory”。两者不能互相替代。

### 7.3 `L_memory-state`

每次有效事件更新后，对 memory readout 添加状态监督。优先监督每个事件后的中间状态，而不应只监督最终状态，否则 credit assignment 会随事件数量快速变难。

### 7.4 `L_initial`

如果任务需要从初始观测建立身份或状态，单独训练 initial-state head。训练早期可用 GT initial state 初始化 recurrent memory，同时训练 initial head；后续必须增加 predicted-initial 的端到端验证。

### 7.5 当前推荐初始权重

```python
initial_loss_weight = 1.0
event_loss_weight = 0.5
event_type_loss_weight = 0.5
memory_state_loss_weight = 1.0
```

这些只是起点。若 false trigger 多，增加 event loss 或 hard-negative 比例；若 trigger 正确但 memory 状态错误，优先检查 event-type 和 memory-state loss，而不是继续提高 event 权重。

## 8. 推荐分阶段训练

### 阶段 A：事件感知和 memory 预训练

目标：证明视觉窗口能够识别事件，并且 recurrent memory 能累计出正确状态。

推荐：

- 冻结大视觉 backbone；
- 训练短窗口 encoder、event head 和 event-type head；
- 如果已有可靠 recurrent updater/readout，先冻结它们；
- 如果新任务没有可复用 updater，则先用 GT event/type 训练 updater 和 readout，再联合训练视觉 event encoder。

阶段 A 的验收不依赖 action 成功率，应先达到可靠的 memory 指标。

### 阶段 B：因果 rollout 训练或微调

训练采样通常知道哪些窗口是正样本，而部署时使用预测 trigger。为减小 train/inference mismatch，可以在阶段 B 中：

- 扫描完整窗口序列；
- 使用模型预测 trigger 更新 memory；
- 对最终状态和触发次数计算 E2E loss/metric；
- 对 early trigger、duplicate trigger 和 missed event 加惩罚。

如果阶段 A 的 causal validation 已经稳定，可以省略或缩短阶段 B。

### 阶段 C：接入 action model

首先冻结 memory 模块，仅训练：

```text
RawMemoryQueryResampler
+ ActionMemoryCrossAttention
+ Pi action expert
```

这样可以先证明 action 会读取正确 memory，而不会在 action loss 的噪声下破坏 event tracker。

### 阶段 D：低学习率联合微调

仅当以下条件满足时再解冻部分 memory：

- memory causal E2E 指标稳定；
- memory shuffle/zero ablation 能显著改变 action 目标；
- action 模型已经具备基本执行能力。

联合微调时，memory 学习率建议为 action expert 的 `0.05～0.2`，并持续监控 memory 指标，避免 action loss 破坏语义表示。

## 9. Teacher forcing 原则

Teacher forcing 可以用于训练稳定性，但不能成为推理依赖。

推荐逐步移除：

1. GT initial state + GT event window + GT event type；
2. GT initial state + GT event window + predicted event type；
3. GT initial state + predicted trigger + predicted event type；
4. predicted initial state + predicted trigger + predicted event type。

验证集至少必须报告第 4 种完全自动条件，或者清楚标注当前指标仍然条件于哪一项 GT。

## 10. Event trigger 推理

### 10.1 最简单的 rising edge

```python
event_high = event_logit > threshold
trigger = event_high and not previous_event_high
```

### 10.2 推荐的 hysteresis

真机分数容易在阈值附近抖动，建议使用不同的触发和重新武装阈值：

```text
inactive → active：event_probability > 0.7
active → inactive：event_probability < 0.3
```

实际代码使用 logit threshold，应先把概率阈值转换到 logit 空间。

### 10.3 未知事件数量

通用 recurrent updater不限制事件总数。任务代码不能因为训练数据通常有 N 个事件，就在推理时强制截断到 N 个。固定数量只可用于离线诊断或有明确任务协议的场景。

## 11. 新任务 Recipe 模板

```python
@dataclasses.dataclass(frozen=True)
class MyTaskEventMemoryConfig(TrainConfig):
    initial_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    event_type_loss_weight: float = 0.5
    memory_state_loss_weight: float = 1.0


def load_episode_label_table(config):
    # episode_index -> initial state, events, intermediate states, final state
    ...


def sample_training_windows(rng, labels):
    # complete positives + cross-boundary negatives + static negatives
    ...


def compute_objective(config, model, rng, observation, labels, *, train):
    if train:
        windows = sample_training_windows(rng, labels)
        outputs = model(..., windows=windows, causal_selection=False)
    else:
        windows = all_overlapping_windows(observation)
        outputs = model(..., windows=windows, causal_selection=True)

    event_loss = ...
    event_type_loss = ...
    memory_state_loss = ...
    loss = (
        config.event_loss_weight * event_loss
        + config.event_type_loss_weight * event_type_loss
        + config.memory_state_loss_weight * memory_state_loss
    )
    return loss, metrics
```

训练入口可以继续复用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python scripts/mem/train_event_semantic_memory.py \
  <task_name>_event_semantic_memory_pretrain \
  --exp-name=<experiment_name>
```

需要在 `train_event_semantic_memory.py` 的 recipe registry 中加入多个任务时，建议使用 Tyro 的 overridable config map，而不是在通用模型中加入任务判断。

## 12. 必须监控的指标

### Event 层

- complete-event recall；
- no-event rejection / precision；
- false triggers per episode 或每分钟；
- duplicate trigger rate；
- missed event rate；
- trigger timing error。

### Event-type 层

- event-type accuracy；
- 完整 event sequence accuracy；
- 各类别 confusion matrix。

### Memory 层

- 每次更新后的 state accuracy；
- final-memory accuracy；
- valid trigger count；
- fully automatic memory E2E accuracy。

### Action 层

- normal memory；
- shuffled memory；
- zero memory；
- wrong-episode memory；
- oracle memory。

如果 normal、shuffle 和 zero 的 action 几乎相同，说明 action model 没有真正读取 memory，即使 memory 分类准确率很高也不能认为整条链路成功。

## 13. 必做消融实验

1. 完整历史离线选择 vs 严格因果滑窗；
2. 重叠滑窗 vs 固定不重叠分块；
3. predicted trigger vs GT trigger；
4. predicted event type vs GT event type；
5. recurrent memory vs 只读最后窗口；
6. normal / shuffle / zero memory 对 action 的影响；
7. 窗口长度和 stride 敏感性；
8. 事件速度、停顿、反向运动和遮挡变化；
9. 未见 episode、未见物体和未见 prompt 的泛化。

## 14. 常见失败模式

### Event recall 高但重复触发多

原因通常是没有 partial/cross-boundary negatives，或缺少 hysteresis。优先修改采样和 trigger，不要直接扩大窗口。

### Event trigger 正确但 memory 状态错误

检查 event-type 标签、类别不平衡、soft probability 接口和 updater/readout 是否训练。event loss 再低也无法解决“发生了什么”的错误。

### 训练 memory 正确，因果 eval 失败

这是 teacher forcing 与 predicted trigger 的分布差异。加入 causal rollout 微调和错误触发状态训练。

### 验证准确率很高但新 episode 失败

首先检查是否按 episode split，以及是否存在固定背景、固定时间轴或物体外观泄漏。

### Action 不受 memory 影响

先冻结 memory，用 normal/shuffle/zero 控制变量检查 action interface。不要立即联合解冻所有模块，否则难以定位条件通路。

## 15. 新任务接入检查表

- [ ] 明确任务中什么状态需要长期记忆；
- [ ] 定义“完整可写事件”，而不只是运动；
- [ ] 准备完整、partial、cross-boundary 和静止窗口；
- [ ] 按 episode 划分 train/validation；
- [ ] 明确 event-type 和 state-after 标签来源；
- [ ] 决定是否需要 initial-state head；
- [ ] 先验证 GT event/type 下 recurrent updater；
- [ ] 再训练视觉 event/type 预测；
- [ ] 使用全重叠窗口进行严格 causal eval；
- [ ] 验证未知事件数量和不同事件持续时间；
- [ ] 完成 normal/shuffle/zero memory action 消融；
- [ ] 最后才进行低学习率端到端联合微调。

## 16. 当前 ShellGame 参考实现

- 训练入口：`scripts/mem/train_event_semantic_memory.py`
- 训练 recipe：`src/openpi/training/mem/recipes/shellgame_event_semantic_memory_pretrain.py`
- 任务模型：`src/openpi/tasks/shellgame/pi0_mem_semantic_event_memory.py`
- 任务适配器：`src/openpi/tasks/shellgame/semantic_memory_event.py`
- 通用视觉/event memory：`src/openpi/models/siglip_mem_semantic_event.py`
- 通用 action 接口：`src/openpi/models/pi0_mem_semantic_event_action.py`
- 120 episode 回归结果：`evaluation/shellgame/generic_causal_window6_recurrent_260822/README.md`

ShellGame 中的 6 帧窗口、三种 relation、三次事件和杯位状态都只是参考任务配置。迁移到其他任务时应保留因果 event-driven memory 原则，而重新定义窗口长度、事件标签、状态空间和评估指标。
