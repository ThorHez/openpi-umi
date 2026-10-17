# RoboMME 四任务 Qwen + Recurrent Updater 训练方案

> 日期：2026-08-26  
> 任务：VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder、PickXtimes  
> 状态：训练设计稿；本文不表示四任务已经完成训练  
> 目标：用最小的任务集合覆盖语义记忆、遮挡后身份更新、有序事件记忆、执行进度计数和 memory-conditioned action

## 1. 核心结论

四个任务不使用“Qwen 直接看完整长视频并输出最终答案”的统一做法，而采用统一分工：

```text
任务 prompt
  -> GoalTokenEncoder（episode 开始时写入一次）

短视频窗口
  -> Qwen3-VL 离线教师：只识别局部语义事件
  -> 结构化 event / state delta / confidence
  -> task-specific symbolic state updater
  -> teacher memory trajectory

连续视觉 clip + previous student memory
  -> direct-visual recurrent updater
  -> next student memory
  -> task readout / action policy
```

Qwen 负责回答“当前窗口发生了什么”，recurrent updater 负责维护“到目前为止任务处于什么状态”。部署时默认只运行 direct-visual student MEM，不要求 Qwen 以 20 Hz 在线运行。

四任务的论文分工为：

| 任务 | 主要证明点 | 当前基础 | 本轮新增重点 |
|---|---|---|---|
| VideoUnmask | 从演示提取并保持被遮挡目标 | Qwen LoRA 和 semantic MEM 已验证 | 统一 goal token、teacher cache 和 student MEM 协议 |
| VideoUnmaskSwap | 遮挡后根据多次 swap 更新目标身份 | 数据已下载，尚未解压和实现 | 局部 swap 教师、身份状态轨迹、连续 recurrent update |
| VideoPlaceOrder | 按顺序写入目标并在 swap 后更新位置 | Qwen 可学习性 probe 已完成 | 将单次最终格点生成拆成 place/swap 事件记忆 |
| PickXtimes | 执行阶段计数并影响 action | event MEM 和 Pi action 已训练 | Qwen 改为局部事件教师，复用现有 updater/action，解决动作几何瓶颈 |

## 2. 训练边界

### 2.1 数据边界

- 所有 representation split 必须按 episode 划分，不能按 frame/window 随机划分。
- Qwen 的 VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder 输入只读取演示阶段，禁止读取执行动作帧。
- PickXtimes 读取执行阶段的因果前缀或局部窗口，但任何输入帧不得晚于监督事件。
- `simple_subgoal`、`is_video_demo`、仿真坐标和 actor identity 可以用于离线构造标签；学生推理输入中不能出现这些字段。
- 所有 Qwen pseudo-label 必须保存置信度、原始输出和合法性标记。低置信度样本可以丢弃，不能用 GT 静默替换后仍称为 Qwen pseudo-label。

### 2.2 统一切分

新四任务联合实验使用固定 split seed `260826`，每个任务独立分层：

```text
train: 70 episodes
dev:   15 episodes
test:  15 episodes
```

分层条件：

- VideoUnmask：difficulty、目标颜色、单/双目标。
- VideoUnmaskSwap：difficulty、swap 次数、目标颜色、单/双目标。
- VideoPlaceOrder：difficulty、ordinal、目标颜色、演示放置次数、是否发生 target swap。
- PickXtimes：目标颜色和 required count 1--5。

`test` 在模型选择完成前保持冻结。已有 80/20 实验作为历史 baseline 保留，不覆盖、不重新解释为新 split 的结果。

### 2.3 RoboMME 的两种时序运行模式

四任务不能假设 ShellGame 式固定 60 帧历史。统一模型必须支持两种运行模式：

#### A. Observe-then-act

适用任务：VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder。

```text
reset(goal)
  -> memory_0

可变长度观察 T_demo：
  clip_0 -> memory_1
  clip_1 -> memory_2
  ...
  clip_S -> memory_demo

demo/action boundary：
  latch semantic memory

执行阶段：
  action_t = policy(obs_t, memory_demo, execution_progress_t)
```

- `T_demo` 和 recurrent step 数 `S` 都是动态的，不使用固定 frame index 或固定 swap slice。
- observation 结束由数据中的 `is_video_demo` boundary 或在线环境的显式 phase signal确定；模型不能通过固定帧数猜 boundary。
- demo 结束时保留 semantic memory。执行阶段的普通画面不能覆盖被遮挡目标或 ordered target memory。
- 双目标任务仍允许在执行阶段更新轻量 `next_pick_rank` / `completed_pick` progress，但目标身份 memory 默认锁存。

#### B. Observe-act-update interleaving

适用任务：PickXtimes；该模式也将用于后续 BinFill、SwingXtimes、StopCube、PickHighlight。

```text
reset(goal)
while not done:
  new frames / proprio -> rolling clip buffer
  update_memory(previous_memory, clip, robot_state)
  action = policy(current_obs, updated_memory)
  carry updated_memory into next policy call
```

- memory 是跨 policy call 的持久状态，不在每次推理时重新读取从 episode 开始的完整历史。
- memory update 频率与 action frequency 解耦：例如 action 20 Hz、视觉 memory 5--10 Hz；两次 memory update 之间 action 读取最近一次锁存状态。
- event 去重使用时间戳、hysteresis 和 previous state，不能依赖“第几个固定窗口”。
- episode 长度由 required count、实际执行速度和是否失败恢复共同决定。

#### C. 统一的可变长度 batch

训练 batch 在 clip/evidence 层做 padding：

```text
evidence_steps: [B, S_max, N, D]
step_mask:      [B, S_max]
frame_mask:     [B, S_max, W]
timestamps:     [B, S_max, W]
```

- `step_mask=false` 的 padded step 必须严格 carry previous memory，不能产生更新或 loss。
- episode 最后的不足 12 帧 clip 使用 `frame_mask`，不能直接丢弃，因为最终事件可能位于尾部。
- 使用 length bucketing 降低不同 demo 长度造成的 padding 浪费。
- loss 由 event/change/hold mask 决定，不由绝对帧号决定。
- 训练加入 temporal stretch、不同窗口 offset、丢帧和 10/20 Hz 重采样，避免把状态转移绑定到固定执行速度。
- 对可放入显存的 episode 使用 full-unroll BPTT；很长的 VideoPlaceOrder 使用 gradient rematerialization 和分层 event segment。若必须截断 BPTT，仍从当前学生重新 rollout，不能从 replay buffer 读取 stale student memory。

## 3. 共享模型设计

### 3.1 Goal token

使用 `src/openpi/models/siglip_mem_semantic_goal.py`：

- OpenPI 现有 PaliGemma prompt encoder 处理完整任务 prompt；
- 冻结语言 embedding 主干；
- `GoalTokenEncoder` 将 prompt 压缩为 2 个、每个 64 维的 goal tokens；
- episode 开始时写入 128 个 memory slots 的前 2 个位置；
- 后续 recurrent update 不重复编码静态 prompt。

两个 token 不硬编码固定语义，但通过辅助任务使其分别容易读出：

```text
目标属性：颜色、目标对象类别
任务参数：ordinal、required count、单/双目标
```

必须进行 `wrong-goal` 和 `shuffled-goal` 消融，确认 memory 确实受 instruction 控制。

### 3.2 Qwen 局部事件教师

初始化使用：

```text
checkpoints/qwen3vl_videounmask_from_shellgame_replay25_B_260825/checkpoint-000300
```

统一输出外层协议：

```json
{
  "task": "TASK",
  "event": "EVENT",
  "arguments": {},
  "complete": true,
  "confidence": 0.0
}
```

训练 assistant token 时不要求模型生成 `confidence`；置信度由合法 JSON 概率、事件分类 margin 或多视图一致性离线计算。最终 cache 必须统一包含该字段。

Qwen 只学习局部事件，不直接学习全局累计 count、整条 ordered state 或精确 EEF 动作。

统一 teacher 时序合同为每条样本 12 个按时间排序的 keyframes。keyframes 可以从更长的因果区间稀疏采样，但最后一帧不得晚于该标签的 commit timestamp。四任务和 10% ShellGame retention replay 都必须满足 12 帧输入，混合数据构建时对帧数做强校验。

### 3.3 Student recurrent MEM

共享结构：

```text
冻结 SigLIP patch encoder
  -> 2x2 spatial pooling
  -> 12-frame DirectVisualSegmentEncoder
  -> 128 x 64 compact memory
  -> shared recurrent updater
  -> carry-biased soft gate, bias=-2
```

统一 student 时序合同为 `window=12, stride=6`。相邻更新重叠 6 帧，训练使用可变长度完整 episode replay、current-student full unroll 和跨 clip BPTT。每个 episode 随机采样初始 offset，避免只适应固定事件边界。固定的 12-frame 只是局部 encoder 的窗口宽度，不是 episode 长度约束；student 仍通过 previous memory 累积整个 episode。

Teacher 与 student 都使用 12 帧并不表示二者必须逐帧一一相同：teacher 可以在一个事件区间内选 12 个语义 keyframes，student 则按真实时间流连续处理 12 帧因果窗口。蒸馏监督只在相同 commit timestamp 对齐，禁止 teacher 使用学生尚未处理的未来帧。

滑动窗口/event trigger 的使用方式分两层：

1. Qwen 离线标注使用重叠窗口和 hysteresis/cluster，得到去重后的局部事件。
2. Student MEM 连续处理小 clip，并由 learned soft gate 决定写入强度。event label 用于 change/hold 监督，不作为推理时的硬二值 gate。

这是对原硬 event trigger 的必要修正：ShellGame replay 消融已经表明“只允许 GT event-overlap clip 更新”会显著降低最终状态准确率，因为跨边界和局部事件窗口也包含有用上下文。

### 3.4 Stateful runtime 接口

在四任务 closed-loop 前必须从训练 graph 中拆出真实在线接口：

```text
reset_memory(goal_tokens) -> memory_0
append_frames(frame, timestamp) -> optional complete clip
update_memory(memory_t, clip, frame_mask, robot_state) -> memory_t+1
latch_demo_memory(memory_t) -> memory_demo
read_memory(memory_t, query) -> task state / action condition
reset_episode() -> clear memory and buffers
```

验收要求：把同一条可变长度 episode 以“离线一次性 unroll”和“逐 clip 跨调用运行”两种方式处理，最终 memory/readout 必须在数值容差内一致。未通过该测试前，不能声称已经支持 RoboMME 的边观察边记忆部署。

### 3.5 共享损失

每个任务使用相同的 loss 组织方式，但 task readout 不同：

```text
L_total = 1.0 * L_final_state
        + 1.0 * L_change_state
        + 1.0 * L_hold_state
        + 0.5 * L_teacher_memory
        + 0.25 * L_event
        + 0.25 * L_task_aux
```

- `L_final_state`：episode 最终 memory 是否表达正确任务状态。
- `L_change_state`：事件完成后的第一个 committed state 是否正确。
- `L_hold_state`：无完整新事件或事件未完成时是否保持旧状态。
- `L_teacher_memory`：student memory 与 frozen teacher memory 的 cosine + MSE 对齐。
- `L_event`：局部事件类型和参数的辅助监督。
- `L_task_aux`：任务特定的 cell、count、ordinal、holding 或目标身份 readout。

三个 state loss 先在各自 mask 内求平均再相加，不能让数量更多的 hold clip 主导优化。

## 4. VideoUnmask 方案

### 4.1 目标状态

```text
goal = (target_color, target_rank)
memory = (target_container_identity, target_cell, covered, confidence)
```

单目标任务写一个目标 token；双目标任务按 instruction 顺序写两个目标 token，action 每完成一次选择后读取下一个目标。

### 4.2 Qwen 样本

复用现有三类样本：

| 样本 | 输出 |
|---|---|
| visible grounding | `target_visible(color, cell/container)` |
| visible-to-covered paired window | `target_covered(color, container)` |
| masked-only / evidence incomplete | `insufficient_evidence` |

当前 B checkpoint 作为教师 baseline，不首先重训单任务 Qwen。VideoUnmask 数据以 15% 比例进入四任务 Qwen 联合训练，主要用于防遗忘和校准统一输出协议。

### 4.3 MEM 训练

- 当前文件中的 demo prefix 恰好是连续 66 帧，但 loader/model 不写死 66；按实际 `is_video_demo` boundary 对可变长度前缀做连续 clip replay。
- visible 阶段产生 target identity；cover 完成后提交 `covered` 状态。
- 盖住后的静态帧只计算 hold loss，不能让 masked frame 覆盖 target memory。
- 使用目标 cell 分类、cell 内 offset 回归和 nearest-container 分类联合监督。
- 加入 wrong episode、masked-only、zero video 和 prompt-color shuffle。

### 4.4 验收

| 指标 | 进入下一阶段条件 |
|---|---:|
| Qwen paired nearest-container | >= 95% |
| Student MEM nearest-container | >= 85% |
| wrong/zero/shuffle 相对正常历史差距 | >= 20 percentage points |
| 单目标 oracle-planner closed-loop | >= 75% |
| 双目标顺序读取 | >= 70% |

## 5. VideoUnmaskSwap 方案

### 5.1 数据准备

当前数据位于：

```text
data/robomme_data_h5/record_dataset_VideoUnmaskSwap.h5.tar.xz
```

压缩文件约 1.6 GiB，解压后的 tar 内容约 21.6 GiB；磁盘空间足够。训练前解压到 `data/robomme_extracted/`，然后生成 episode audit：

- demo prefix 范围；
- visible、cover、每次 swap 的开始/结束；
- swap 次数 1--3；
- swap pair；
- 每次 swap 后目标容器 identity/cell；
- 单/双目标和 instruction 顺序。

### 5.2 目标状态

```text
goal = (target_color_1[, target_color_2])
memory = {
  target identity -> current container/cell,
  completed_swap_count,
  next_pick_rank,
  confidence
}
```

memory 不保存“第几个预定义容器”的仿真 actor ID，而保存可由视觉 token 和 camera-relative cell 读取的目标身份表示。Actor ID 只用于 teacher label 和离线评估。

### 5.3 Qwen 局部事件

```text
target_visible(color, source_cell)
target_covered(color, container_cell)
swap_complete(cell_a, cell_b)
no_completed_event
incomplete_event
```

每次 swap 构建：

- 完整 before/after 正样本；
- 只含 swap 前半段的 incomplete negative；
- 静态 no-event；
- 同一次 swap 的多重滑窗样本，训练后通过 hysteresis 聚为一个事件；
- 不同速度、窗口 offset 和帧率采样变体。

Qwen 只输出“哪两个容器完成交换”，不输出目标最终在哪。目标位置由 teacher/updater 根据 previous memory 和 swap pair 递归计算。

### 5.4 MEM 训练

- 从 VideoUnmask student 权重初始化 target-visible/cover encoder。
- 从 ShellGame gated replay step 1499 初始化通用 recurrent updater 和 soft gate。
- 先冻结 updater，训练 VideoUnmaskSwap task adapter/readout 500 steps。
- 再解冻 updater，用完整 demo unroll 训练 2000--3000 steps。
- `swap_times=1--3` 自然产生不同观察长度和不同 recurrent step 数；训练时按实际 boundary/mask 展开，不补成固定三次 swap 语义。
- 对每次 swap 后状态、partial-swap hold 和最终状态分别监督。
- 双目标 episode 同时跟踪两个身份，但 action readout 按 goal 顺序选择。

### 5.5 验收

| 指标 | 进入下一阶段条件 |
|---|---:|
| Qwen complete-swap event F1 | >= 85% |
| incomplete/no-event rejection | >= 90% |
| 每次 swap endpoint identity | >= 85% |
| final target identity/cell | >= 80% |
| full state sequence | >= 60% |
| normal 相对 shuffled-swap/zero-history | >= 20 percentage points |
| 单目标 oracle-planner closed-loop | >= 70% |

双目标 closed-loop 先作为扩展指标，不阻塞单目标主实验。

## 6. VideoPlaceOrder 方案

### 6.1 为什么修改现有协议

当前 Qwen probe 把约 900--1100 帧演示压缩到 12 帧，并直接生成最终 8x8 cell；历史结果为 28.33% exact、60% within-one-cell。后续审计发现该 probe 将 H5 的 `<y,x>` 按 `<x,y>` 解释，因此这个数只作为历史参考，不能作为最终可学习性结论。该协议同时承担了：

1. 找出 instruction 指定的第 N 次放置目标；
2. 在后续 swap 中持续跟踪该目标。

本轮不沿用该最终答案协议作为主训练目标，而拆成逐事件写入和更新。

### 6.2 目标状态

```text
goal = (target_color, queried_ordinal)
memory = [target_token_1, target_token_2, target_token_3, target_token_4]
target_token_i = (visual identity, current cell, valid, confidence)
```

ordinal slot 的身份不因 swap 改变；swap 只更新对应 token 的当前位置。

### 6.3 Qwen 局部事件

```text
place_complete(target_cell)
swap_complete(target_cell_a, target_cell_b)
no_completed_event
incomplete_event
```

- 每次放置完成附近采样完整窗口和半事件 negative。
- static/swap 段用重叠窗口标注局部 swap，而不是只均匀取 4 帧。
- 演示阶段操作的是示范方块；执行阶段的 `target_color` 不是每次演示放置事件的属性，不能写入局部 place 标签。
- ordinal 由 recurrent state 的 `written_count + 1` 确定；Qwen 不生成 queried ordinal 的最终 cell，只提供局部 place/swap 事实。
- `target_visual_token` 由 Qwen 定位到的 target crop 再经过冻结视觉 encoder 构造并写入 teacher cache，不要求 Qwen 在 JSON 中生成连续 token。
- 当前 `qwen3vl_videoplaceorder_from_multitask_B_260825/checkpoint-000400` 仅作为初始化/对照，不覆盖原 checkpoint。

### 6.4 MEM 训练

- easy/medium 先训练按顺序写入 2--4 个 target tokens。
- 第二阶段加入 hard target swap，冻结 place encoder，先训练 swap update。
- 第三阶段混合 easy/medium/hard，完整 demo replay 和 full-unroll BPTT。
- 演示放置数、静态段和 swap 段长度均可变；使用 event segment + step mask，不把约 900--1100 帧压缩成固定 12 帧作为 student 的唯一输入。
- final readout 使用 goal ordinal 查询对应 token，然后在最终画面的候选 target 中分类。
- 主要监督改为 candidate-target classification + target-token contrastive loss；8x8 cell 文本生成只保留为辅助指标。

### 6.5 验收

| 指标 | 进入下一阶段条件 |
|---|---:|
| place-complete event F1 | >= 90% |
| swap-complete event F1 | >= 85% |
| ordered slots 全部正确 | >= 70% |
| final candidate-target accuracy | >= 75% |
| final 8x8 exact / within-one-cell | >= 60% / 85% |
| hard-swap final candidate accuracy | >= 65% |
| truncated/local-only rejection | >= 90% |

动作评估先使用 oracle planner 验证目标选择；目标选择稳定后再接入学习型 Pi action，避免把 memory 错误和低层动作错误混在一起。

## 7. PickXtimes 方案

### 7.1 修改 Qwen 监督目标

现有 Qwen contract 要求模型从 causal prefix 同时输出事件和累计次数。本轮废弃“Qwen 自己累计”的主目标，改为：

```text
pick_complete(target_color)
place_complete(target_color, target_region)
press_complete(stop_button)
no_completed_event
incomplete_event
```

`completed_count`、`holding`、`ready_to_press` 和 `done` 由 recurrent updater 根据 previous state 和局部 event 计算。Qwen 的 local-only 样本不再要求输出 `insufficient_history`，因为局部事件本来就不需要完整历史。

### 7.2 目标状态

```text
goal = (target_color, required_count)
memory = (completed_count, holding, ready_to_press, done, last_event_id)
```

`last_event_id` 或 event embedding 用于去重，避免同一 place transition 被相邻滑窗重复计数。

### 7.3 MEM 训练

- 复用既有 PickXtimes train70/dev15/test15 split、event label 和 updater 权重。
- 用 Qwen local event cache 替换 clean subgoal proxy，先只测试 teacher 错误传播。
- student 仍输入图像、gripper state、previous memory 和 goal token，不输入 Qwen 文本。
- 使用 overlapping-window teacher label、cluster/hysteresis 去重；student replay 使用连续 clip 和 soft gate。
- 训练随机截取不同长度的因果前缀，并在 stateful rollout 中跨 action step carry memory，覆盖 count 1--5 和不同执行速度。
- change loss 监督 place 后 count 增加；hold loss 覆盖接近、抓取中、搬运中和不完整放置窗口。
- press 只在 `completed_count == required_count` 时为合法状态转移，加入提前 press hard negative。

### 7.4 Action 训练

现有 step 2999 的离线 position error 已降到 4.72 cm，但 5 条 corrected-Z 诊断仍为 0 次有效 first pick；主要瓶颈是 approach XY/Z 没有同时进入 grasp region，而不是 memory count。

因此 action 阶段：

1. 保留 RoboMME absolute EEF7 输出接口。
2. 新增 delta/velocity 或 waypoint residual 辅助头，重点监督 approach-and-descend。
3. 上采样 near-grasp 且 XY/Z 同时对齐的样本。
4. 冻结 MEM，先训练 action adapter；只有 predicted-memory 明显优于 zero/shuffled memory 后才联合微调。

### 7.5 验收

| 指标 | 进入下一阶段条件 |
|---|---:|
| Qwen local event F1 | >= 85% |
| dev final count | >= 80% |
| dev full state sequence | >= 45% |
| early/duplicate press rejection | >= 90% |
| normal 相对 zero/shuffled event history | >= 20 percentage points |
| 10 条 first-pick smoke | >= 5/10 后再扩大闭环评估 |

## 8. 四任务联合训练课程

### Stage 0：数据审计与标签

- 解压和审计 VideoUnmaskSwap。
- 为四任务生成统一 episode split manifest。
- 为每个任务生成 event/change/hold 状态表。
- 检查 future leakage、demo/execution boundary、重复事件和不完整窗口。

Go 条件：随机抽查视频与 label 对齐；train/dev/test episode 零重叠。

### Stage 1：多任务 Qwen LoRA

从 VideoUnmask B step 300 初始化。建议采样比例：

| 数据 | 比例 |
|---|---:|
| VideoUnmask | 15% |
| VideoUnmaskSwap | 25% |
| VideoPlaceOrder | 30% |
| PickXtimes | 20% |
| ShellGame retention replay | 10% |

ShellGame 只作为防遗忘控制，不进入 RoboMME 主结果。

训练建议：

```text
LoRA rank/alpha: 16/32
learning rate:   1e-5
pilot:           400 optimizer steps
full:            1000--1200 optimizer steps
eval interval:   100 steps
```

每个任务单独计算 validation 指标并早停，不能仅根据混合 token loss 选 checkpoint。若任何旧任务 event F1 相对初始化下降超过 5 个百分点，增加该任务 replay 比例。

### Stage 2：Teacher cache

对每个 episode 离线运行 frozen Qwen：

```text
episode_index
task_id
goal fields
clip_frame_indices
event_type / arguments
event_valid / confidence
teacher_state_before / after
teacher_memory
```

报告 pseudo-label coverage、合法率、事件 F1 和被过滤比例。clean metadata teacher 和 Qwen pseudo-label teacher 必须分开保存与命名。

### Stage 3：Task adapter warm-up

- 共享 visual clip encoder、recurrent core 和 128x64 memory format。
- 每个任务拥有 event adapter、state readout 和 loss mask。
- 冻结 recurrent core，分别训练 task adapter/readout 500--1000 steps。
- VideoUnmaskSwap 从 VideoUnmask adapter 初始化；PickXtimes 直接复用已有 adapter。

Go 条件：oracle event 输入下，每个任务 final state 达到各自验收阈值，先排除 state schema/updater 实现错误。

### Stage 4：Joint recurrent replay

- 从 ShellGame replay step 1499 初始化 shared soft-gated updater。
- task-balanced batch；先均匀采样四任务，再在任务内部均衡 difficulty/state transition。
- adapter/clip encoder LR `1e-4`，shared updater LR `1e-5`。
- 训练 2000 steps 后评估；必要时继续到 4000 steps，不以单纯增加步数替代数据和状态标签修正。
- dev checkpoint 选择使用四任务标准化指标的 harmonic mean，避免容易任务掩盖困难任务。

### Stage 5：Action

先做 memory-to-oracle-planner 高层选择：

- VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder：验证 memory 是否选择正确容器/target。
- PickXtimes：使用真实 Pi policy，单独解决 approach geometry。

高层选择通过后，冻结 Qwen/MEM，只训练 action memory adapter；最后才进行低学习率 joint fine-tune。

## 9. 必做对照与论文指标

每个任务至少报告：

1. base Pi / 无 memory；
2. current frame only；
3. Qwen full-video final answer；
4. recurrent MEM，clean event teacher；
5. recurrent MEM，Qwen pseudo-label teacher；
6. direct-visual student，推理无 Qwen；
7. zero memory、shuffled memory、wrong episode、wrong goal；
8. soft gate 与 hard event gate；
9. 单任务 Qwen 与多任务 replay Qwen。

论文主表优先放：

```text
VideoUnmask:       target selection / closed-loop success
VideoUnmaskSwap:   final identity + full state sequence
VideoPlaceOrder:   final candidate target + hard-swap accuracy
PickXtimes:        final count + full state sequence + closed-loop success
```

同时报告 teacher 与 student 之间的差距，避免把 Qwen perception error、memory transition error 和 action error混成一个成功率。

## 10. 实施顺序

按信息增益和复用程度排序：

1. VideoUnmaskSwap 数据解压、审计和局部事件 manifest。
2. VideoUnmaskSwap Qwen local swap pilot；通过后训练 recurrent updater。
3. VideoPlaceOrder 将 full-answer manifest 改为 place/swap local events。
4. PickXtimes 将 Qwen cumulative contract 改为 local event contract。
5. 运行四任务 Qwen replay LoRA。
6. 生成 Qwen teacher cache，训练共享 recurrent updater。
7. 先做三个目标选择任务的 oracle-planner 闭环。
8. 最后处理 PickXtimes action geometry 并做统一 action 评估。

最先执行 VideoUnmaskSwap 的原因是：它同时复用现有 VideoUnmask 的目标语义和 ShellGame 的 swap updater，是验证 Qwen + recurrent updater 能否迁移到 RoboMME 的最低风险、最高信息量实验。

## 11. 现有资产索引

- 12-frame teacher/student 时序合同：`src/openpi/tasks/robomme/four_task_temporal_contract.py`
- 12-frame Qwen 混合数据：`artifacts/robomme_four_task_qwen_unified_optimized_v2_12f_mixture_seed260826`
- 12-frame/stride-6 student episode 审计：`artifacts/robomme_four_task_pilot_12f_seed260826/sequence_audit.json`
- VideoUnmask Qwen 消融：`docs/qwen3vl_videounmask_lora_ablation_260825.md`
- VideoPlaceOrder Qwen probe：`docs/qwen3vl_videoplaceorder_lora_probe_260825.md`
- Qwen -> recurrent MEM：`docs/qwen_to_recurrent_mem_distillation_training_guide_260825.md`
- Recurrent replay/updater：`docs/recurrent_compact_visual_memory_replay_training_260826.md`
- Goal token：`src/openpi/models/siglip_mem_semantic_goal.py`
- VideoUnmask semantic MEM：`evaluation/robomme/videounmask_semantic_memory_v1_260823/README.md`
- PickXtimes step 2999：`evaluation/robomme/pickxtimes_pi_action_round16/STEP2999_TRAINING_AND_EVAL.md`
