# RoboMME 环境特权信息 → recurrent MEM 蒸馏流程

日期：2026-08-27

## 1. 目标与约束

目标是在仿真训练阶段使用 RoboMME simulator 的特权状态，教会当前
trigger-free recurrent MEM 稳定维护任务进度、事件、计数和遮挡信息；测试时
student 只能接收因果 RGB、机器人状态和任务 prompt。

保持当前模型结构不变：

- 每个输入 chunk 为连续、非重叠的 12 帧；
- VisualWindowEncoder 编码当前 chunk；
- 同一组 shared recurrent updater 参数用于所有时间步；
- 128 个、宽度 64 的 memory tokens；
- 每个 chunk 一个可学习标量 soft write gate；
- 不增加任务专用 memory head；
- 不使用独立 event trigger 决定是否调用 updater。

推理更新仍为：

`M_t = M_{t-1} + g_t (M_candidate_t - M_{t-1})`, `g_t ∈ [0, 1]`。

特权信息只能进入 teacher、训练标签和 loss，不能拼接进 student 输入。

## 2. 知识分工

不要把 GroundSG 的全部信息都压入 recurrent MEM。按时间属性分成三类：

### 2.1 recurrent MEM 负责

- 当前任务阶段：observe / pick / place / swap / press / done；
- 当前目标实体；
- 已完成次数、要求次数、当前 order pointer；
- holding、covered、visible、ready_to_press、done；
- 已发生的 pick/place/swap/write/press 事件；
- 物体被遮挡前最后可见的稳定区域或世界位置；
- VideoPlaceOrder 已写入的顺序；
- 对多种可能状态的 belief（信息不足时不强制单点答案）。

### 2.2 当前图像 grounder 负责

- 当前目标的像素 point、bbox、mask；
- 当前 target/button 的像素坐标；
- 由相机运动造成的坐标更新。

MEM 只提供 query，例如“寻找 red cube、第二次 Pick”；grounder 使用当前图像
预测它现在的位置。只有目标被遮挡时，MEM 才保存 last-known world/region belief。

### 2.3 action backbone 负责

- 从目标身份、阶段、当前 grounding 和机器人状态生成动作块；
- 不要求 recurrent MEM 直接回归关节动作。

## 3. 统一特权状态协议

为四个任务使用同一个 teacher schema，不建立任务专用输出头：

```json
{
  "task_id": 3,
  "phase": "place",
  "target_entity": "red_cube",
  "secondary_entity": "none",
  "relation": "holding",
  "completed_count": 1,
  "required_count": 3,
  "order_pointer": 0,
  "holding": true,
  "covered": false,
  "ready_to_press": false,
  "done": false,
  "last_event": "pick_complete",
  "last_event_frame": 174,
  "target_visible": true,
  "target_region": "region_2",
  "target_world_xyz": [0.42, -0.13, 0.02],
  "target_uv": [117, 139],
  "target_bbox_xyxy": [102, 124, 132, 154]
}
```

可按任务屏蔽无意义字段，但字段编号、类别定义和 readout 保持统一。

最低限度可以从现有 H5 获取或解析：

- `simple_subgoal_online`：phase、entity、ordinal；
- `grounded_subgoal_online`：上述字段加 target uv；
- `is_subgoal_boundary`：阶段变化；
- joint/gripper state 和 action；
- front/wrist RGB。

重新采集 simulator rollout 时建议额外保存：

- object/target/button world pose；
- camera intrinsics/extrinsics；
- instance segmentation、bbox、depth、visibility；
- contact、grasp、release、button state；
- simulator task automaton state；
- success/failure reason。

## 4. 数据采集

### 4.1 每个原生环境 step 保存

```text
timestamp / step_id
front_rgb, wrist_rgb
joint_state, gripper_state
executed_action
task_prompt
privileged_state_json
simple_subgoal_online
grounded_subgoal_online
event_boundary
episode_success
```

始终先保存原生频率数据，再离线生成 12 帧 chunk，避免采集阶段就丢失事件时序。

### 4.2 轨迹来源

训练 mixture 建议逐步从 100% oracle 变为：

- 50% oracle GroundSG 成功轨迹；
- 30% 当前 MEM + action 的 on-policy 轨迹；
- 20% 扰动和恢复轨迹。

所有轨迹都由 simulator 自动补齐特权标签。on-policy 数据很重要，因为它覆盖
student 漏抓、提前松手、阶段迟滞之后的非专家状态。

### 4.3 划分与覆盖

- 按 episode seed 划分 train/dev/test，不能按 frame 随机划分；
- test seed 不参与 teacher、student、阈值或 gate calibration；
- PickXTimes 平衡 color × required_count；
- VideoUnmask 平衡 color × final region × observation length；
- Swap/Swing 平衡交换次数和交换方向；
- VideoPlaceOrder 平衡目标排列和 demonstration 时长。

第一版 PickXTimes pilot 建议至少 100 train / 20 dev episode，并覆盖 count 1–5。

## 5. 因果对齐与标签生成

第 `k` 个 chunk 只包含 `[12k, 12k+11]` 的图像。student 的 `M_k` 标签必须是
截至 chunk 末帧已经成立的 simulator 状态，禁止使用 chunk 末端之后的信息。

为每个 chunk 保存：

```text
frame_indices[12]
state_before
state_after_at_chunk_end
events_completed_inside_chunk
teacher_memory_after
soft_gate_target
geometry_target_at_chunk_end
visibility_mask
```

事件发生在 chunk 内时，从该 chunk 输出的 memory 开始使用新状态；相邻 chunk
可以获得较弱 gate 监督，但状态标签不能提前切换。

## 6. Privileged teacher

使用现有统一 GT teacher 思路，将结构化特权状态编码为 teacher memory trajectory：

```text
privileged initial task state
          ↓
unified symbolic teacher updater
          ↓
T_0, T_1, ..., T_K
          ↓
frozen unified state readout
```

先单独验证 teacher：

- 每字段准确率接近 100%；
- final state exact ≥ 99%；
- transition state exact ≥ 99%；
- 结构化 state → teacher memory → frozen readout 可逆；
- teacher 不读取 student 图像。

Teacher 的作用不是生成本来没有的 GT，而是把完整特权轨迹转换成平滑、富语义、
适合 student 模仿的 latent trajectory。

## 7. Soft gate 标签

Gate 继续由模型从上一 memory 和当前视觉 evidence 预测，不使用硬 event trigger。

对包含真实状态变化事件的 chunk，生成峰值；对相邻 chunk 做时间平滑：

`g*_k = g_floor + (g_peak - g_floor) max_e exp(-d(k,e)^2 / (2σ^2))`

第一版建议：

- `g_floor = 0.05`；
- `g_peak = 0.8`；
- `σ = 0.75 chunk`；
- 事件所在 chunk 的 `d=0`；
- 按事件重要性调节峰值：任务状态变化 1.0，纯连续运动 0.0；
- padding chunk 的 gate target 和 mask 均为 0。

事件重要性只包括会改变记忆状态的事件，例如 pick_complete、place_complete、
swap_complete、target_covered、target_revealed、write_complete、press_complete。
目标在当前图像中轻微移动不应触发大 memory write。

Gate loss 使用 soft BCE 或 Huber：

`L_gate = mean(mask * BCE(g_t, stop_gradient(g*_t)))`。

权重保持较小，初始建议 `λ_gate = 0.02–0.05`。同时加入排序约束比绝对数值更稳：

`L_rank = max(0, margin - mean(g_event) + mean(g_hold))`。

Gate 不是最终判断器；state、memory trajectory 和 persistence loss 仍决定更新内容。

## 8. Student 训练目标

Student 输入只有：

```text
12-frame visual tokens
task prompt / goal tokens
previous student memory
sequence padding mask
```

不得输入 event id、simulator state、oracle subgoal、bbox 或未来帧。

推荐总 loss：

```text
L = λ_mem   L_teacher_memory
  + λ_state L_unified_state
  + λ_trans L_transition_state
  + λ_final L_final_state
  + λ_keep  L_no_change_consistency
  + λ_gate  L_soft_gate
  + λ_rank  L_gate_ranking
  + λ_belief L_occluded_spatial_belief
```

第一版相对权重建议：

```text
λ_mem    = 1.0
λ_state  = 0.5
λ_trans  = 0.75
λ_final  = 1.0
λ_keep   = 0.05
λ_gate   = 0.03
λ_rank   = 0.02
λ_belief = 0.1（只在需要遮挡记忆的任务启用）
```

不要把所有 loss 先乘进同一个 state weight。memory trajectory、transition readout、
final readout 和 gate 必须分别归一化，防止大量 no-change chunk 淹没事件监督。

## 9. 三阶段训练

### Stage A：privileged teacher

1. 用完整 simulator state sequence 训练统一 teacher；
2. 验证 teacher transition/final 几乎无误；
3. 冻结 teacher updater 和 state readout；
4. 为所有 split 缓存 teacher memory trajectory。

### Stage B：视觉 recurrent MEM

1. 从当前已验证 Pick/four-task checkpoint 初始化；
2. 保留 shared updater 和 soft gate；
3. 使用 oracle + on-policy mixture；
4. 前 500–1000 步冻结或降低 goal initializer 学习率；
5. transition episode 和普通 episode 以平衡概率采样；
6. 每 25–50 步在完整 dev sequence 上评估；
7. 选择 checkpoint 时不能只看 final，使用 constrained/Pareto selection。

建议选择规则：先要求 final ≥ 93%，然后最大化 transition，再最大化 no-change 和
all-state；gate 指标只用于诊断，不作为最高优先级。

### Stage C：grounding 与 action

1. 冻结通过 Stage B 的 canonical MEM；
2. 训练跨任务共享的 goal-conditioned grounder；
3. 输入当前图像和 MEM target query，监督 bbox/point/region；
4. 冻结 canonical MEM，训练 action-facing residual adapter；
5. action adapter 使用 GroundSG/action teacher，但不能反向破坏 canonical memory；
6. 最后才进行极低学习率联合微调。

## 10. On-policy 迭代

每轮执行：

```text
当前 MEM + action rollout
        ↓
simulator 对每个实际到达状态自动标注
        ↓
加入 recovery / no-progress / wrong-phase 数据
        ↓
继续训练 MEM 和 action-facing adapter
```

至少比较：

- Round 0：oracle demonstration only；
- Round 1：加入 20–30% student rollout；
- Round 2：加入失败恢复和随机扰动。

## 11. 评估漏斗

### 11.1 Offline MEM

- field accuracy；
- all-state exact；
- transition-state exact；
- no-change exact；
- final exact；
- completed/final count；
- sequence exact；
- gate event/hold 均值和比值；
- gate AUPRC（对平滑 privileged event target）；
- no-change memory drift；
- 按任务、事件类型、count、episode length 分组。

第一版进入 action 的门槛建议：

- final state ≥ 93%；
- transition state ≥ 55%；
- no-change state ≥ 45%；
- final count ≥ 90%；
- event gate mean / hold gate mean ≥ 2；
- 三个 seed 趋势一致。

### 11.2 Grounding

- visible target point error；
- bbox IoU；
- target identity accuracy；
- occluded region/belief accuracy；
- 坐标在相机/物体运动后的刷新能力。

### 11.3 Action 分解

按顺序运行：

1. official action + oracle GroundSG；
2. official action + oracle memory state + predicted grounding；
3. official action + predicted memory + oracle grounding；
4. official action + predicted memory + predicted grounding；
5. no-memory control；
6. shuffled/zero-memory control。

该分解能判断失败来自 temporal memory、spatial grounding 还是 controller。

## 12. 必要消融

| 实验 | Privileged dense state | Soft gate supervision | On-policy | Grounding |
|---|---:|---:|---:|---:|
| A 当前 baseline | 否 | 否 | 否 | 否 |
| B dense state | 是 | 否 | 否 | 否 |
| C dense + soft gate | 是 | 是 | 否 | 否 |
| D + on-policy | 是 | 是 | 是 | 否 |
| E 完整系统 | 是 | 是 | 是 | 是 |

另做 hard gate 对照，但不作为主方法。预期 soft gate 在事件跨 chunk、视觉证据渐进和
标签边界抖动时优于 0/1 hard gate。

## 13. 防止特权泄漏

- 对 student batch 做 key allowlist，拒绝 `info/*`、oracle subgoal 和 simulator state；
- teacher cache 与 visual cache 分开存储；
- student dataloader 只按 row/state index 取 teacher target；
- 检查每个 target 的最大 frame id不超过 student chunk end；
- 对 RGB 随机置换后性能应明显下降，排除只靠 goal/count shortcut；
- 推理代码不依赖 H5 info 或 simulator object handle；
- 论文明确说明 privileged information 仅用于 simulation training。

## 14. 推荐的第一版最小实验

先只做 PickXTimes：

1. 使用现有 70 train / 15 dev split；
2. 从 H5 解析每个原始 step 的 phase/count/holding/ready/done 和 grounded uv；
3. 生成严格因果的 12 帧 dense state sequence；
4. 由真实 event frame 生成 `floor=0.05, peak=0.8, σ=0.75` soft gate；
5. 从当前 Pick best checkpoint 训练 300–500 步；
6. 同时跑 baseline、dense-state-only、dense+soft-gate 三组；
7. 三个 seed；
8. 达到 offline 门槛后，只跑 5 条 `predicted memory + oracle grounding` action smoke；
9. 通过后再训练 current-image grounder，并扩展到另外三个任务。

这个最小实验首先回答两个问题：特权 dense state 是否能提升 recurrent transition，以及
soft gate 是否能在不破坏 persistence/final 的情况下利用这些高密度监督。
