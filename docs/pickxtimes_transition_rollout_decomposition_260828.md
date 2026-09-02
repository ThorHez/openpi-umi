# PickXTimes 状态转移与 free-rollout 误差解耦实验（2026-08-28）

## 研究问题

此前 opaque recurrent MEM 在 locked test 上长期停留在约 70%–80% transition、40%–45% hold，full-sequence exact 始终为 0。本实验不再调整 gate 或 loss 权重，而是依次检查：

1. 给定 GT 前一状态，固定 12 帧 RGB/proprio 是否足以预测下一状态；
2. 给定 GT 前一状态和 GT event，结构化 updater 是否能学会状态转移；
3. 使用 learned 前一状态和 GT event 自由滚动时，是否出现递归漂移；
4. 使用 learned 前一状态和 RGB/proprio 自由滚动时，完整系统是否出现 exposure gap；
5. 上述结果与当前 opaque latent MEM 的差距有多大。

## 数据与状态

沿用 episode-disjoint PickXTimes 划分：

| split | episodes | fixed chunks | transition | no-change |
|---|---:|---:|---:|---:|
| train | 70 | 2928 | 438 | 2490 |
| dev | 15 | 672 | 99 | 573 |
| locked test | 15 | 672 | 97 | 575 |

输入 observation 与原 MEM 相同：每步是固定、不重叠的 12 帧 chunk，RGB 使用缓存的 4×4 SigLIP patch tokens，proprio 使用 gripper state/command 与 EEF-Z 等六维时序特征。

为了避免 teacher latent basis 干扰，只预测 Pick 的四个动态状态字段：

```text
completed_count, holding, ready_to_press, done
```

GT event 由相邻 GT 状态确定为：

```text
no_change, pick_complete, place_complete, press_complete
```

推导出的 event boundary 与原生 simulator `state_change_mask` 在 train/dev/test 上完全一致。

## 模型与训练方式

使用一个显式 structured transition updater：

```text
previous structured state + required_count + evidence -> next structured state
```

Evidence 分为两种：

- `gt_event`：GT event type；
- `rgb_proprio`：固定 12 帧 RGB+proprio。

训练阶段始终输入 GT previous state，transition/no-change 各占 batch 的 50%。checkpoint 按 dev teacher-forced `min(transition, no-change)` 选择。评估时分别运行：

- teacher-forced：每步输入 GT previous state；
- free-rollout：只在 episode 开头输入初始状态，之后递归使用模型上一步预测。

该模型不使用 teacher latent、不使用 write gate，也不接收推理时 GT state。

## Locked-test 结果

两个 seed 得到相同的关键结果：

| Evidence | Previous state | State | Transition | Hold | Sequence | Final | TF→free gap |
|---|---|---:|---:|---:|---:|---:|---:|
| GT event | GT | 100% | 100% | 100% | 100% | 100% | — |
| GT event | Learned rollout | 100% | 100% | 100% | 100% | 100% | 0 |
| RGB+proprio | GT | 99.70% | 100% | 99.65% | 86.67% | 100% | — |
| RGB+proprio | Learned rollout | 99.70% | 100% | 99.65% | 86.67% | 100% | 0 |

RGB+proprio 每个 seed 只在 575 个 test no-change chunk 中错 2 个；错误是孤立的，并在后续窗口自动恢复，因此 teacher-forced 与 free-rollout 指标完全相同。15 条 episode 中 13 条全序列完全正确，所有 final state 都正确。

逐 seed：

| Mode | seed | best step | TF state/T/H/seq/final | Free state/T/H/seq/final |
|---|---:|---:|---|---|
| GT event | 260864 | 200 | 100/100/100/100/100% | 100/100/100/100/100% |
| GT event | 260865 | 200 | 100/100/100/100/100% | 100/100/100/100/100% |
| RGB+proprio | 260866 | 800 | 99.70/100/99.65/86.67/100% | 99.70/100/99.65/86.67/100% |
| RGB+proprio | 260867 | 800 | 99.70/100/99.65/86.67/100% | 99.70/100/99.65/86.67/100% |

## 与当前 opaque latent MEM 的公平对比

对当前最佳 checkpoint `pickxtimes_joint_gate_margin_seed260862_from260854_260828` 重新只计算相同四个动态字段：

| 模型 | State | Transition | Hold | Sequence | Final |
|---|---:|---:|---:|---:|---:|
| 当前 opaque latent MEM | 50.07% | 79.38% | 45.25% | 0% | 66.67% |
| Structured RGB+proprio updater | 99.70% | 100% | 99.65% | 86.67% | 100% |

原 MEM 的全字段 transition 为 78.35%，动态字段 transition 为 79.38%；静态字段并不是主要误差来源。

## 结论

### 1. 当前窗口信息足够

`GT previous state + RGB/proprio` 达到 100% transition 和 99.65% hold，证明固定 12 帧输入中已经包含可靠的 Pick/Place/Press/no-change 判别信息。继续增加 simulator 字段、调整窗口或只增强 gate 不是当前的主矛盾。

### 2. 状态转移算子本身可学

`GT previous state + GT event` 的 teacher-forced 和 free-rollout 都为 100%，说明计数、holding 切换、ready 和 done 逻辑没有容量或数据瓶颈。

### 3. Exposure bias 不是主因

structured RGB/proprio updater 的 teacher-forced 与 free-rollout gap 为 0。即使出现两个孤立 hold 错误，下一窗口也能恢复。因此原 MEM 的 sequence=0 不能主要归因于“训练看 GT、推理看自己”的 exposure mismatch。

### 4. 核心瓶颈是 latent 表示与更新接口

同样的数据在显式状态接口下几乎达到上界，而 opaque latent MEM 只有 50.07% dynamic state exact。主要问题已经定位为：

- teacher latent 与 student latent 的 basis/geometry 对齐并非任务所必需，却引入了困难目标；
- attention candidate update、scalar gate、teacher-memory trajectory 和 shared readout 需要共同形成离散状态机，优化条件很差；
- no-change identity 与 event state transition 没有在表示层被显式约束；
- gate 只控制写入幅度，无法保证写入内容对应正确语义状态。

因此继续进行 gate-loss sweep 或训练步数扩展预计只能移动原有 Pareto 曲线，无法解释约 50 个百分点的结构化接口差距。

## 对后续模型设计的直接建议

本 probe 使用 Pick 四字段是为了诊断，不建议最终模型增加 Pick 专用 head。更通用的修改应是：

1. 保留统一的跨任务 semantic state schema 与 shared readout；
2. 将上一时刻 shared semantic state distribution 重新编码，作为下一次 recurrent updater 的显式输入；
3. 用 observation 预测 semantic delta/next state，latent residual memory 只保存 schema 以外的信息；
4. action expert 读取 semantic state tokens + optional residual latent，而不是只能依赖未经约束的 opaque memory；
5. 先在四任务上验证 structured free-rollout，再讨论 gate 或 action 联训。

这相当于把当前架构从：

```text
opaque latent -> readout only for supervision
```

改为：

```text
shared semantic state -> recurrent feedback -> next shared semantic state
                         + optional residual latent
```

该方向直接针对本实验定位出的主瓶颈，同时仍可保持一个统一模型、统一 readout，不需要为每个任务增加定制 head。

## 产物

- 实验入口：`scripts/mem/probe_pickxtimes_transition_rollout_decomposition.py`
- GT event seed 260864：`checkpoints/pickxtimes_transition_decomp_gt_event_seed260864_260828/`
- GT event seed 260865：`checkpoints/pickxtimes_transition_decomp_gt_event_seed260865_260828/`
- RGB+proprio seed 260866：`checkpoints/pickxtimes_transition_decomp_rgb_proprio_seed260866_260828/`
- RGB+proprio seed 260867：`checkpoints/pickxtimes_transition_decomp_rgb_proprio_seed260867_260828/`
- 原 MEM 动态字段复评：`checkpoints/pickxtimes_joint_gate_margin_seed260862_from260854_260828/test_visual_dependence.json`
