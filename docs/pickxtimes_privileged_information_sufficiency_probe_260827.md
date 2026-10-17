# PickXTimes 环境特权信息充分性检验

日期：2026-08-27

## 结论

对于当前 RoboMME `PickXTimes` 成功示范，环境提供的特权信息**足够训练 recurrent MEM 学会完整符号状态轨迹**。问题不是缺少状态标签，而是当前视觉 student 尚未稳定地把 RGB evidence 转换成正确的事件更新，并且 direct teacher-latent delta 与 student/readout 的坐标空间不匹配。

最强证据是：不使用图像、不使用 Qwen、不使用 canonical GT-state encoder，只用一个从随机初始化训练的 recurrent updater；即使将每个事件都抹成相同的二值 boundary，仅保留“此刻发生了一个完成事件”和 goal required count，也能在 episode-disjoint test 上达到：

- state exact：100%；
- post-event state exact：100%；
- full-sequence exact：100%；
- final state exact：100%。

相反，完全移除事件输入、只提供 goal 后，full-sequence 与 final exact 都是 0%。因此事件流包含完成该任务所必需的信息，而且当前 simulator boundary 已经足够。

## 要回答的问题

此前 native oracle correction 只把 PickXTimes test all-state 从 35.81% 提升到 45.41%，transition 反而从 38.14% 降到 34.02%。这存在两种解释：

1. simulator 特权信息本身不够，无法决定 memory 应怎样更新；
2. 信息是够的，但视觉 updater、蒸馏目标或 latent 接口没有把信息学进去。

本实验去掉视觉感知和 teacher latent 对齐，只检查第一种解释是否成立。

## 实验设计

### 数据划分

沿用固定的 episode-disjoint PickXTimes 划分：

| split | episodes |
|---|---:|
| train | 70 |
| dev | 15 |
| locked test | 15 |

事件时刻可由原生 H5 `info/is_subgoal_boundary` 提供；此前全量审计已确认，每条 Pick episode 的 native boundary 数量与 canonical Pick/Place/Press 完成事件数完全一致。状态监督为每个事件前后的完整 Pick 符号状态：

```text
required_count
completed_count
holding
ready_to_press
done
```

模型还共享读取静态 task/color 字段，但以下结果表中的动态字段均单独核验。

### 隔离变量

模型是一个小型共享 recurrent memory updater：32 个 memory token、width 64、depth 1。模型从随机初始化训练 1500 steps；它只能按事件顺序递归更新，不能访问 GT-state encoder。

四组输入消融：

| 组别 | goal | 事件边界 | 事件类型 | entity/region |
|---|---:|---:|---:|---:|
| Rich | 是 | 是 | 是 | 是 |
| Event type | 是 | 是 | 是 | 否 |
| Binary boundary | 是 | 是 | 全部映射为同一种事件 | 否 |
| Goal only | 是 | 否 | 否 | 否 |

所有组均满足：

```text
pixels_used = false
qwen_used = false
teacher_state_encoder_used = false
recurrent_rollout_only = true
```

## 锁定测试集结果

| 输入信息 | 最佳 dev step | Field | State exact | Post-event | Sequence exact | Final exact |
|---|---:|---:|---:|---:|---:|---:|
| Rich | 200 | 100% | 100% | 100% | 100% | 100% |
| Event type | 200 | 100% | 100% | 100% | 100% | 100% |
| Binary boundary | 500 | 100% | 100% | 100% | 100% | 100% |
| Goal only | 100 | 82.92% | 13.39% | 8.25% | 0% | 0% |

Goal-only 的逐字段结果为：

| 字段 | Accuracy |
|---|---:|
| required_count | 100% |
| completed_count | 26.79% |
| holding | 63.39% |
| ready_to_press | 86.61% |
| done | 86.61% |

较高的 `done`/`ready_to_press` 单字段 accuracy 主要来自状态不均衡，不能表示模型学会了过程；严格的 sequence 和 final 指标均为 0%。

## 如何解释结果

### 1. 当前 Pick 特权标签并不缺

Rich 和 Event type 都在 200 steps 达到满分，说明事件语义与目标次数能够无歧义地确定状态更新。Binary boundary 也在 500 steps 达到满分，进一步说明在当前成功示范分布内，只需精确事件边界和 goal，递归网络就能学会计数与 phase 转换。

因此，当前视觉 MEM 的 transition 低不能归因为“simulator 没有提供足够标签”。

### 2. 事件内容有助于样本效率，但在当前 Pick 数据上不是必需的

PickXTimes 成功轨迹具有高度规则的状态机：

```text
pick -> place -> pick -> place -> ... -> press
```

所以 binary-boundary 模型可以结合 `required_count` 和事件序号推断 holding、completed count、ready-to-press 与 done。显式 event type 将收敛从约 500 steps 加速到 200 steps。

### 3. 这定位到的是学习接口问题

此前 native oracle + direct teacher delta 实验中，correction prediction RMS 约为 `0.0216`，而 target RMS 约为 `0.7997`。它只实现了很小一部分目标更新；与此同时，full-latent delta 会改变后续整个 recurrent trajectory，而原 readout 未必能解码新 latent。

综合两个实验，当前瓶颈更可能是：

- 从 RGB/chunk 判断准确 event timing 和 event semantics；
- full teacher latent 与 student latent 的 basis mismatch；
- transition、trajectory、final readout 与 hold loss 之间的梯度竞争；
- correction 改写 memory 后，后续 readout/rollout 的兼容性。

## 不能外推的部分

本实验只证明 **PickXTimes 成功、canonical 执行轨迹** 的信息充分性，不证明：

- RGB 一定能够稳定恢复这些事件；
- 只有 binary boundary 就能处理抓取失败、重试、越序动作或额外接触；
- binary boundary 对 VideoUnmaskSwap、VideoPlaceOrder 等置换/空间关系任务也足够；
- 当前模型已经可以安全接入 action。

尤其是 VideoUnmaskSwap 与 VideoPlaceOrder，需要知道参与交换/放置的实体和区域；此前 event-only recurrent rollout 在锁定 test 上分别只有 20% 和 80% final exact。它们仍可能缺少更细的物理谓词、遮挡身份或异常交互标签，需要单独做相同的信息充分性消融。

## 对下一步的直接含义

PickXTimes 不应继续通过增加更多 simulator 字段来碰运气。更有信息量的下一步是保持 native event mask，先训练显式、可解码的语义更新：

```text
event chunk -> event/state logits -> recurrent symbolic state -> memory tokens
```

先在 oracle boundary 下验证 visual features 能否预测 event type 和 post-event state，再把 oracle boundary 蒸馏到视觉 soft gate。若该版本仍失败，才说明 RGB evidence 或视觉表示本身不足；现在的证据不支持“环境特权信息不足”这一判断。

## 可复现实验产物

- 入口：`scripts/mem/probe_pickxtimes_privileged_information_sufficiency.py`
- Rich：`checkpoints/pickxtimes_privileged_info_probe_rich_seed260832_260827/result.json`
- Event type：`checkpoints/pickxtimes_privileged_info_probe_event_type_seed260832_260827/result.json`
- Binary boundary：`checkpoints/pickxtimes_privileged_info_probe_binary_boundary_seed260832_260827/result.json`
- Goal only：`checkpoints/pickxtimes_privileged_info_probe_goal_only_seed260832_260827/result.json`

本次为单 seed 的机制诊断；由于三种含事件输入的设置在严格指标上均已达到 ceiling，它足以否定“Pick 特权信息不够用”这一解释，但正式论文表格仍建议补 3 seeds。
