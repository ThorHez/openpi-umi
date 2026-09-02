# Event-Supervised Latent Memory Distillation：论文定位与实验备忘录

> 日期：2026-08-26  
> 用途：记录方法定位、创新边界、相关工作差异、关键实验和 ICRA 8 页论文写作思路。  
> 状态：研究路线备忘录，不代表当前实验已经支持所有主张。

## 1. 一句话方法定位

我们研究如何将大视觉语言模型对机器人视频的结构化事件理解，经由因果 symbolic recurrent teacher 转换为逐时刻连续状态，再蒸馏到一个固定容量、可在线递归更新、部署时无需大 VLM 的 direct-visual memory，并将该 memory 提供给 VLA action expert。

建议英文概括：

> We distill structured event understanding from a large vision-language teacher into a compact recurrent visual memory through causally supervised state transitions, enabling fixed-cost online memory for high-frequency VLA control without running the teacher at deployment.

建议暂定方法名：

- Event-Supervised Latent Memory Distillation（ESL-Mem）
- Causal Event-to-State Memory Distillation（CESM）
- Neuro-Symbolic Recurrent Memory Distillation（NSR-Mem）

当前更推荐 **Event-Supervised Latent Memory Distillation**，因为它突出监督来源和训练机制，避免把贡献误写成普通 recurrent memory。

## 2. 问题定义

标准 VLA 常基于当前观测预测 action chunk：

\[
\pi(a_{t:t+H}\mid o_t,g).
\]

在计数、遮挡追踪、有序位置回忆和边观察边行动的任务中，当前观测不足以决定动作，需要持久状态：

\[
M_t=U_\theta(M_{t-1},o_{t-k:t},g),
\qquad
\pi(a_{t:t+H}\mid o_t,g,M_t).
\]

直接端到端学习 \(M_t\) 面临三个问题：

1. action loss 对早期 memory update 的监督弱，容易形成不可读或不稳定的 latent state；
2. 密集滑窗会对同一事件重复写入，导致计数溢出或状态污染；
3. 在线运行 4B 级 VLM 生成 symbolic memory 成本高，难以适配高频控制。

本文目标是在保留 VLM 高层视觉事件知识的同时，让最终策略只运行轻量视觉 memory updater。

## 3. 方法总览

```text
训练阶段

短视频窗口
   │
   ├── Large VLM / Qwen teacher
   │       └── event、entity、region_a、region_b
   │
   ├── causal validation / deduplication / transition grammar
   │
   └── symbolic recurrent teacher
           previous symbolic state + goal + event
                         ↓
           clean/pseudo teacher state sequence
                         ↓ state encoder
           continuous teacher memory [M,D]

原始视觉窗口 + previous student memory
                         ↓
           direct-visual recurrent updater
                         ↓
           student memory [M,D]
                         ↓
      memory alignment + per-step state readout losses


部署阶段

当前短视频 + previous student memory
                  ↓
          next student memory
                  ↓
             VLA action expert

不运行 Qwen，不输入 GT event，不输入 simulator metadata。
```

## 4. 三层训练监督

### 4.1 事件监督

统一事件合同：

```json
{
  "event": "place_complete",
  "entity": "red_cube",
  "region_a": null,
  "region_b": null
}
```

事件检测器需要学习：

- 是否发生完成事件；
- 事件类别；
- 涉及的实体；
- 涉及的空间区域；
- `no_completed_event`、`incomplete_event` 与真正完成事件的边界。

若使用 soft teacher，可蒸馏：

\[
L_{\text{event-distill}}
=
D(z^{S}_{\text{gate}},z^{T}_{\text{gate}})
+
D(z^{S}_{\text{type}},z^{T}_{\text{type}}),
\]

其中 \(D\) 可使用 Huber 或 KL divergence。

### 4.2 Symbolic state transition 监督

symbolic teacher 将事件序列转换为状态序列：

\[
s_t=T(s_{t-1},e_t,g).
\]

PickXTimes 示例：

```text
state = {
  completed_count,
  remaining_count,
  holding,
  should_press,
  done
}
```

次数只在 `place_complete` 后增加；`pick_complete` 只改变 `holding`。非法状态转移被拒绝。

### 4.3 Continuous memory 监督

symbolic state 经 teacher encoder/updater转换成连续 teacher memory：

\[
M^T_t=U_T(M^T_{t-1},e_t,g).
\]

学生只看视觉：

\[
M^S_t=U_S(M^S_{t-1},o_{t-k:t},g).
\]

总损失建议为：

\[
L =
\lambda_m L_{\text{memory}}
+\lambda_s L_{\text{state}}
+\lambda_e L_{\text{event}}
+\lambda_a L_{\text{action}}.
\]

其中：

\[
L_{\text{memory}}
=
L_{\text{cosine}}(M^S_t,\operatorname{sg}(M^T_t))
+\alpha L_{\text{MSE}}(M^S_t,\operatorname{sg}(M^T_t)).
\]

`sg` 表示 stop-gradient。

`state readout` 从每一步 memory 解码任务状态，用于确保 memory 语义可读，并将逐时刻梯度传给 updater。readout 是训练与诊断接口，不必作为最终 action head。

## 5. 真正可能成立的创新点

### 5.1 Symbolic-to-latent temporal distillation

不是只蒸馏单帧分类或 Qwen hidden state，而是：

```text
VLM event semantics
→ causal symbolic transition
→ continuous recurrent teacher state
→ direct-visual recurrent student state
```

核心在于蒸馏“状态如何随事件变化”，而不是蒸馏某一次回答。

### 5.2 训练时使用大 VLM，部署时完全移除

目标是在保持 VLM 事件理解能力的同时，将在线推理路径压缩为：

```text
SigLIP short clip + previous fixed memory → next memory → action
```

论文应同时报告成功率、推理延迟、显存和随历史长度增长的计算量。

### 5.3 Event-triggered、因果、固定容量的状态写入

memory 只在可见完成事件发生时写入，并结合：

- incomplete/no-event rejection；
- hysteresis 或 rising-edge trigger；
- overlap deduplication；
- transition grammar；
- illegal transition rejection。

它适用于事件稀疏但控制频率高的任务，可避免同一事件在多个重叠窗口中重复计数。

### 5.4 Per-step state-readable memory

逐步 readout 监督使 memory 在接入 action 前就能检查：

- full state sequence；
- final state；
- count/location/order；
- holding/done；
- 下一事件。

潜在作用是改善普通 RMT 仅靠 action loss 训练时的不稳定性。该作用必须通过消融验证。

### 5.5 跨任务统一事件与 memory core

强版本应做到：

- Qwen 使用一个共享模型、统一 prompt 和统一四字段输出；
- memory updater 参数共享；
- 不为效果差的任务增加专用 Qwen head；
- goal/state 通过统一 token contract 表示；
- readout 可作为训练 probe，但核心 memory/action 接口保持共享。

如果最终仍依赖大量手写任务分支，方法的通用性主张需要降低。

## 6. 与主要相关工作的差异

### 6.1 RoboMME / MME-VLA

[RoboMME](https://arxiv.org/abs/2603.04639) 分别研究：

- Qwen/Gemini symbolic subgoal memory；
- perceptual history memory；
- TTT/RMT recurrent memory；
- Context、Modulator、Expert 三种接入方式。

其项目页指出 recurrent memory 整体较弱，可能受训练不稳定影响；symbolic memory 更擅长 counting 与短时视觉推理。[项目页](https://robomme.github.io/)

我们的潜在差异不是再提出一个 RMT 或 Memory-as-Expert，而是连接两条原本分开的路径：

```text
RoboMME：Symbolic memory 与 recurrent memory 分开比较
我们：Symbolic teacher → recurrent latent student → action
```

因此“接入 action expert”不能单独声称创新，主要贡献必须是蒸馏和训练闭环。

### 6.2 MEM

[MEM](https://arxiv.org/abs/2603.03596) 使用短期视频 memory 与长期语言 memory，支持分钟级真实任务。

我们的潜在优势：

- 在线不生成长文本；
- 固定 latent 容量；
- 更适合高频事件统计和精确状态跟踪；
- 连续 memory 可直接供 action cross-attention。

我们的不足：

- 开放世界表达能力弱于语言 memory；
- 尚未证明十分钟级时长；
- 数据规模和真实机器人覆盖远小于 MEM。

不应声称比 MEM 更通用，应定位为事件稀疏、状态明确、高频控制场景中的轻量方案。

### 6.3 MemoryVLA

[MemoryVLA](https://arxiv.org/abs/2508.19236) 使用 Perceptual-Cognitive Memory Bank，包含检索、门控融合和合并，并以 memory-conditioned diffusion expert 生成动作。

我们的区别：

- 不维护随时间增长的 memory bank；
- 不做历史检索，使用固定 recurrent belief；
- 通过 symbolic transition 提供明确状态监督；
- 计算成本原则上与历史长度无关。

代价是学生 memory 对未在状态合同中表达的细节可能保留不足。

### 6.4 μVLA 与 RB-VLA

[μVLA](https://arxiv.org/abs/2606.12497) 隔离研究少量 recurrent tokens 和 TBPTT，不使用辅助 loss；[RB-VLA](https://arxiv.org/abs/2602.20659) 使用紧凑、action-conditioned belief 和世界模型目标。

因此以下内容不是我们的独有贡献：

- fixed-size recurrent tokens；
- TBPTT；
- compact latent belief；
- memory-conditioned action。

我们的区别应集中在事件教师、因果状态转移和逐步语义蒸馏。当前方案相比 RB-VLA 的明显不足是还没有 action-conditioned dynamics 或 recovery modeling。

### 6.5 Explicit Language Memory 与 LaMem-VLA

[Explicit Language Memory](https://arxiv.org/abs/2608.04765) 递归更新可解释的语言历史；[LaMem-VLA](https://arxiv.org/abs/2607.07608) 将短期和长期检索结果重构为 VLA 原生 latent tokens。

我们的方案更强调 O(1) recurrent state 和事件转移，不强调可检索的长期经验库。对于需要回看细粒度历史证据的任务，我们可能不占优势。

## 7. 我们方案的主要优势假设

这些是需要实验验证的 hypothesis，而不是当前已成立事实。

1. **低延迟**：部署时移除 Qwen，适配 10–20 Hz action loop。
2. **固定容量**：memory 大小不随 episode 时长增长。
3. **状态可读**：能直接检查 count、location、order 和 done。
4. **因果写入**：减少密集滑窗重复触发和记忆覆盖。
5. **数据效率**：利用 VLM 与 symbolic teacher 提供比 action loss 更密集的监督。
6. **动作友好**：连续 memory token 比文本 subgoal 保留更高带宽接口。
7. **可变时长**：共享 updater 可处理先观察后行动和边观察边行动。
8. **模块化调试**：可分别评估 event、state transition、memory readout 和 action。

## 8. 当前不能声称的内容

1. 不能声称“首次提出 recurrent memory VLA”。
2. 不能把 readout、goal token、cross-attention 或 Memory-as-Expert 单独写成核心创新。
3. 不能声称新四任务 Qwen 已经蒸馏到 unified MEM；目前还没有完成该训练。
4. 不能把 ShellGame 的 metadata proxy 实验描述成逐 episode 真实 Qwen pseudo-label 蒸馏。
5. 不能声称当前方法已优于 RoboMME baseline；尚无相同协议下的 action success 对比。
6. 不能把 15-episode test 的 memory 指标当作闭环任务成功率。
7. 不能声称任务无关，除非消除或严格限制任务特定状态机和 head。

## 9. 当前实验事实

### 9.1 ShellGame

- 已验证 symbolic teacher memory → direct-visual recurrent student 的可行性；
- 5,000 episode 上的旧实验 final cup accuracy 为 94.6%；
- 但 teacher event 使用 simulator metadata 作为已验证 Qwen 输出的 clean proxy；
- 这证明了 clean-event 条件下的蒸馏机制，不包含真实 Qwen error propagation。

### 9.2 RoboMME 统一 Qwen

- 统一四字段 JSON valid rate：100%；
- 60 个锁定测试 episode、470 个规范窗口；
- overlap-dedup 后最终完整状态：30%；
- 最终答案：35%；
- VideoUnmask final state/final answer：80%/80%；
- PickXTimes、VideoPlaceOrder、VideoUnmaskSwap 仍受边界触发和 region 字段错误影响。

因此当前 RoboMME 结果不足以直接进入四任务 action 联训或支撑主要论文结论。

## 10. 论文必须完成的基线与消融

### 10.1 方法基线

- memoryless π model；
- fixed history / frame sampling；
- Qwen online symbolic memory；
- direct RMT，仅 action loss；
- RMT + per-step readout；
- oracle symbolic state；
- 完整 symbolic-to-latent distillation；
- RoboMME 官方最佳可复现 memory variant。

### 10.2 核心消融

| 消融 | 要回答的问题 |
|---|---|
| 无 memory distillation | teacher latent target 是否真正有用？ |
| 仅 final-state loss | per-step supervision 是否必要？ |
| 无 state readout | memory 是否变得不可读或训练不稳？ |
| 无 event gate | 密集更新是否导致重复计数？ |
| hard gate / soft gate / hysteresis | 哪种因果写入最稳定？ |
| oracle event / Qwen event / student event | 错误传播来自哪里？ |
| hard event / soft event distribution | soft teacher 是否更鲁棒？ |
| 无 goal token | prompt 与状态转移是否真正绑定？ |
| detached recurrence / TBPTT | 跨窗口梯度是否必要？ |
| task-specific / unified contract | 性能与通用性的权衡是什么？ |
| memory→Context / Modulator / Expert | action 接口是否影响收益？ |

### 10.3 泛化实验

- 未见过的 required count；
- 未见过的 prompt 表达；
- 更长 episode；
- 可变 observation 时长；
- 不同事件速度和帧率；
- 滑窗 offset 与 stride；
- 丢帧、遮挡、视觉干扰；
- 事件间隔变化；
- 边观察边记忆；
- train task 到 held-out task 的 memory transfer。

### 10.4 效率指标

- 在线 Qwen vs student MEM latency；
- 10 Hz 与 20 Hz 最大可持续频率；
- GPU memory；
- FLOPs/token count；
- memory storage 随 horizon 的增长；
- action policy 吞吐下降比例。

## 11. 建议验收门槛

在接 action 前：

- Qwen/student local event positive accuracy ≥80%；
- incomplete false-commit ≤15%；
- full-state sequence ≥70%；
- final state ≥70%；
- final answer ≥80%；
- dense sliding-window duplicate test 通过。

论文主实验建议达到：

- 至少 3 个具有不同记忆结构的 RoboMME 任务显著优于 direct RMT；
- action success 相比 memoryless 和 direct RMT 有统计显著提升；
- student 接近 online Qwen/oracle teacher，但在线成本显著降低；
- 至少一个跨长度或跨 prompt 泛化结果；
- 报告每任务不少于 50 个正式 rollout，条件允许应达到 100。

## 12. 推荐的四任务选择

当前适合围绕以下任务展示互补记忆需求：

1. **PickXTimes**：事件计数与 procedural state；
2. **VideoUnmask**：初始可见、随后遮挡的 object permanence；
3. **VideoUnmaskSwap**：目标身份与空间交换递归；
4. **VideoPlaceOrder**：有序空间历史与查询。

这四个任务分别覆盖 count、identity/location、relation transition 和 ordered memory，比只做一个任务更能支持“统一 memory core”主张。若受 ICRA 篇幅限制，正文突出 2–3 个代表任务，其余完整结果放附录。

## 13. ICRA 8 页正文建议结构

### 第 1 页：Introduction

- VLA 在 partial observability 和事件统计任务中的问题；
- online VLM symbolic memory 昂贵；
- direct recurrent memory 难训练；
- 一句话方法和三条贡献。

### 第 2 页：Related Work + Problem Setup

- symbolic/text memory；
- perceptual/memory-bank 方法；
- recurrent/belief memory；
- teacher-student VLA distillation；
- 明确我们的交叉位置。

### 第 3–4 页：Method

- 统一 causal event contract；
- symbolic recurrent teacher；
- continuous teacher/student memory；
- event-triggered update；
- readout、memory 和 action losses；
- train/inference 数据流。

### 第 5 页：Experimental Setup

- RoboMME 任务；
- episode split；
- Qwen teacher 与 student 配置；
- action policy；
- baselines 和指标。

### 第 6–7 页：Results

- action success 主表；
- state sequence 与 final state；
- teacher/student/latency 对比；
- 核心消融；
- 泛化或错误分析。

### 第 8 页：Discussion + Conclusion

- 哪类任务获益最大；
- symbolic prior 的优势与限制；
- Qwen error propagation；
- 开放世界与真实机器人扩展。

## 14. 建议主图

主图应突出训练与部署路径不对称：

```text
                    TRAIN ONLY
Video ──→ Qwen ──→ Event ──→ Symbolic Teacher ──→ Teacher Memory
  │                                                     │
  └────→ Visual Student Updater ──→ Student Memory ─────┘ distill
                                      │
                                      ▼
                                  Action Expert

                    DEPLOYMENT
Video clip + Previous Memory ──→ Student Updater ──→ Action
```

图中必须显式标注：

- Qwen 与 symbolic teacher 为 train-only；
- teacher event/state 不进入学生部署路径；
- memory 固定大小；
- updater 跨任意数量窗口共享；
- action 读取连续 memory，而不是 readout 分类结果。

## 15. 可能的论文标题

- **Event-Supervised Latent Memory Distillation for Vision-Language-Action Models**
- **Distilling Structured Visual Events into Compact Recurrent Memory for Robot Control**
- **From Events to Actions: Neuro-Symbolic Distillation of Recurrent Memory for VLAs**
- **Learning Causal Recurrent Memory for VLAs from Structured Vision-Language Teachers**
- **Fixed-Cost Long-Horizon Memory via Event-to-State Distillation for Robot Policies**

## 16. 推荐贡献表述

在证据完成后，可考虑使用：

1. We introduce an event-supervised latent memory distillation framework that transfers structured visual event knowledge from a large VLM, through causal symbolic state transitions, into a compact direct-visual recurrent memory.
2. We develop a fixed-capacity event-triggered updater with per-step state-readable supervision, supporting variable-length observation and online memory updates without running the VLM at deployment.
3. We demonstrate improved memory-state tracking, closed-loop manipulation success, and inference efficiency over online symbolic memory, perceptual history, and direct recurrent-memory baselines on representative RoboMME tasks.

在 action 与四任务实验完成前，应使用“we aim to / we investigate”，不要提前使用“we demonstrate”。

## 17. 最大风险与应对

### 风险 1：被认为只是模块拼接

应对：把主贡献落在 temporal distillation objective 和 teacher/student 因果训练协议，并用直接 RMT、readout-only、event-only 消融证明组合不是简单相加。

### 风险 2：symbolic teacher 过于任务特定

应对：统一事件合同、goal/state tokenization 和 memory core；限制手写逻辑只用于生成训练 target；测试 held-out prompt、count 或任务迁移。

### 风险 3：Qwen pseudo-label 不稳定

应对：GT teacher 预训练、soft teacher、置信度过滤、transition grammar、scheduled teacher/student mixing，并单独报告 oracle 与真实 Qwen 上界差距。

### 风险 4：memory 指标提高但 action 不提高

应对：action 直接读取 raw memory tokens，加入 frozen-memory、shuffled-memory、wrong-memory 和 oracle-memory 因果消融。

### 风险 5：只有 benchmark，没有真实价值

应对：强调 latency、固定容量、可变时长和 10–20 Hz 在线控制；条件允许增加至少一个真实机器人或跨帧率实验。

## 18. 最终 Go / No-Go 判断

### Go

满足以下条件时值得作为 ICRA 主线：

- 至少 3 个任务上 student final state 和 action success 明显优于 direct RMT；
- student 接近 online Qwen/oracle teacher；
- 延迟显著低于 online Qwen；
- ablation 证明 symbolic transition、per-step readout 和 event-triggered update 都有独立贡献；
- 统一 core 不依赖新增任务专用 Qwen head。

### No-Go / 降级为实验章节

若最终只有：

- ShellGame 单任务结果；
- GT metadata teacher；
- memory probe 提升但 action 无提升；
- 大量任务专用状态机/head；
- 与 direct RMT 差异不显著；

则不应把它作为完整新方法主线，可降级为 RoboMME benchmark adaptation、训练分析或工程性 memory distillation 实验。

## 19. 相关本地材料

- Qwen→MEM 蒸馏说明：`docs/qwen_to_recurrent_mem_distillation_training_guide_260825.md`
- 递归视觉 replay 说明：`docs/recurrent_compact_visual_memory_replay_training_260826.md`
- 四任务训练计划：`docs/robomme_four_task_qwen_recurrent_training_plan_260826.md`
- 四任务 Qwen 优化：`docs/robomme_four_task_qwen_unified_optimized_v2_260826.md`
- episode rollout 验证：`docs/robomme_four_task_qwen_recurrent_rollout_validation_260826.md`
- unified event contract：`src/openpi/tasks/robomme/qwen3vl_unified_event_contract.py`
- symbolic state machines：`src/openpi/tasks/robomme/four_task_state.py`
- generic recurrent updater：`src/openpi/models/siglip_mem_semantic.py`

## 20. 写作时反复检查的问题

1. 论文的主创新是否仍能在删除“Qwen、RMT、readout”这些名词后用一句机制性语言说清？
2. teacher 给学生的究竟是标签、事件、状态还是连续 memory？是否定义清楚？
3. 学生部署时有没有任何 teacher 信息泄漏？
4. 性能提升来自更好的 memory，还是更强的 action policy？
5. 方法是否真正支持可变长度和边观察边行动？
6. 是否报告了完整状态序列，而不仅是最终答案？
7. 是否同时报告 Qwen error propagation 和 oracle 上界？
8. 是否证明固定 memory 在更长 horizon 下不退化？
9. 是否对比在线成本，而不仅对比准确率？
10. 当前证据是否足以使用“general”“unified”“causal”“distilled”等关键词？
