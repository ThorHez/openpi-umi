# ICRA 2027 论文初步框架：事件监督的固定容量递归记忆蒸馏

> 日期：2026-08-27  
> 目标会议：IEEE ICRA 2027  
> 文档性质：论文 v0 框架、证据盘点和写作边界；不是最终实验结论  
> 推荐暂定方法名：**Event-Supervised Latent Memory Distillation (ESL-Mem)**

## 0. 先给结论：这篇论文现在应该讲什么

推荐的核心故事是：

> Long-horizon manipulation does not merely need more visual history; it needs a compact state that changes only when new evidence is worth committing. We use structured event/state supervision to construct a continuous teacher memory, distill its causal state trajectory into a direct-visual recurrent student, and condition a robot action policy on the resulting fixed-capacity memory. The structured teacher is training-only; deployment uses short visual chunks, the previous memory, and a learned soft write gate.

对应的中文主张：

> 长时机器人操作真正缺少的不是更长的图像堆叠，而是一个能够选择性写入、持续保持并可被动作策略直接读取的紧凑状态。我们用结构化事件与状态转移构造连续 teacher memory，再把整条因果状态轨迹蒸馏到只看原始短视频的递归学生中；部署时不运行大 VLM、不输入事件标签，历史被压缩在固定容量 latent memory 中，并通过 soft write gate 在线更新。

这条主线比以下两个旧版本更接近当前实验事实：

1. 不再把“online Qwen symbolic memory”写成最终部署方法。四任务 Qwen 已能稳定输出合法结构，但 recurrent rollout 的 final state 目前只有 30%，误差累积尚不适合做主部署链路。
2. 不再把“hard event trigger”写成唯一写入机制。固定非重叠短片段配合 carry-biased soft gate 的结果表明，模型需要在完整事件以外逐步积累局部证据；硬门控会丢失这些信息。

当前最合适的论文定位不是“首次提出 recurrent-memory VLA”，而是：

> **一种把结构化事件/状态知识变成可部署连续 recurrent belief 的训练机制，以及对 selective write、trajectory supervision 和 action use 的受控验证。**

## 1. ICRA 2027 的硬约束

根据 ICRA 2027 官方 Call for Papers：

- 截稿：**2026-09-15 23:59 Pacific Time**；
- 初稿为 **8 页完整论文**，正文、图、表、致谢和参考文献全部计入 8 页；
- 双栏 ICRA 模板；
- **double-anonymous**，初稿不能出现作者和单位；
- 没有普通 supplementary PDF，审稿人只需阅读 8 页论文和可选视频；
- 可选视频最长 180 秒、最多 20 MB、最低 480p/20 fps；
- 至少选择 3 个 ICRA keywords；
- 生成式 AI 产生的论文内容需要按当年 RAS 规则处理和披露，单纯语言润色通常不在强制披露意图内。

官方页面：<https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/>

这意味着论文不能沿用旧的“8 页正文 + 不限参考文献”思路。建议从第一版 LaTeX 起就把参考文献压在 0.75--1.0 页内，并把完整视频结果放进 accompanying video，而不是等待最后再压缩。

## 2. 最新思路的收敛过程

### 2.1 从完整历史到固定容量状态

早期方案尝试完整视频、历史帧拼接或直接 recurrent token。当前结果支持：

- 单纯增加完整上下文不能稳定解决事件顺序绑定；真实杯子 36 帧 full-context 的单事件平均准确率为 51.7%，完整三事件仅 15%。
- 固定容量 memory 可以用短片段递归更新；ShellGame 的 6 帧非重叠 clip、10 次共享更新在 500 个 held-out episode × 6 个 offset 上达到 77.23% final slot accuracy。
- 但上述 77.23% 仍使用 GT initial slot 和已知状态变化位置构造训练监督，因此只能证明结构与 full-unroll recipe 可行，不能直接写成端到端闭环成功率。

### 2.2 从显式事件输入到“事件只监督、不进入部署”

Qwen3-VL 在正常速度 ShellGame 滑窗上能达到 100% event precision/recall，但对快速事件、窗口位置和去重尺度敏感。四任务统一 Qwen 的 clip-level event macro accuracy 已达到 72.74%，但 episode rollout 的 final state/final answer 仅为 30%/35%。

因此最新、更稳的分工是：

```text
structured event/state semantics
        │
        ├── construct causal state trajectories
        ├── construct canonical continuous teacher memory
        └── supervise when/what the student should retain

raw visual chunks + previous student memory
        │
        └── direct-visual recurrent student used at deployment
```

事件教师的核心价值从“线上替策略做决定”改成“离线塑造一个可学、可读、可递推的 latent state space”。

### 2.3 从硬触发到 carry-biased soft write

固定 12 帧、无 trigger 的四任务学生在 40--95 次连续更新中发生严重 drift：all-state 21.2%，final 6.7%。加入一个不使用 event/gate label 的标量 soft gate 后：

- all-state：21.2% → 46.0%；
- change-state：6.7% → 20.1%；
- final：6.7% → 13.3%；
- change chunk 的平均 gate 比 hold chunk 高 22.9%。

final/change 加权续训进一步把 final 提升到 35.0%，但 all-state 从 46.0% 降到 43.4%，暴露出终态与完整轨迹之间的优化权衡。

这一结果支持将方法写成：

> **event/state-supervised, but detector-free at deployment**。

不建议声称 soft gate 已经等价于 calibrated event detector：zero-video 时 gate 反而异常增大，且倒序敏感性仍偏弱。

### 2.4 从“memory probe”到“memory → action”

最新 ShellGame action 接口使用 continuous memory 解码目标 waypoint，并让 π action expert 生成局部连续控制：

- 在 21 个 direct memory 语义正确的 held-out episode 上，hard waypoint anchor 达到 21/21 target selection、21/21 进入 30 mm precision radius、9/21 lift success（42.9%）；
- 关闭 waypoint anchor 时 target selection 为 13/21、precision 为 14/21、lift 为 0/21；
- 相同 step-2000 checkpoint 的平均 policy inference 约 50 ms；anchor-ablation server 的端到端记录约 133 ms，二者协议不同，最终论文前必须统一计时方法。

这证明 memory 已能参与动作选择，但当前结果仍是“在 memory 正确条件下的 action bridge”，不是完整的端到端 memory success。论文最终必须补：

```text
raw video → online recurrent memory → action → task success
```

并与 zero/wrong/shuffled/oracle memory 在同一批 episode、同一 diffusion noise 下成对比较。

## 3. 推荐的最终系统定义

### 3.1 训练期

对 goal (g)、视频 (o_{1:T})、结构化事件 (e_t) 和状态 (s_t)：

1. 因果状态转移：

   \[
   s_t = \mathcal{T}(s_{t-1}, e_t, g).
   \]

2. Canonical continuous teacher memory：

   \[
   M_t^T = E_T(s_t,g), \qquad M_t^T \in \mathbb{R}^{K\times d}.
   \]

3. Direct-visual recurrent student：

   \[
   \tilde M_t^S = U_\theta(M_{t-1}^S, V_\theta(o_{t-k+1:t}), g).
   \]

4. Carry-biased soft write：

   \[
   \alpha_t = \sigma(G_\theta(M_{t-1}^S,V_\theta(o_{t-k+1:t}))),
   \]

   \[
   M_t^S = M_{t-1}^S + \alpha_t(\tilde M_t^S-M_{t-1}^S).
   \]

5. 轨迹级训练目标：

   \[
   \mathcal L =
   \lambda_m \sum_t w_t\mathcal L_{mem}(M_t^S,\operatorname{sg}(M_t^T))
   +\lambda_s\sum_t w_t\mathcal L_{state}(R_T(M_t^S),s_t)
   +\lambda_a\mathcal L_{action}.
   \]

其中：

- (R_T) 是冻结的 teacher readout，防止新 head 迁就任意 student latent；
- (w_t) 平衡 final/change/hold，避免大量 no-change chunk 主导训练；
- `sg` 表示 stop-gradient；
- `memory loss` 使用 token-aligned cosine + MSE；
- 训练以当前 student 完整 unroll，允许梯度穿过多个 update，而不是使用 stale precomputed student state。

### 3.2 部署期

```text
short visual chunk + previous fixed memory + goal
                    ↓
       visual encoder + recurrent updater
                    ↓
            learned soft write gate
                    ↓
              updated memory
                    ↓
     memory-conditioned waypoint/action expert
```

部署时不输入：

- GT event / GT state；
- Qwen JSON / Qwen latent；
- symbolic teacher memory；
- simulator metadata；
- task-specific memory head。

### 3.3 方法最关键的三个区分

1. **不是长历史编码器**：状态大小 (K\times d) 不随 episode 长度增长。
2. **不是 online language memory**：结构化语义只在训练目标构造中出现。
3. **不是普通 RMT**：训练显式对齐完整的因果 latent-state trajectory，并由冻结 readout 约束语义可读性。

## 4. 当前证据账本

### 4.1 已经能够安全写进初稿的事实

| 事实 | 当前结果 | 可以支持的表述 | 仍需补充 |
|---|---:|---|---|
| Teacher latent loss 有用 | ShellGame mean stage 45.00% → 68.61%；final 33.75% → 50.00% | teacher memory provides a useful transition/representation anchor | 3 seeds；至少一个 RoboMME 任务复现 |
| 短片段递归可行 | 500 ep × 6 offsets，final 77.23% | fixed-capacity recurrent update can track multi-event state | 去掉 GT initial；长序列/变速 |
| Oracle-centered 四任务视觉蒸馏可行 | field 91.8%，state 57.2%，final 46.7%，sequence 26.7% | one shared visual student can align to four-task canonical memory | event trigger/固定 chunk 端到端 |
| 无条件连续写入会 drift | fixed chunk all-state 21.2%，final 6.7% | dense recurrent updates require selective write | 多 seed |
| Soft gate 抑制 drift | all-state 46.0%，final 13.3%；change gate +22.9% | learned write-rate control improves long recurrent stability | OOD calibration；倒序/速度 |
| 终态与轨迹有权衡 | final 加权后 final 35.0%，all-state 43.4% | optimizing final readout alone can hide trajectory degradation | gradient-balanced 训练 |
| 多任务负迁移不是唯一瓶颈 | 单任务 macro final 40.0% vs unified 35.0% | hard tasks are limited by visual/state transition quality, not only task interference | 扩大 test；置信区间 |
| Qwen 可迁移到真实杯子 | local pair 35% → 80%；all-3 exact 5% → 55% | structured event supervision can be adapted to real video | 自动 boundary；学生蒸馏；真机动作 |
| Memory 可驱动 action | 正确-memory 子集 hard anchor：selection 100%，lift 42.9% | continuous memory can condition target selection and control | 完整端到端、多条件成对对照 |

### 4.2 目前只能写成限制或未来工作的内容

- 四任务 direct-visual student 已经全面优于 RoboMME 官方 memory baselines；
- 统一 Qwen 已经能为四任务提供可靠在线 teacher；
- soft gate 已经学到稳定、可解释、跨速度泛化的 event detector；
- 四任务 full-sequence memory 已解决；当前固定 chunk unified model 仍为 0%；
- memory 提升已经在四任务上转化为 closed-loop action success；
- 当前方法已经在真实机器人上闭环验证；
- 方法是 task-agnostic；canonical state field 和 transition target 仍带任务结构。

## 5. 论文标题与一句话卖点

### 5.1 推荐标题

**Learning What to Remember: Event-Supervised Latent Memory Distillation for Robot Control**

优点：突出 selective write 和训练机制；不误称首次提出 recurrent VLA；允许正文同时容纳 structured teacher、soft gate 和 action。

### 5.2 备选标题

1. **From Events to Actions: Distilling Structured State into Compact Recurrent Memory for Robot Policies**
2. **Fixed-Cost Recurrent Memory for Robot Control via Event-to-State Distillation**
3. **Distilling Causal State Trajectories into Direct-Visual Memory for Long-Horizon Manipulation**
4. **Event-Supervised Recurrent Belief Learning for Memory-Augmented Vision-Language-Action Models**

若最终 closed-loop VLA 结果不足，建议用第 3 个标题，降低 action/VLA 主张；若闭环主表完成且显著优于 memoryless/RMT，再使用推荐标题或第 4 个。

### 5.3 一句话 elevator pitch

> We turn structured event trajectories into a training signal for a compact visual belief state, so that a robot can remember with fixed cost and act without running the semantic teacher online.

## 6. 贡献点的推荐写法

以下版本可作为 Introduction 末尾的 v0；第三点中的结果需要在最终主实验后替换。

1. **Training mechanism.** We introduce an event-supervised latent memory distillation framework that maps structured causal state trajectories into a continuous teacher space and distills the full trajectory into a direct-visual recurrent student.
2. **Deployable memory.** We develop a fixed-capacity student with carry-biased soft writing and per-step state-readable supervision. It processes non-overlapping visual chunks and does not require event labels, a VLM, or growing history at deployment.
3. **Controlled evaluation.** We disentangle the effects of teacher-memory supervision, write gating, trajectory-versus-final objectives, and memory-conditioned action on ShellGame and representative RoboMME memory tasks, with additional real-video event adaptation.

只有补齐同协议 closed-loop 主表后，第三点才改成：

> We demonstrate statistically significant gains in closed-loop manipulation success and inference efficiency over memoryless, fixed-history, online-symbolic, and directly recurrent baselines.

## 7. Abstract v0

下面这版有意保守，只使用当前成立的机制与结果；`[TBD]` 必须由最终闭环实验替换。

> Long-horizon robot manipulation is partially observable: task-relevant events may disappear from view long before they determine an action. Retaining more frames increases cost, while directly recurrent vision-language-action policies receive weak supervision for deciding what to write and what to preserve. We present Event-Supervised Latent Memory Distillation (ESL-Mem), a training framework that converts structured event and state trajectories into a continuous teacher memory and distills the resulting causal trajectory into a fixed-capacity direct-visual recurrent student. The student processes short non-overlapping video chunks, updates its memory through a carry-biased soft write gate, and conditions a continuous action expert; the structured teacher, event labels, and large vision-language model are absent at deployment. Controlled ShellGame experiments show that teacher-memory supervision improves mean recurrent-stage accuracy from 45.0% to 68.6%. Across four representative RoboMME tasks, learned soft writing improves all-step state accuracy from 21.2% to 46.0% over unconditional recurrent updating under identical data and training budgets. In closed-loop control, ESL-Mem improves task success from [TBD]% to [TBD]% over a memoryless policy while reducing online semantic-memory latency by [TBD]×. These results suggest that structured supervision is most useful not as an online symbolic controller, but as a training-time scaffold for compact visual state estimation.

最终 abstract 应遵循 5 句结构：

1. 机器人问题与 partial observability；
2. 现有长历史/online VLM/direct RMT 的共同缺口；
3. 方法一句话；
4. 两个最强闭环数字 + 一个效率数字；
5. 结论与适用范围。

不要在摘要堆四任务的所有 memory probe 数字。

## 8. 8 页论文结构与页数预算

| 内容 | 目标页数 | 必须承载的内容 |
|---|---:|---|
| Title + Abstract + I. Introduction | 1.10 | 问题、缺口、核心图、三条贡献 |
| II. Related Work + Problem Formulation | 0.80 | 三类 memory、符号到 latent 的空位、POMDP/固定容量定义 |
| III. Method | 1.75 | teacher state、canonical memory、visual recurrence、soft gate、loss、action interface |
| IV. Experimental Setup | 0.65 | 任务、split、baselines、指标、统计协议 |
| V. Results | 2.25 | 闭环主表、核心消融、泛化/效率、错误分析 |
| VI. Discussion and Conclusion | 0.45 | 适用边界、限制、结论 |
| References | 1.00 | 约 22--28 篇精简引用 |
| **合计** | **8.00** | 参考文献也在 8 页内 |

### I. Introduction

建议用四段完成：

1. **任务矛盾**：机器人需要记住早期 reveal、完成次数、对象交换和位置顺序，但当前画面可能完全相同。
2. **现有解法缺口**：长历史和 memory bank 随 horizon 增长；online language/VLM memory 慢且错误会递归污染；纯 recurrent tokens 又缺少“何时写、写成什么状态”的监督。
3. **方法直觉**：structured semantics 最有价值的用法不是部署时反复推理，而是训练时定义 causal state geometry；学生只看短视频并学习 soft write。
4. **贡献**：使用第 6 节三点。

Introduction 中建议明确写一句：

> Our goal is not to recover every pixel from the past, but to retain the task-relevant belief that makes the next action identifiable.

### II. Related Work and Problem Formulation

#### A. Memory-Augmented Robot Policies

压缩成三类：

- history window / video context；
- retrieval or memory bank：MemoryVLA、LaMem-VLA；
- compact recurrence or belief：RoboMME RMT、μVLA、RB-VLA、ReMem-VLA。

我们的差异不应写成“fixed recurrent tokens”，而应写成：

> We supervise the geometry and transition of the recurrent state with structured causal trajectories, then remove the structure-producing teacher at deployment.

#### B. Semantic and Language Memory

引用 MEM、Explicit Language Memory、RoboMME symbolic memory。强调这些方法在线保留文本/语义生成能力，而我们牺牲开放式可读性，换取固定成本、高频 latent control interface。

#### C. Distillation for Robot Learning

只需一小段，将本文与普通 logits/feature distillation 区分：目标是整条 recurrent state trajectory，而不是单帧答案或 frozen VLM hidden state。

#### D. Problem Setup

定义 POMDP：

\[
M_t=U_\theta(M_{t-1},o_{t-k+1:t},g), \qquad
a_{t:t+H}\sim\pi_\phi(o_t,g,M_t).
\]

目标同时包括：

- state sufficiency；
- causal online update；
- fixed memory/storage cost；
- closed-loop task success；
- teacher-free deployment。

### III. Method

#### A. Overview

放主图 Fig. 2 或 Fig. 1；图中必须将 train-only 与 deployment 用颜色/虚线明确分开。

#### B. Structured Causal Teacher

定义统一 event contract：

```json
{"event": "...", "entity": "...", "region_a": "...", "region_b": "..."}
```

但正文不要花太多篇幅解释 Qwen prompt。核心是事件经 transition grammar 得到 committed state，非法/未完成事件为 no-op。Qwen 是一种 event annotation 来源；仿真 GT 和人工标签是当前主要高质量监督来源。

#### C. Canonical Latent State

说明为什么不是直接蒸馏离散标签：

- 离散 readout 对 transition geometry 监督不足；
- canonical memory 提供高带宽 token-aligned target；
- frozen readout 约束 student memory 落入同一语义 basis。

#### D. Direct-Visual Recurrent Student

写清：固定 chunk、共享 updater、full unroll、previous memory carry、padding no-op。

#### E. Soft Write and Trajectory Objective

这是方法最需要展开的一节。说明 carry bias、residual interpolation 和 final/change/hold balancing。不要把 gate 宣称成显式 event detector。

#### F. Memory-Conditioned Action

建议把 action 接口写成两层：

1. raw memory token cross-attention 提供语义条件；
2. waypoint auxiliary head/anchor把目标选择与局部操作解耦。

最终若 waypoint anchor 仍是推理期 hard overwrite，必须在方法和消融中完全披露，不能只把它描述成普通 auxiliary loss。

### IV. Experimental Setup

#### A. Tasks

正文优先呈现互补的三类记忆需求：

1. ShellGame / VideoUnmaskSwap：对象身份与交换递归；
2. PickXTimes：计数和 procedural state；
3. VideoUnmask：遮挡下 object permanence；
4. VideoPlaceOrder：ordered spatial memory，可放表内但少展开。

若篇幅不足，四任务仍保留完整主表，但定性图只展示 ShellGame、Pick、Unmask。

#### B. Baselines

最小可投稿 baseline 集合：

1. Memoryless π policy；
2. Fixed history / uniform frame sampling；
3. Online Qwen symbolic memory；
4. Direct recurrent memory，action loss only；
5. Direct recurrent + state readout；
6. ESL-Mem without soft gate；
7. Full ESL-Mem；
8. Oracle state/memory upper bound。

若能公平复现，再加入 RoboMME 最佳 memory variant；不能公平复现时不要用不同数据/预算的数字做 SOTA 对比。

#### C. Metrics and Statistics

必须分层报告：

- event：precision、recall、incomplete false commit；
- memory：field、full-state、change/hold、final、full sequence；
- action：task success、target selection、precision reach、lift/press completion；
- efficiency：latency、peak memory、history storage、policy Hz；
- statistics：至少 3 个训练 seed；每任务建议 50--100 rollout；paired episode/noise；95% bootstrap CI 或 Wilson interval。

### V. Results

建议用问题式小节标题：

#### A. Does Structured Teacher Memory Improve Recurrent State Learning?

放 teacher-memory controlled ablation：45.00% → 68.61% mean stage，33.75% → 50.00% final。补齐 3 seeds 后再作为主结论。

#### B. Does Selective Writing Prevent Long-Horizon Memory Drift?

放 fixed chunk no-gate vs soft-gate：21.2% → 46.0% all-state；分析 change/hold gate。将 final/change weighted fine-tune 放附属列，明确 trajectory/final tradeoff。

#### C. Does the Distilled Memory Improve Closed-Loop Control?

这是最终主表，当前尚未完成。至少报告：

| Method | PickXTimes | Unmask | Swap/ShellGame | PlaceOrder | Macro | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Memoryless | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed history | TBD | TBD | TBD | TBD | TBD | TBD |
| Online symbolic | TBD | TBD | TBD | TBD | TBD | TBD |
| Direct RMT | TBD | TBD | TBD | TBD | TBD | TBD |
| ESL-Mem | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle memory | TBD | TBD | TBD | TBD | TBD | TBD |

若四任务闭环在截稿前无法完成，最低可接受替代是 ShellGame + Pick + Unmask 三任务，每个任务 50+ rollout，并把四任务 memory probe 作为泛化表；只有 21 个条件化 ShellGame episode 不足以撑起 ICRA 主表。

#### D. Is the Policy Causally Using Memory?

同一 episode 和 diffusion noise 下比较：

- correct student memory；
- oracle teacher memory；
- zero memory；
- wrong-episode memory；
- shuffled memory token；
- memory with correct final readout but wrong trajectory。

这张表比单纯相关性 probe 更重要。已有 6-episode pilot 显示 wrong/zero memory 会降低 cup selection，但样本量太小，需扩大。

#### E. Generalization, Efficiency, and Failure Modes

优先顺序：

1. 更长 episode / 更多状态转移；
2. clip offset、速度 6/10/14、丢帧；
3. unseen required count / prompt paraphrase；
4. 真实杯子 local event → student memory；
5. latency 随 horizon 的增长曲线。

本节要承认的失败：swap relation grounding、倒序不敏感、zero-video gate 打开、final/trajectory tradeoff。

### VI. Discussion and Conclusion

只保留三点：

1. 适合稀疏事件、显式 task state、需要高频控制的部分可观测任务；
2. 相比文本/检索 memory，固定 belief 丢失开放式历史细节；
3. 当前 structured state contract 仍需人工/仿真设计，未来应由视频预测或自监督 dynamics 扩展。

## 9. 主图与主表规划

### Figure 1：任务动机 + 一句话结果

左：早期 reveal/交换/遮挡；右：当前帧相同但正确动作不同。下方画 history cost：frame stack 随时间增长，ESL-Mem 固定。

### Figure 2：训练/部署非对称架构

```text
                         TRAIN ONLY
Event annotations ──→ causal state trajectory ──→ canonical teacher memory
       ▲                                                   │
       │ VLM / GT / human                                  │ distill
video ─┴─→ chunk encoder → recurrent candidate → soft gate ┴→ student memory
                                                           │
                                                           ▼
                                                     action expert

                         DEPLOYMENT
short visual chunk + previous memory ─→ updater + soft gate ─→ action
```

图中必须标：

- teacher/Qwen/labels are train-only；
- memory shape 固定；
- updater 参数跨所有时间步共享；
- action 读取 latent memory，不读取 readout class；
- readout 是训练/诊断接口。

### Figure 3：Selective write 行为

画 episode timeline：hold、partial event、committed transition、post-event；同时画 gate、state accuracy 和 memory delta。相比只画均值柱状图，这张图更能解释方法机制。

### Table I：Closed-loop 主结果

必须是同协议多任务 success + latency。当前缺失。

### Table II：核心消融

建议只留 5 行：

1. state labels only；
2. + latent teacher；
3. unconditional write；
4. + soft write；
5. + full trajectory/final-balanced training。

列包含 mean stage、change、hold、final、sequence；不要把十几个 pilot checkpoint 全塞进正文。

### Table III：因果 memory/action 对照或泛化

根据最终结果二选一：

- 如果 action 条件对照明显，放 correct/oracle/wrong/zero memory；
- 如果 action 差异不稳，放 length/offset/speed 泛化并在正文讨论 action failure。

## 10. 补实验优先级

### P0：不完成就无法支撑主论文

1. **端到端闭环主表**：至少 3 个任务、每任务 50--100 rollout、paired seed/noise。
2. **直接 RMT 公平基线**：相同 backbone、数据、参数量和训练步数；只去掉 structured latent target/readout。
3. **Teacher latent ablation 3 seeds**：当前关键数字只有一个 seed。
4. **Soft gate ablation 3 seeds**：确认 21.2% → 46.0% 不是单 seed 波动。
5. **统一效率测量**：同硬件、warm-up、batch=1，分别测 online Qwen、student updater、action expert 和端到端 Hz。
6. **正确/错误 memory 的 action 因果对照**：同 episode、同 diffusion noise。

### P1：显著提高接受概率

1. 去掉 GT initial slot，将 initial perception 纳入完整指标；
2. 训练长度外推和速度/offset 泛化；
3. final/trajectory 的梯度平衡或 alternating optimization；
4. VideoUnmaskSwap 的 swap-order hard negatives；
5. 真实杯子自动 boundary → recurrent student 的 memory 结果；
6. accompanying video：任务歧义、memory update、wrong-memory counterfactual、实时闭环。

### P2：可放 rebuttal 或后续版本

- 多假设 belief；
- action-conditioned dynamics/world-model loss；
- 更开放的语言状态合同；
- 跨任务 held-out transfer；
- 真机高频完整闭环。

## 11. 写作边界与审稿风险

### 风险 1：被认为只是 teacher + RNN + gate 的模块拼接

回应必须靠实验，而不是措辞：

- state-only vs latent teacher；
- latent teacher without gate vs with gate；
- final-only vs full trajectory；
- direct RMT vs ESL-Mem；
- correct vs wrong memory 对动作的因果影响。

### 风险 2：teacher 太任务特定

正文主动承认 structured state contract；强调共享 19-field readout、共享 memory core 和统一 event schema。不要写 task-agnostic。可以写：

> task-structured supervision with a shared recurrent architecture.

### 风险 3：memory 指标提高但 action 没提高

动作主表必须优先于继续刷 probe。若最终 action 不显著，应将论文降级为“recurrent state learning analysis”，不要在标题中使用 VLA control 的强主张。

### 风险 4：Qwen 结果与主方法脱节

Qwen 的正文角色限定为：

1. structured event annotation 的一种来源；
2. 说明 online symbolic rollout 的误差累积；
3. 真实视频语义适配的证据。

不要把 Qwen local clip accuracy 与 student closed-loop success 混成一条未验证链路。

### 风险 5：数据泄漏或指标条件化

必须逐项披露：

- GT-centered event window；
- GT initial slot；
- 只在 direct memory 正确子集上的 action result；
- simulator metadata proxy；
- target identity 是否仅用于评分；
- final label 是否参与 episode 筛选。

所有条件化结果都要与完整 end-to-end success 分开。

## 12. 初稿写作顺序

不要从 Introduction 顺写。推荐：

1. 先锁定 Figure 2 和方法公式；
2. 建 Table I 的空表并确定所有 baseline/指标；
3. 写 Experimental Setup，暴露协议缺口；
4. 写 Method；
5. 结果到齐后写 Results；
6. 最后写 Introduction、Abstract 和标题；
7. 同步准备 180 秒视频，不把关键证据只放视频。

从 2026-08-27 到 2026-09-15 的建议节奏：

- 08-27--08-30：冻结方法与评估协议，搭 LaTeX 骨架、主图和空表；
- 08-30--09-05：P0 baseline、3 seeds、closed-loop rollout；
- 09-05--09-09：结果冻结、统计、图表、初稿 v1；同时完成第一视频窗口；
- 09-10--09-12：内部 review、匿名化、压到 8 页；
- 09-13--09-14：PaperPlaza compliance、视频、最终交叉检查；
- 09-15：只做上传与余量修正，不再改变实验协议。

## 13. 推荐关键词

提交时从官方 ICRA keyword 列表中选择最接近的至少三项。内容上优先考虑：

- Vision-Based Control；
- Deep Learning for Visual Perception；
- Robot Learning；
- AI-Based Methods；
- Grasping / Manipulation Planning（根据官方词表实际名称选择）。

## 14. 精简相关工作清单

正文约保留 22--28 篇。与本文最直接相关的近期论文：

- RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies, arXiv:2603.04639.
- MEM: Multi-Scale Embodied Memory for Vision Language Action Models, arXiv:2603.03596.
- MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation, arXiv:2508.19236 / ICLR 2026.
- μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models, arXiv:2606.12497.
- Recursive Belief Vision Language Action Models (RB-VLA), arXiv:2602.20659.
- ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries, arXiv:2603.12942.
- Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation (LaMem-VLA), arXiv:2607.07608.
- Explicit Language Memory for Long-Horizon Planning in Vision-Language-Action Models, arXiv:2608.04765.

相关工作中要明确承认：fixed recurrent tokens、TBPTT、memory bank、language memory 和 memory-conditioned action 都不是本文独创。创新边界落在 structured causal trajectory → continuous teacher state → direct-visual recurrent student 的训练闭环。

## 15. 本地证据与产物索引

### 论文定位

- `docs/event_supervised_latent_memory_distillation_paper_positioning_260826.md`
- `docs/qwenvl_hierarchical_recursive_compressed_memory_design_260824.md`

### Teacher 与 student

- `docs/robomme_four_task_unified_gt_teacher_v2_260826.md`
- `docs/robomme_four_task_visual_student_distillation_260826.md`
- `docs/teacher_memory_ablation_paper_results_260826.md`
- `docs/recurrent_compact_visual_memory_replay_training_260826.md`

### Fixed chunk 与 soft gate

- `docs/robomme_four_task_fixed_chunk_no_trigger_experiment_260826.md`
- `docs/robomme_four_task_fixed_chunk_soft_gate_experiment_260826.md`
- `docs/robomme_four_task_soft_gate_final_finetune_260827.md`
- `docs/robomme_four_task_decoupled_trajectory_final_loss_260827.md`
- `docs/robomme_single_task_equal_exposure_ablation_260827.md`

### Qwen 与真实视频

- `docs/qwen3vl_sliding_event_trigger_generalization_260825.md`
- `docs/robomme_four_task_qwen_unified_optimized_v2_260826.md`
- `docs/robomme_four_task_qwen_recurrent_rollout_validation_260826.md`
- `docs/real_cup_qwen3vl_gt_finetune_results_260826.md`
- `docs/real_cup_qwen3vl_full_context36_results_260826.md`

### Action

- `evaluation/shellgame/waypoint_grasp_v6_step2000_val21_finalslot_balanced_replan8_260827/result.json`
- `evaluation/shellgame/waypoint_anchor_none_step2000_val21_finalslot_balanced_replan8_260827/result.json`
- `evaluation/shellgame/waypoint_anchor_hard_step2000_val21_finalslot_balanced_replan8_260827/result.json`
- `examples/shellgame/serve_qwen_event_memory_waypoint_anchor_ablation.py`

## 16. 可直接建立的 LaTeX 章节骨架

```latex
\title{Learning What to Remember: Event-Supervised Latent Memory Distillation for Robot Control}

\begin{abstract}
% Problem; gap; method; two closed-loop numbers + latency; conclusion.
\end{abstract}

\section{Introduction}
\section{Related Work and Problem Formulation}
\subsection{Memory-Augmented Robot Policies}
\subsection{Semantic Memory and Temporal Distillation}
\subsection{Problem Formulation}

\section{Event-Supervised Latent Memory Distillation}
\subsection{Method Overview}
\subsection{Structured Causal Teacher}
\subsection{Canonical Latent State}
\subsection{Direct-Visual Recurrent Student}
\subsection{Soft Writing and Trajectory Supervision}
\subsection{Memory-Conditioned Action}

\section{Experimental Setup}
\subsection{Tasks, Data, and Metrics}
\subsection{Baselines and Implementation Details}

\section{Results}
\subsection{Teacher-Memory Supervision}
\subsection{Selective Writing and Memory Drift}
\subsection{Closed-Loop Manipulation}
\subsection{Causal Memory Interventions}
\subsection{Generalization, Efficiency, and Failure Modes}

\section{Discussion and Conclusion}
```

## 17. 提交前的最后判据

### 作为完整 ICRA 方法论文提交

至少同时满足：

- 3 个任务上闭环 success 显著优于 memoryless 和 direct RMT；
- teacher latent 与 soft gate 的收益在多 seed 下成立；
- correct/wrong/zero memory 能因果改变 action success；
- 学生在线成本显著低于 online Qwen；
- 论文完全披露 hard waypoint anchor 和所有 GT 条件。

### 降低主张后提交

若只有 ShellGame 闭环，但 memory 学习消融完整，可改成“structured recurrent state learning for manipulation”，减少 unified/general/VLA 的表述。

### 不建议按当前强标题提交

若到截稿时仍只有：

- memory probe，无非条件化闭环主表；
- 单 seed 关键消融；
- GT-centered windows / GT initial slot；
- 21 个且仅筛选 memory 正确的 action episode；

则证据不足以支持“long-horizon robot control”的强结论，应继续补实验或显著收窄论文定位。
