# RoboMME 单一 Persistent Latent 严格消融（2026-09-01）

> **定位修正（2026-09-01）：**本实验只说明“删除原版 learned recurrent
> hidden state，并强制 `128×64` action readout tokens 自身承担递归”会失败；它不
> 反驳原版 recurrent MEM。原版模型的有效持久状态是 learned hidden state，action
> tokens 是它的 readout。正确结构的配对消融见
> `docs/robomme_correct_recurrent_hidden_state_ablation_3seed_260901.md`。本页结果仅
> 作为 naive output-token self-carry 的负面对照，不应进入主表。

## 结论

本实验修正了旧表 4 的结构矛盾：所有主消融均关闭外部
`CausalEvidenceScanCell`，历史只能由输出给 action 的 `128×64` latent memory
携带。实验结果证明 latent carry 和 soft write 能提高离线状态保持，但没有证明模型
使用视觉历史，也没有稳定提高最终任务 Answer。因此当前统一 latent 模型**不能接入
action，也不能作为视觉 recurrent MEM 已成立的正面证据**。

最关键的反证是：主模型在 zero-video、倒序和跨 episode 视频替换下几乎不下降。
其离线状态优势主要来自 goal、任务先验、有效长度和自主递归时钟，而不是视觉事件解析。

## 严格实验协议

- 一个 checkpoint/训练 seed 同时服务四个 RoboMME 任务；
- 训练 seeds：`260971/260972/260973`；
- 每组 2,000 步、batch size 4、固定非重叠 12 帧 chunk；
- 所有组 `causal_evidence_state=false`；
- 唯一允许跨 chunk 持久化的状态是 `128×64` output latent；
- 所有组统一按 dev terminal Answer、final state、all-state 的字典序选择 checkpoint；
- test 不参与 checkpoint 选择；
- D 仅监督 episode 终点 query 字段，不使用 teacher latent、中间状态、事件、gate、
  online phase 或 action distillation loss；
- D 与其余组共享冻结 readout，因此准确名称是 `w/o trajectory teacher`，不是
  `w/o all teacher components`。

## A/B/C/D 测试结果

以下均为三个训练 seed 的 mean ± sample SD，单位为百分比：

| Variant | Field | State | Change | Hold | Final | Answer |
|---|---:|---:|---:|---:|---:|---:|
| A. Latent carry + soft write | **88.5 ± 2.5** | **40.8 ± 7.2** | **12.6 ± 4.8** | **45.1 ± 7.6** | 7.2 ± 4.2 | 18.3 ± 2.9 |
| B. Reset latent + soft write | 83.2 ± 0.3 | 26.7 ± 1.4 | 12.2 ± 1.4 | 28.9 ± 1.4 | **11.1 ± 7.9** | 22.8 ± 9.2 |
| C. Latent carry + unconditional write | 80.1 ± 0.6 | 28.1 ± 0.7 | 8.8 ± 1.0 | 31.0 ± 0.7 | 5.0 ± 0.0 | 15.6 ± 5.9 |
| D. A w/o trajectory teacher | 28.9 ± 1.2 | 0.0 ± 0.0 | 0.1 ± 0.3 | 0.0 ± 0.0 | 0.6 ± 1.0 | **45.6 ± 1.0** |

### Paired seed 差值

- A−B：State `+14.1 ± 7.9`、Change `+0.4 ± 5.4`、Hold
  `+16.2 ± 8.2`、Final `−3.9 ± 11.8`、Answer `−4.4 ± 12.1`；
- A−C：State `+12.8 ± 6.6`、Change `+3.9 ± 3.8`、Hold
  `+14.1 ± 7.1`、Final `+2.2 ± 4.2`、Answer `+2.8 ± 6.3`；
- A−D：State `+40.8 ± 7.3`、Change `+12.5 ± 5.1`、Hold
  `+45.1 ± 7.6`，但 Answer `−27.2 ± 3.8`。

latent carry 对 State/Hold 的增益在三个 seed 上均为正，但对 Change、Final 和
Answer 不稳定。soft write 也主要改善 State/Hold；Answer paired gain 并不一致。
D 则稳定学会终点 query，却没有形成中间状态轨迹，是典型 terminal shortcut。

## 按任务 Answer

| Variant | Unmask | UnmaskSwap | PlaceOrder | PickXTimes |
|---|---:|---:|---:|---:|
| A | 20.0 | 15.6 | 37.8 | 0.0 |
| B | 33.3 | 15.6 | 17.8 | 24.4 |
| C | 17.8 | 4.4 | 35.6 | 4.4 |
| D | 20.0 | 15.6 | 46.7 | 100.0 |

A 在 PickXTimes 上的 Answer 为 0%，所以即使其 dense State 较高，也不能视为
action-ready memory。

## 视觉因果干预

对 A 的三个 dev-best checkpoint 重新推理：

| Input | State | Change | Final | Answer |
|---|---:|---:|---:|---:|
| Normal | 40.8 ± 7.2 | 12.6 ± 4.8 | 7.2 ± 4.2 | 18.3 ± 2.9 |
| Zero video | 41.6 ± 8.4 | 14.9 ± 4.4 | 7.8 ± 1.9 | 18.3 ± 3.3 |
| Reversed chunks | 40.1 ± 6.6 | 12.4 ± 4.5 | 7.2 ± 4.2 | 18.3 ± 2.9 |
| Within-task episode shuffle | 41.2 ± 7.7 | 13.8 ± 5.4 | 7.8 ± 4.2 | 18.9 ± 3.5 |

Zero video 不降、倒序不降、跨 episode 视频替换不降，否定了“当前 A 通过视觉
历史维护任务状态”的解释。State/Hold 增益更可能来自自主递归动力学和数据中的时间/长度
先验。正常输入下 gate 均值仅为 `0.0174 ± 0.0014`；change 与 hold 分别为
`0.0202 ± 0.0011` 和 `0.0169 ± 0.0014`。因此 gate 基本处于关闭状态，也没有形成
可用的事件选择性。

## 论文与下一步边界

1. 旧表 4 的 external causal-evidence 主模型不能继续代表论文公式中的 output-latent
   recurrence；
2. 新 A/B/C/D 可以作为诚实的结构诊断，但结论是负面的，不能用于声称视觉记忆成功；
3. 不应把 A 接入 action，也不应花费 3×50 闭环预算；
4. 下一步必须先加入能够阻止 time/goal shortcut 的视觉依赖目标，例如同任务、同长度的
   counterfactual video pairing，使相同 goal/step index 下不同事件轨迹必须产生不同
   memory；
5. 新模型必须先通过 `Normal > zero/shuffle/reverse` 的预注册门槛，再讨论 action。

## 产物

- 汇总（含视觉干预和 action promotion 判定）：
  `checkpoints/robomme_single_latent_ablation_3seed_260901_single_latent_confirm.json`
- 训练目录：`checkpoints/robomme_single_latent_<variant>_seed<seed>_260901_single_latent_confirm/`
- 训练 runner：`scripts/mem/run_single_latent_matched_ablation_3seed.sh`
- 干预 runner：`scripts/mem/run_single_latent_interventions_3seed.sh`
- 汇总审计：`scripts/mem/summarize_single_latent_matched_ablation.py`
