# RoboMME 正确 Recurrent Hidden-State 消融（2026-09-01）

## 结论

原版 recurrent MEM 有效。此前的结构冲突来自把“跨窗口持久化的 hidden state”和
“输出给下游的 `128×64` memory readout tokens”误认为必须是同一个张量。实际有效
结构为：

```text
h_i = C(h_{i-1}, z_i)
M_i = M_0(g) + alpha_i [U(M_0(g), z_i, h_i) - M_0(g)]
```

其中 `h_i` 是 learned、视觉驱动、跨 chunk 持久化的64维 hidden state；`M_i` 是
与 teacher memory 同形状的 `128×64` readout。它可被状态头或下游策略读取，但不需要
再次作为 `h_{i+1}` 输入。这是标准的 recurrent hidden-state/readout 分工，推理不使用
GT event、state 或 simulator 标签。

## 严格协议

- 四任务共享一个 checkpoint；
- 训练 seeds：`260951/260952/260953`；
- 每组相同数据、模型宽度、batch size和2,000步预算；
- 为消除 checkpoint-selection 混杂，所有行统一评测固定的 step 2,000；
- A/B/C 复用已有训练，D 新训练三个 seed；
- D 关闭 teacher-memory、轨迹 state、event、gate、online phase 和 action
  distillation，只监督 episode 终点 query 字段；
- test 只用于最终评分。

## A/B/C/D 固定 step-2,000 结果

三个训练 seed 的 mean ± sample SD，单位为百分比：

| Variant | Field | State | Change | Hold | Final | Answer | Sequence |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Recurrent hidden + soft readout | **91.5 ± 0.2** | **55.2 ± 0.4** | **35.4 ± 4.7** | **58.3 ± 1.0** | **28.3 ± 15.3** | 36.1 ± 13.9 | **2.2 ± 2.5** |
| B. Reset hidden + soft readout | 83.0 ± 0.3 | 26.7 ± 1.5 | 15.2 ± 3.4 | 28.5 ± 1.9 | 17.2 ± 6.9 | 25.6 ± 9.6 | 0.0 ± 0.0 |
| C. Recurrent hidden + unconditional readout | 90.6 ± 0.3 | 50.1 ± 0.6 | 26.5 ± 1.8 | 53.7 ± 0.4 | 15.6 ± 2.5 | 20.0 ± 3.3 | 1.1 ± 1.9 |
| D. A w/o trajectory teacher | 27.9 ± 3.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **48.3 ± 1.7** | 0.0 ± 0.0 |

配对 seed 差值：

- A−B：State `+28.5 ± 1.8`、Change `+20.2 ± 4.3`、Hold
  `+29.8 ± 2.7`、Answer `+10.6 ± 5.1`；
- A−C：State `+5.1 ± 0.5`、Change `+8.9 ± 3.5`、Hold
  `+4.5 ± 1.1`、Answer `+16.1 ± 15.0`；
- A−D：State `+55.2 ± 0.4`、Change `+35.4 ± 4.7`、Hold
  `+58.3 ± 1.0`，但 Answer `−12.2 ± 14.4`。

因此 recurrent hidden state 对 State/Change/Hold 的增益在三个配对 seed 上均为
正；soft readout 对这三项也均有正增益。D 的 Answer 较高但整条状态轨迹为零，是
终点 shortcut，只能证明“终点标签可以直接拟合”，不能替代 recurrent memory。

## 视觉因果干预

对 A 的固定 step-2,000 checkpoint：

| Input | State | Change | Final | Answer | Mean gate |
|---|---:|---:|---:|---:|---:|
| Normal | **55.2** | **35.4** | **28.3** | **36.1** | 0.421 |
| Zero video | 18.7 | 0.0 | 0.0 | 0.0 | 0.115 |
| Reversed chunks | 51.5 | 30.7 | 17.2 | 21.7 | 0.421 |
| Within-task episode shuffle | 53.3 | 31.5 | 29.4 | 32.2 | 0.421 |

Zero video 使 State 下降36.5点、Answer降到0，证明 A 的输出不是只依赖 goal 或内部
时钟。倒序也使 Answer 下降14.4点，表明决定性 readout 使用时间顺序。长度匹配的
跨 episode shuffle 下降较小，说明当前模型对 episode identity 的区分仍弱，这是局限，
但不否定视觉依赖。

## 与 naive single-latent 实验的关系

`robomme_single_latent_matched_ablation_3seed_260901.md` 删除了有效的 hidden state，
改成让 action tokens 自身递归。该版本 gate 坍缩到约0.017且 zero-video 不降，说明这种
替换训练失败。它应作为负面结构消融，而不是原版 recurrent MEM 的主结果。

## 闭环边界

三个静态 RoboMME 任务的官方 GroundSG action 接口实际接收由 MEM readout 解码并
grounding 后的 subgoal 文本，不是直接注入 `128×64` tokens。因此对应的闭环干预必须
称为 `memory/subgoal causal intervention`。PickXTimes 另有直接 latent adapter；两种
接口不能混写。

## 产物

- 汇总：`checkpoints/robomme_correct_hidden_state_ablation_3seed_step2000.json`
- D checkpoints：
  `checkpoints/robomme_unified_framework_no_trajectory_teacher_seed<seed>_260901_correct_hidden_state/`
- D runner：`scripts/mem/run_correct_hidden_state_no_trajectory_teacher_3seed.sh`
- 固定步重评：`scripts/mem/run_correct_hidden_state_fixed_step_interventions.sh`
- 汇总审计：`scripts/mem/summarize_correct_hidden_state_teacher_ablation.py`
