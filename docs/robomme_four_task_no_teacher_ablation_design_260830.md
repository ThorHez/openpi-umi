# RoboMME 四任务 No-Teacher 主表消融设计（2026-08-30）

> **结果状态更新（2026-09-01）：** 第 4 节的主表模板已同步当前正式闭环结果。
> 四项任务均已刷新。No-Teacher 正式结果为 PickXTimes `36.0 ± 2.0%`、
> VideoUnmask `26.0 ± 0.0%`、VideoUnmaskSwap `20.7 ± 1.2%`、
> VideoPlaceOrder `28.0 ± 0.0%`，合计 166/600。此前 PlaceOrder 的 86.0% 因
> learned-event 参数未覆盖而误加载 Full checkpoint，现已由 150 条 trace 核验更正。
> 当前行仍是 task-specific 系统基线；
> Pick updater 与 Full 不同，因此不能把四任务差值解释成统一的 loss-only 因果效应。

## 1. 要回答的问题

在保持视觉输入、goal conditioning、recurrent MEM 结构、训练 episode、优化预算、
MEM-to-action 接口和官方 MME action checkpoint 不变时，训练期的特权轨迹 teacher
是否显著提高四个任务的闭环成功率？

这里的 `No Teacher` 指 **不使用任何中间过程 teacher supervision**，而不是随机化
memory，也不是只去掉 Qwen。当前主模型在线推理本来就不调用 Qwen。

## 2. 严格实验定义

### Full model

使用现有训练配方中的特权轨迹监督：

- 每个 chunk 的 semantic state target；
- `event_type`；
- write 的 `entity/region`；
- swap 的 region pair；
- 训练期 previous-state teacher forcing/curriculum；
- episode terminal answer。

teacher 只在训练期使用，测试期仍为 causal free rollout。

### Ours w/o Teacher

网络、输入和数据 episode 与 Full model 完全相同，但移除：

- 所有中间 event/payload label；
- 所有中间 semantic-state target；
- previous-state teacher forcing；
- teacher latent、teacher delta 或 teacher readout loss。

仅保留每条 episode 可直接提供的 terminal task-answer loss：

```text
PickXTimes       : required pick count / terminal done state
VideoUnmask      : queried color(s) 的最终 region
VideoUnmaskSwap  : swap 后 queried color(s) 的最终 region
VideoPlaceOrder  : queried ordinal object 的最终 region
```

因此训练目标为：

```text
L_no_teacher = L_terminal_answer
```

权重衰减等与 Full 相同的普通优化正则可以保留；不得新增 event pseudo-label、
oracle window、dense state loss 或 task-specific head。否则不再是严格的 No-Teacher
对照。

最终答案标签不算 trajectory teacher。若连 terminal answer 也移除，在当前冻结 action
backbone、非可微 simulator 的设置下将没有可用训练信号，得到的只是 random-memory
control，而不是有意义的 teacher 消融。

## 3. 公平控制项

两组必须固定：

| 项目 | 控制规则 |
|---|---|
| train/dev/test episode | 完全相同，episode-disjoint |
| RGB/proprio/prompt | 完全相同 |
| chunk | 12 帧，保持当前 causal 顺序 |
| goal token | 完全相同 |
| MEM 参数量与初始化 | 完全相同 |
| frozen visual backbone | 完全相同 |
| optimizer / LR / batch | 完全相同 |
| 总训练 step / episode exposure | 完全相同 |
| action backbone | `symbolic-grounded-subgoal/79999`，冻结 |
| action horizon | 1300 simulator steps |
| action seeds | 7、17、27 |
| 每任务 episode | 每个 seed 50 条 |

Full 与 No-Teacher 都应按同一个、两者都可访问的 dev 指标选 checkpoint：

```text
mean four-task dev terminal-answer accuracy
tie-break: minimum per-task dev terminal-answer accuracy
```

不能让 Full 用 dense teacher trajectory metric 选 checkpoint，而 No-Teacher 只能用
terminal metric，否则 checkpoint selection 本身会成为额外变量。

## 4. 两级评测

### A. Memory-only 诊断

训练时不提供中间标签，但评估时可用锁定的 simulator GT 计算：

- terminal answer exact accuracy；
- transition state exact；
- no-change/hold state exact；
- all-state accuracy；
- full-sequence exact；
- event false-positive rate 和 full-update recall（前三个 region 任务）。

这些指标用于解释闭环差异，不能用于 No-Teacher checkpoint 选择。

### B. 正式闭环主表

四任务都使用相同协议：3 action seeds × 50 episodes × 1300-step 上限。报告：

- 每个任务 mean ± std；
- 每个任务 aggregate success（150 episodes）；
- 四任务 macro average；
- teacher gain：`Full - No Teacher`。

当前正式结果主表：

| Method | PickXTimes | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | Avg. |
|---|---:|---:|---:|---:|---:|
| No Memory / current-only | 32.0 ± 2.0 | — | — | — | — |
| Ours w/o Teacher (task-specific) | **36.0 ± 2.0** | 26.0 ± 0.0 | 20.7 ± 1.2 | 28.0 ± 0.0 | **27.7 ± 0.6** |
| Ours (task-specific) | **82.0 ± 6.0** | **90.0 ± 7.2** | **92.7 ± 5.8** | **86.0 ± 0.0** | **87.7 ± 3.8** |
| Oracle Memory | 89.3 ± 4.6 | — | — | — | — |

No-Teacher 总计为 166/600，Ours 总计为 526/600。PickXTimes No-Teacher 三个 seed
分别是 19/50、18/50、17/50；三个刷新 region 任务分别为
`[13,13,13]`、`[10,11,10]`、`[14,14,14]`。No Memory 和
Oracle Memory 目前只有 PickXTimes 在同一受控 FrameSamp action 协议下完成，因此
其余单元格不填入跨接口数字。

这张表当前只能支持 **system-level comparison**：

- Pick No-Teacher 使用 action-only goal-conditioned neural RMT，Full 使用
  explicit-event teacher-distilled stack，二者 action 初始化、训练数据、训练预算和评测
  协议匹配，但 updater 结构不完全相同；
- 三个 region 任务现已使用当前 deployable 模型族、observable execution FSM 和相同
  binding correction 重跑；训练仅使用 terminal answer，且不加载 teacher checkpoint、
  不使用 teacher forcing，也不以轨迹指标选点；
- VideoPlaceOrder 的 No-Teacher 为 42/150、Full 为 129/150，下降 58.0 个点；Unmask
  与 Swap 分别下降 64.0 和 72.0 个点；
- 因此 `87.7%-27.7%=60.0 pp` 仍不能写成统一的 loss-only teacher 因果增益，尤其是
  Pick 的 updater 不同，ShellGame 也不属于这张四任务表。

正式结果与逐 seed 计数记录在：

- `../robomme_policy_learning/docs/four_task_closed_loop_3seed50_results_260831.md`
- `../robomme_policy_learning/runs/evaluation/pick-no-teacher-recurrent-3seed50-260901/summary.json`
- `../robomme_policy_learning/runs/evaluation/no-teacher-region-refresh-3seed50-260901/summary.json`
- `../robomme_policy_learning/runs/evaluation/no-teacher-place-corrected-3seed50-260901/summary.json`

## 5. 当前代码对应关系与实施边界

前三个任务当前共享 `pooled_soft_causal` explicit-event region MEM。其 No-Teacher
版本应在相同 recurrent model 上关掉 operation loss、dense table trajectory loss和
teacher forcing，只对 final queried field 反传。

PickXTimes 当前使用 unified semantic-feedback MEM。其 No-Teacher 版本应去掉每个
chunk 的四个 dynamic-field state loss和前 400 步 GT previous-state teacher forcing，
只监督 terminal `completed_count/done`（以及任务答案所需的 terminal field）。

这两个实现遵循同一个 No-Teacher 原则，但目前不是同一个四任务 checkpoint。因此：

- 若论文方法允许 per-task MEM checkpoint，可以直接按上述设计完成主表；
- 若论文声称一个统一四任务 MEM，则应先把 Pick 扩展进同一个 shared recurrent schema，
  再进行 No-Teacher 对照；不能把两个模型家族包装成一个统一 checkpoint。

另外，当前前三任务 3×50 的 action 流程仍由 simulator 提供 online simple phase，MEM
提供 target region；Pick 使用另一套 latent-codebook bridge。论文表格必须在脚注中说明，
或进一步统一 action conditioning，避免把 action bridge 差异归因于 teacher。

## 6. 推荐执行顺序

1. 先做四任务 memory-only、3 个训练 seed 的 Full/No-Teacher 配对实验；
2. 若 No-Teacher 的 terminal accuracy 已明显低于 Full，再各取预注册的 dev-best
   checkpoint 进入闭环；
3. 先做每任务 10-episode smoke，确认接口无错误；
4. 固定配置后完成 3 seeds × 50 episodes，smoke 结果不并入正式统计；
5. 同时补跑 PickXTimes Full 的 3×50，因为当前 Pick 只有 10-episode 结果，不能直接与
   另外三任务的正式数字放在同一主表。

## 7. 预期可解释结论

- 若 No-Teacher terminal memory 和 action success 都显著下降：证明 dense privileged
  trajectory teacher 为 recurrent credit assignment 提供了关键监督。
- 若 terminal memory 接近、但 transition/hold 和 action 下降：证明 teacher 的主要价值
  是学习稳定的在线状态轨迹，而不只是最终答案。
- 若 memory 指标接近且 action 也接近：teacher 不是主性能来源，应降级为训练便利项，
  不能作为核心贡献。
- 若 No-Teacher memory 明显下降但 action 不降：说明当前 action bridge 没有真正利用
  memory，需先修正端到端评测后再讨论 teacher 贡献。
