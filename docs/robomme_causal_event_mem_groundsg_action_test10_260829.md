# Causal-event recurrent MEM 接入 GroundSG action 的闭环实验（2026-08-29）

## 结论

将 `pooled_soft_causal` recurrent MEM 接到 RoboMME 官方
`symbolic-grounded-subgoal/79999` action checkpoint 后，在同一批 test episode、
seed 7、每条最多 1300 步的协议下得到 **25/30（83.3%）**：

| 记忆输入 | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | 总计 |
|---|---:|---:|---:|---:|
| Oracle semantic region | 10/10 | 10/10 | 10/10 | 30/30（100.0%） |
| 旧 recurrent MEM | 4/10 | 1/10 | 1/10 | 6/30（20.0%） |
| **Causal-event recurrent MEM** | **9/10** | **9/10** | **7/10** | **25/30（83.3%）** |

新 MEM 相对旧 MEM 提升 **+63.3 pp**，填补了旧 MEM 到 oracle 之间
**79.2%** 的闭环差距。三项任务的 oracle gap closure 分别为 83.3%、
88.9% 和 66.7%。

逐 episode 核验表明，动作成败完全由 semantic region 是否正确解释：

- MEM 给出完整且正确 region：25 条，action 成功 25/25；
- MEM region 错误或缺失：5 条，action 成功 0/5。

因此在本实验范围内，官方 action backbone 和 region-to-GroundSG grounding
不是主要瓶颈；剩余的 16.7% 闭环差距来自 MEM 的语义状态错误。

## 实验设置

- 数据划分：RoboMME `test`
- 任务：`VideoUnmask`、`VideoUnmaskSwap`、`VideoPlaceOrder`
- episode：每项前 10 条，共 30 条
- seed：7
- 最大执行步数：1300
- action checkpoint：
  `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999`
- MEM checkpoint：
  `/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/robomme_explicit_event_pooled_soft_causal_seed260908_260829/params.msgpack`
- action 输入：官方 GroundSG 文本接口；semantic region 来自 MEM，目标像素由统一
  region-grounding bridge 生成
- MEM 输入：完整 demonstration 的因果 12 帧 chunks；RGB 被变换为与训练缓存一致的
  grid-8 raw-RGB 特征；不使用 simulator GT、oracle event 或 oracle region
- action phase 仍采用官方 online simple subgoal，用于隔离“记忆目标是否正确”这一变量

运行目录：

`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/causal-event-mem-action-test10-three-task-seed7-260829`

## 与结构消融的联系

接入 action 的模型来自显式事件瓶颈消融中的胜者 `pooled_soft_causal`。相对相同
pooled soft updater、但没有 causal evidence state 的 baseline，三个 seed 的离线均值为：

| 模型 | Event FPR | Full state | Final | Transition | Hold | All-state |
|---|---:|---:|---:|---:|---:|---:|
| pooled soft baseline | 12.00 | 33.11 | 43.65 | 41.47 | 48.17 | 47.27 |
| **+ causal evidence state** | **1.31** | **57.72** | **62.56** | **69.55** | **73.12** | **72.64** |

这说明主要增益不是“换一个更大的 head”，而是把视觉事件解析需要的短期因果证据
从长期语义 memory 中分离出来。事件 head 负责解析当前窗口，确定性 updater 只执行
受约束的状态转移；长期 memory 不再同时承担视觉识别、证据积累和状态存储。

完整离线消融见：
`docs/robomme_explicit_event_bottleneck_factorial_ablation_260829.md`。

## 五条失败的归因

| 任务/episode | 失败类型 | MEM 输出 |
|---|---|---|
| VideoUnmask ep3 | 第二个目标缺失 | green=0，red=-1 |
| VideoUnmaskSwap ep2 | region 选错 | red=0，oracle region=2 |
| VideoPlaceOrder ep2 | 顺序目标选错 | first target=0，oracle region=1 |
| VideoPlaceOrder ep3 | 顺序关系未恢复 | second target=-1 |
| VideoPlaceOrder ep5 | 顺序关系未恢复 | second target=-1 |

错误高度集中在两类问题：多目标完整性，以及 PlaceOrder 的 ordinal binding。
这比继续调 action controller、像素 grounding 或通用 hold loss 更值得优先优化。

## 实现与验证

- 在线 MEM provider：
  `scripts/mem/robomme_explicit_event_inference.py`
- action bridge：
  `robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- launcher：
  `robomme_policy_learning/scripts/run_region_grounding_three_task.sh`
- 对缺失 semantic region 的处理：按该 episode 的模型失败计分，并正常保存 trace，
  而不是误报为 API error 或中断整个评测。
- 在线 raw RGB 特征路径与训练缓存路径的表输出逐元素一致，最大差值为 0。
- `test_region_grounding.py`：5/5 passed；相关 MEM 文件 Ruff 检查通过。

## 下一步

当前结果已经证明这条结构路线可以接入 action。下一阶段应聚焦 MEM，而不是 action：

1. 对 `VideoPlaceOrder` 加入显式 `(object, ordinal, region)` binding 监督，但保持共享
   event bottleneck 和共享 updater，不增加任务专用 action head；
2. 对多目标任务加入 set-completeness loss，惩罚已请求对象被解码为 `-1`；
3. 先做三 seed 的 10-episode 配对复现；结果稳定后再扩到官方 50 episode 协议。

本结果只有一个 action seed、每任务 10 条，适合作为强 smoke/机制证据，暂不应直接作为
论文最终主表数值。

## 三 action seeds 更新

后续补跑 action seeds 17、27，均复现 seed 7 的 Unmask 9/10、Swap 9/10、Place
7/10，总计 25/30。三 seeds 名义合计 75/90，成功率 83.3%，seed 标准差为 0。
详细结果见 `docs/robomme_causal_event_mem_action_three_seed_260829.md`。
