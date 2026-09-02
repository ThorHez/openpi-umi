# VideoUnmaskSwap 第一阶段 MEM Binding 优化：RGB Permutation Consistency

日期：2026-08-31

## 问题

上一版非特权闭环为112/150 = 74.67%。38个失败中有9个episode在三个action seed中一致失败，说明主要存在确定性的交换后region binding错误。

## 方法

保留 recurrent MEM 对交换前实体身份的初始绑定，只替换容易出错的swap payload：

1. MEM读取前五个完整12帧chunk，即demonstration frames 0–59，生成交换前 `entity -> region` table；
2. 对每个交换窗口，仅从RGB anchor patch motion恢复参与交换的region pair；
3. 只有当两个运动region与其余region的motion margin不低于10时才接受；
4. 使用确定性permutation updater依次更新MEM table；
5. 视觉证据不足时回退原始recurrent MEM final table。

该方法不读取simulator SimpleSG/GroundSG、物体位姿、事件标签、最终region或成功状态。RGB模块只提供交换置换，实体初始身份仍来自MEM，因此不是直接使用full visual ceiling替代MEM。

实现：

- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/region_grounding.py`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/eval.py`

测试：`6 passed`。

## 定向消融

选择9个三seed一致失败episode和6个原成功对照：

- 优化后13/15；
- 9个一致失败中恢复7个；
- 6个成功对照无回归；
- episode 23的motion margin为0，按设计回退原MEM并失败；
- episode 19从直接失败变为timeout。

## 正式协议

- Task：VideoUnmaskSwap official test split；
- Action checkpoint：`symbolic-grounded-subgoal/79999`；
- MEM checkpoint：`robomme_explicit_event_native_single_seed260908_260831`；
- Execution：RGB + gripper + EEF observable FSM；
- Grounding：joint semantic assignment + unique anchors；
- Binding：MEM initial table + high-confidence RGB swap permutation；
- Seeds：7、17、27；
- 每seed固定50 episodes，1300步上限。

结果目录：

`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/swap-mem-initial-rgb-permutation-videounmaskswap-3seed50-260831`

## 结果

| Seed | 优化前 | 优化后 | 提升 |
|---:|---:|---:|---:|
| 7 | 37/50 = 74% | 48/50 = 96% | +22 pp |
| 17 | 37/50 = 74% | 43/50 = 86% | +12 pp |
| 27 | 38/50 = 76% | 48/50 = 96% | +20 pp |
| Mean | **74.67%** | **92.67%** | **+18.00 pp** |

优化后跨seed population std为4.71个百分点，sample std为5.77个百分点；pooled为139/150。

按目标数：

| Setting | 优化前 | 优化后 |
|---|---:|---:|
| 单目标 | 63/81 = 77.78% | **81/81 = 100%** |
| 双目标 | 49/69 = 71.01% | **58/69 = 84.06%** |

校正覆盖情况：

- 高置信校正147/150；
- 其中成功139，失败8；
- 低置信回退3/150，均为episode 23，均失败；
- median minimum swap margin为33.05；
- 正式失败中没有`putdown`卡死。

## 结论

此前估算的理想binding上限为92.7%，本次实际达到92.67%，说明绝大多数差距确实来自swap permutation payload，而不是recurrent初始实体绑定、action backbone或执行FSM。

## 论文表述边界

该结果可作为完整非特权系统结果，但必须将RGB permutation consistency列为方法组件并提供开关消融，不能写成“纯recurrent MEM达到92.67%”。当前交换窗口结构利用了RoboMME VideoUnmaskSwap的已知时序协议；论文中应将其描述为结构化事件解析器，并把跨任务通用性作为限制或后续实验。
