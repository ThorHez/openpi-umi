# VideoUnmaskSwap 非特权执行 FSM 优化与 3 Seeds × 50 Episodes

日期：2026-08-31

## 改动

VideoUnmaskSwap 的交换只发生在 demonstration 中，执行阶段与 VideoUnmask 共用：

`pick target 0 -> putdown -> pick target 1 -> done`

因此复用统一的 RGB + gripper + EEF FSM，不增加任务专用神经网络 head。首轮 smoke 发现一条双目标 episode 已经打开夹爪并释放容器，但 EEF 没有下降到原固定阈值，导致 FSM 永久停留在 `putdown`。新增非特权 release 分支：

- 原分支：EEF 到达释放高度，夹爪打开，EEF 抬离；
- 新分支：夹爪持续完全打开 8 帧；
- 两个分支均经过 phase 最短持续时间和3帧去抖确认。

实现：

- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/execution_phase_fsm.py`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/subgoal_predictor.py`

单元测试与 grounding 测试：`5 passed`。

## Smoke 消融

相同 seed7、相同8条混合难度 episode：

| 设置 | 成功率 |
|---|---:|
| 初始 observable FSM | 5/8 |
| 优化后 observable FSM | 6/8 |
| Simulator oracle phase | 6/8 |

优化后 FSM 在这8条上不再产生相对 oracle phase 的额外损失。两条共同失败 episode 1、2均为交换后目标 region/grounding错误。

## 正式协议

- Task：VideoUnmaskSwap test split
- Action checkpoint：`symbolic-grounded-subgoal/79999`
- MEM checkpoint：`robomme_explicit_event_native_single_seed260908_260831`
- Grounding：semantic assignment + unique anchors
- Execution phase：observable RGB + gripper + EEF FSM
- Seeds：7、17、27
- 每个 seed：固定50 episodes，最多1300步
- 总计：150 rollouts

为规避 SAPIEN/Vulkan 连续创建第25个环境时的 renderer 资源问题，每个 seed 固定分为 `0–23`、`24–47`、`48–49`。所有 episode 只运行一次，失败不重跑。

结果目录：

`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/observable-fsm-open-release-videounmaskswap-3seed50-260831`

## 正式结果

| Seed | Success | Rate |
|---:|---:|---:|
| 7 | 37/50 | 74.0% |
| 17 | 37/50 | 74.0% |
| 27 | 38/50 | 76.0% |
| Mean | 37.33/50 | **74.67%** |

- 跨 seed population std：0.94 percentage points；
- 跨 seed sample std：1.15 percentage points；
- pooled：112/150 = 74.67%。

按目标数分解：

| Setting | Success | Rate |
|---|---:|---:|
| 单目标 | 63/81 | 77.78% |
| 双目标 | 49/69 | 71.01% |

历史旧版 causal-event MEM + oracle phase 三个 seed 为70%、68%、72%，平均70.0%。当前完整非特权系统为74.67%，高4.67个百分点；但由于 MEM checkpoint、semantic assignment、unique anchors 和 execution FSM 同时不同，这个差值只能作为完整系统对比，不能全部归因于 release FSM。

## 失败归因

38个失败中：

- 35个由环境直接判定为 `fail`，表现为拾取了错误容器；
- 3个为 `done` 后 timeout，属于完成检测假阳性；
- **0个停留在 `putdown`**。

这证明持续开夹爪 release 分支解决了本轮针对的 putdown phase 问题。剩余主要瓶颈转移到 demonstration 交换后的实体-region绑定及视觉 grounding，而不是执行阶段切换。

## 论文使用建议

主表可写为 `Ours: Recurrent MEM + Observable FSM + MME Action`，VideoUnmaskSwap 为 **74.67 ± 0.94%**。必须注明推理只使用 RGB、gripper和EEF；训练期 teacher 与推理期权限分开描述。

下一步优先对三 seed 一致失败的 episode 做交换事件绑定分析，而不是继续放宽 FSM 阈值。
