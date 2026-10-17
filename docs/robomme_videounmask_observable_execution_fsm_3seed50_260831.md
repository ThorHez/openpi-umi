# VideoUnmask 非特权执行 FSM：3 Seeds × 50 Episodes

日期：2026-08-31

## 目的

把此前闭环评测中的 simulator SimpleSG phase oracle 替换为非特权执行阶段 FSM。语义 MEM 仍负责从 demonstration 推断目标 region；FSM 在线只读取：

- 前视 RGB；
- observation 中的 gripper state；
- 机器人自身的 EEF Z。

FSM 不读取 simulator SimpleSG/GroundSG oracle、物体位姿、接触标志、任务阶段索引或成功标志。

## 实现

- FSM：`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/execution_phase_fsm.py`
- 闭环接入：`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- CLI：`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/eval.py`
- 启动器：`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/scripts/run_region_grounding_three_task.sh`

状态为 `pick -> putdown -> pick -> done`。目标顺序来自 task prompt，目标位置来自 MEM 的 `region_by_color`。拾取证据是夹爪闭合、EEF 抬升以及目标 anchor 的 RGB/可见容器变化；放下证据是到达释放高度、夹爪打开并抬离。双目标在放下完成后切换到第二个 MEM region。

单元测试覆盖单目标完成和双目标完整转移，结果为 `2 passed`。

## 评测协议

- Task：VideoUnmask test split
- Action checkpoint：`symbolic-grounded-subgoal/79999`
- MEM checkpoint：`robomme_explicit_event_native_single_seed260908_260831`
- Grounding：semantic assignment + unique anchors
- Seeds：7、17、27
- Episodes：每个 seed 固定 50 条
- 最大步数：1300
- 总 rollout 数：150

结果根目录：

`/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/observable-fsm-videounmask-3seed50-260831`

由于 SAPIEN/Vulkan 每个评测进程连续创建第 25 个环境时出现 `ErrorIncompatibleDriver`，每个 seed 通过同一 `progress.json` 断点续跑。已完成 episode 不重跑，模型失败照常保留，因此不改变评测样本或统计口径。

## 结果

| Seed | Success | Rate |
|---:|---:|---:|
| 7 | 49/50 | 98.0% |
| 17 | 44/50 | 88.0% |
| 27 | 42/50 | 84.0% |
| Mean | 45/50 | **90.0%** |

- seed rate population std：5.89 percentage points；
- seed rate sample std：7.21 percentage points；
- pooled success：135/150 = 90.0%。

按目标数分解：

| Setting | Success | Rate |
|---|---:|---:|
| 单目标 | 113/114 | **99.12%** |
| 双目标 | 22/36 | **61.11%** |

## 失败归因

共 15 个失败：

- 8 个停在 `putdown`：已正确识别并拾取第一个目标，但 action 没有把 EEF 降到 FSM 的释放高度，因此 FSM 没有切换到第二个 region；
- 7 个进入 `done`，但环境未成功：RGB anchor 变化、闭合夹爪和抬升 EEF 的联合证据仍可能把遮挡或失败抓取判断为完成。

几乎全部损失来自双目标：单目标仅失败 1/114，双目标失败 14/36。

同一 MEM、grounding 和 action checkpoint 在 simulator oracle phase 条件下，三个 seed 均为 50/50。因此本次从 100% 到 90% 的增量损失主要由执行阶段估计及其与 action 的闭环耦合造成，不能归因于 demonstration region MEM 本身。

## 结论

非特权 FSM 已经能可靠处理单目标，但尚不能替代双目标任务中的 oracle phase。当前 90.0% 可以作为严格非特权闭环结果，但主表必须明确标记为 `RGB + gripper + EEF execution FSM`，不能与此前 oracle-phase 100% 混写。

下一步优先级应是放下阶段，而不是继续修改 MEM：增加“持续打开夹爪 + RGB 中容器已释放/稳定”的非特权 release 分支，并对 `putdown` 检测做独立消融；随后再改善完成检测的遮挡鲁棒性。
