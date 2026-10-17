# PickXTimes / VideoUnmask 单任务 MEM Action Smoke Test（2026-08-27）

## 目标

验证单任务 fixed-chunk MEM 的离线提升能否转化为闭环 action 收益。实验不访问冻结 test，不用 oracle memory 替换预测 memory，并把 memory、target reach、grasp/lift 和最终任务成功分别统计。

## 接口实现

### PickXTimes

- MEM：`checkpoints/robomme_single_task_pick_equal_exposure_seed260827_260827/best/params`。
- 在线更新：从 reset 开始每 12 帧进行一次非重叠 fixed-chunk update；无滑动窗口、无显式 event trigger。
- Action 接口：直接向 action adapter 输入 `[128,64]` memory tokens。
- 为避免把新 latent 输入旧 adapter 的坐标系错配，重新生成 70 train + 15 dev 的 frozen-memory action cache，并重新训练 8-step spatial action chunk adapter。
- 新 adapter 最佳 dev checkpoint 为 step 500：chunk position MAE 1.51 cm、first-step position MAE 1.12 cm、gripper accuracy 96.8%、phase accuracy 83.4%。
- 闭环：固定 dev episodes 10 条，每条最多 500 simulator steps，只运行 predicted-memory 分支。

### VideoUnmask

- MEM：`checkpoints/robomme_single_task_unmask_equal_exposure_seed260827_260827/best/params`。
- 使用 reset 提供的完整可变长度 demonstration，按 12 帧非重叠 chunk 推理。
- 共享 readout 解码目标颜色对应的 `*_cell`，得到 `region_i`；再按演示首帧的 row-major candidate 顺序把 region 映射为二维 target point。
- Action expert 仍是已有 point-conditioned Pi0.5，不使用 prompt color 直接选择 oracle point。
- 闭环：val 的前 10 条单目标 episodes，每条最多 160 steps；同时运行相同 episodes 的 oracle-point control。

## 结果

### PickXTimes：10 条有效 rollout

| Metric | Result |
|---|---:|
| Valid episodes | 10 / 10 |
| Rollout steps | 500 / episode |
| First pick completed | **0 / 10** |
| Full task success | **0 / 10** |
| MEM final `completed_count >= required_count` | 8 / 10 |
| MEM final `done=1` | 7 / 10 |

所有 10 条 episode 的 simulator oracle subgoal 最终仍是“第一次 pick”，但 MEM 在 8 条中已经自报达到所需计数，并在 7 条中输出 `done=1`。例如：

- episode 2：真实状态仍为第一次 pick；required=2；MEM 输出 completed=2、done=1。
- episode 4：真实状态仍为第一次 pick；required=3；MEM 输出 completed=3、done=1。
- episode 29/70：required=1；MEM 都输出 completed=2、done=1。

这不是 latent/action adapter 坐标错配：adapter 已使用同一个新 MEM checkpoint 生成的 action cache 重新训练。主要问题是闭环策略没有完成动作，而 fixed-rate updater 在持续 on-policy 失败画面上累积漂移，产生了虚假的 pick/place/press 进度。

### VideoUnmask：10 条 paired rollout

| Target source | Region/target correct | Reach | Grasp | Lift | Full success |
|---|---:|---:|---:|---:|---:|
| Single-task MEM | **3 / 10** | 3 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Oracle point | 10 / 10 | **8 / 10** | **1 / 10** | **1 / 10** | 0 / 10 |

进一步拆分：

- MEM region 正确时 reach `2/3`。
- MEM region 错误时，对真实目标 reach `1/7`。
- MEM 分支有 1 条环境 IK error；oracle 分支没有 evaluator/environment error。
- oracle point 已能把 reach 提升到 80%，但仍只有 10% grasp/lift、0% 完整成功，因此 VideoUnmask 当前同时存在 memory target error 和 action grasp error。

本次使用 val10，而前一轮 40% final memory accuracy 来自锁定 test15，二者不能直接当成同一估计；val10 的 region exact 为 30%，方向一致但样本量仍小。

## 结论

### 1. 两个 MEM 都已经完成 action 接口接通，但尚未达到可用于正式 action 成功率实验的状态

- PickXTimes 的核心阻塞是 **on-policy memory hallucination + action 无法完成第一次 pick**。离线 80% final 并不代表闭环可用。
- VideoUnmask 的 memory 确实影响 action：region 正确时明显更容易 reach；但是当前 region exact 只有 30%，且 action expert 即使用 oracle point 也不能稳定 grasp。

### 2. 现在不能把 0% action success 归因于 MEM 一项

- Pick 已通过 matched-latent adapter 排除了最明显的接口分布错配，但 action controller 与 on-policy MEM 都失败。
- VideoUnmask 的 paired oracle control 清楚表明：oracle target 能解决大部分 reach，却不能解决 grasp/lift。

## 下一步优先级

1. **Pick MEM 先做 on-policy no-progress hard-negative 训练**：收集当前 10 条失败 rollout 的连续画面，监督 memory 在没有真实 subgoal 完成时保持初始状态；提高 keep/no-change loss，抑制小 gate 在 40+ chunks 上累积成假事件。继续保持 fixed chunk，不重新引入 event trigger。
2. **Pick action 单独做 oracle-state/oracle-memory first-pick 诊断**：如果 oracle memory 下仍是 first-pick `0/10`，先修 action，不继续消耗 MEM 训练预算。
3. **VideoUnmask 先提高 region exact**：按 region/color 平衡单任务数据并增加位置 counterfactual；目标至少达到 val/test region exact 70% 后再做正式闭环。
4. **VideoUnmask action 改善 grasp**：oracle-point 已证明 reach 80%，下一步应优先尝试 delta EEF / waypoint + oracle-gripper 诊断，而不是继续修改 MEM-action 接口。

## 产物

- Fixed-chunk 推理封装：`scripts/mem/robomme_fixed_chunk_inference.py`
- Pick memory-action cache 生成：`scripts/mem/cache_robomme_fixed_chunk_pick_action.py`
- Pick 新 action cache：`data/robomme_extracted/pickxtimes_fixed_chunk_single_mem_action_train70_dev15_stride2_260827.h5`
- Pick action adapter：`evaluation/robomme/pickxtimes_single_fixed_chunk_mem_action_260827/spatial4x4_seed260827_600/`
- Pick smoke shards/retries：`evaluation/robomme/pickxtimes_single_fixed_chunk_mem_action_260827/`
- VideoUnmask predicted/oracle smoke：`evaluation/robomme/videounmask_single_fixed_chunk_mem_action_260827/`

