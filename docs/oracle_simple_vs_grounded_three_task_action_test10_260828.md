# Oracle SimpleSG vs GroundSG：三项空间任务 action 上界实验

日期：2026-08-28

## 实验问题

固定官方 `symbolic-grounded-subgoal/79999` action checkpoint，比较 simulator oracle
提供的 SimpleSG 与 GroundSG，验证 VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder 中仅有
高层语义、没有像素 grounding 时，是否足以驱动动作。

## 评测设置

- dataset split：官方 `test`；
- episode：每个任务前 10 条；
- 最大步数：1300；
- action seed：7；
- action checkpoint：`symbolic-grounded-subgoal/79999`；
- 两组唯一差异：`simple_subgoal_oracle` vs `grounded_subgoal_oracle`；
- 三个任务共 30 条，两组共 60 条；无评测错误。

## 结果

| 任务 | Oracle SimpleSG + GroundSG action | Oracle GroundSG + GroundSG action | Grounding收益 |
|---|---:|---:|---:|
| VideoUnmask | 3/10 = 30% | 10/10 = 100% | +70pp |
| VideoUnmaskSwap | 2/10 = 20% | 10/10 = 100% | +80pp |
| VideoPlaceOrder | 2/10 = 20% | 10/10 = 100% | +80pp |
| **总体** | **7/30 = 23.33%** | **30/30 = 100%** | **+76.67pp** |

SimpleSG 成功 episode：

- VideoUnmask：4、8、9；
- VideoUnmaskSwap：1、8；
- VideoPlaceOrder：2、7。

GroundSG 在三任务的 episode 0--9 全部成功。

## 结论

1. 官方 GroundSG action backbone 在这 30 条 episode 上具备完整动作能力；GroundSG oracle
   达到 100%，因此低层 action 不是当前瓶颈。
2. SimpleSG 即使来自 simulator oracle，仍缺少目标容器或目标位置的像素 grounding，在三项
   空间任务上只能达到 20%--30%。因此它不能被解释为完美 memory 上界，只能作为
   `no-grounding` 消融。
3. VideoUnmaskSwap 和 VideoPlaceOrder 不仅需要语义 state，还必须将历史身份、交换轨迹或顺序
   映射到当前图像位置。后续 recurrent MEM 接入 action 时，需要统一的 grounded readout：

   ```text
   recurrent semantic/spatial memory
       -> target identity / historical relation
       -> current-image grounding (x, y or grounded token)
       -> frozen GroundSG action
   ```

4. 下一阶段应分别报告 memory-state accuracy、grounding accuracy 与 closed-loop success，避免
   将正确语义但错误位置的样本计作 memory 正确。

## 结果目录

- SimpleSG：`robomme_policy_learning/runs/evaluation/oracle-simple-vs-grounded-three-task-test10-seed7-260828/simple/`
- GroundSG：`robomme_policy_learning/runs/evaluation/oracle-simple-vs-grounded-three-task-test10-seed7-260828/grounded/`
- 启动脚本：`robomme_policy_learning/scripts/run_oracle_simple_grounded_pick_official.sh`
