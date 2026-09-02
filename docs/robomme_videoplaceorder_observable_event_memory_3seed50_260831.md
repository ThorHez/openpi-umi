# VideoPlaceOrder observable event memory + MME action（2026-08-31）

## 目标

在不读取 simulator SimpleSG/GroundSG、物体位姿、接触、成功标志或最终 region 的条件下，
验证 VideoPlaceOrder 的动作上界，并判断原 causal recurrent MEM 的主要损失是否来自放置事件
边界和 ordinal-region binding。

## 统一闭环协议

- Action：官方 `symbolic-grounded-subgoal/79999`；
- seeds：7、17、27；
- 每个 seed：test episode 0–49，共 50 条；
- 每条最多 1300 simulator steps；
- execution phase：仅使用执行期 RGB、gripper 和 EEF 的 observable FSM；
- region anchor：demonstration RGB；
- 不使用 simulator phase oracle。

## 视觉上界分解

相同 seed7 前 10 条的开发过程如下：

| 条件 | Action success | 结论 |
|---|---:|---|
| demonstration boundary/grounded anchor ceiling | 10/10 | 特权标签上界，不能当作可部署视觉模型 |
| 纯 RGB away→near parser | 6/10 | 初始 source target 被误计为第一次放置 |
| gripper release + RGB region | 8/10 | 能恢复正确事件，但 gripper settling 会重复提交 |
| + same-region release debounce + joint-motion static boundary | 10/10 semantic target；闭环配对恢复 | 非特权事件解析上界成立 |

最终 observable parser 的因果过程是：

```text
sustained gripper close
  -> sustained gripper release
  -> RGB cube-to-target region vote
  -> suppress consecutive releases into the same region
  -> append region to ordered table
  -> infer relocation interval after the last robot joint motion
  -> ordinal readout
```

演示期输入只有 RGB、robot joint state 和 gripper state。执行期 FSM 输入 RGB、gripper、EEF。

## MEM 配对消融

两行使用相同 causal MEM checkpoint、action checkpoint、observable FSM、episodes 和 seeds。
唯一差异是是否用 observable completed-place event 对 ordinal table 做确定性校正。

| 方法 | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| causal MEM + observable FSM | 34/50（68%） | 35/50（70%） | 35/50（70%） | **69.33 ± 1.15%** | **104/150** |
| + observable placement-event consistency | 47/50（94%） | 46/50（92%） | 48/50（96%） | **94.00 ± 2.00%** | **141/150** |
| Gain | +26 pp | +22 pp | +26 pp | **+24.67 pp** | **+37/150** |

配对 episode 的 exact McNemar 统计：

| Seed | correction-only success | baseline-only success | both success | neither | exact p |
|---:|---:|---:|---:|---:|---:|
| 7 | 14 | 1 | 33 | 2 | 9.77e-4 |
| 17 | 13 | 2 | 33 | 2 | 7.39e-3 |
| 27 | 14 | 1 | 34 | 1 | 9.77e-4 |

## 难度分解

合并三个 action seeds：

| Difficulty | Baseline | Event consistency | Gain |
|---|---:|---:|---:|
| Easy | 59/78（75.64%） | 77/78（98.72%） | +23.08 pp |
| Medium | 24/36（66.67%） | 36/36（100.00%） | +33.33 pp |
| Hard | 21/36（58.33%） | 28/36（77.78%） | +19.45 pp |

observable correction 在每个 seed 的 50 条上都成功解析事件，并在 16/50 条改变原 MEM region；
34/50 条与原 MEM 一致。改变 region 的样本中闭环分别成功 13/16、13/16、14/16；未改变的
样本分别成功 34/34、33/34、34/34。

## 剩余失败

- ep11：校正 region 与旧 simulator-oracle audit 一致，但 observable FSM 过早进入 `done`，
  三个 seed 都失败，属于执行阶段判定问题而不是 ordinal readout；
- ep24：region 正确，仅 seed17 失败，属于 action seed sensitivity；
- ep27：两个 seed 失败、一个成功，需要继续区分 hard relocation 与 release timing；
- ep47：train/dev 固定的 motion threshold `19.0` 上出现边界值 `19.43`，把 region1 改为
  region0。不能根据 test 单条重新调阈值，应通过 train/dev 的 margin calibration 或 learned
  confidence 解决。

## 可以和不可以写进论文的结论

可以写：

> A causal, observable placement-event consistency module improves the same recurrent-memory/action
> system from 69.3% to 94.0% on VideoPlaceOrder under a 3-seed × 50-episode protocol.

不能把 94.0% 直接写成“纯 learned recurrent MEM”。当前 correction 在推理期使用确定性的
gripper-release/RGB region parser；它是一个结构消融和可部署 hybrid memory updater。论文方法若要
主张完全 learned MEM，下一步必须把它产生的 `commit boundary + destination region + ordinal table`
轨迹蒸馏给 learned event head，并以当前 hybrid 作为 teacher/upper bound。

## 产物

- observable FSM：
  `robomme_policy_learning/examples/robomme/execution_phase_fsm.py`
- event correction / ordinal updater：
  `robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- flags：
  `region_grounding_place_rgb_only_ceiling`、
  `region_grounding_place_visual_event_correction`
- runner：
  `robomme_policy_learning/scripts/run_region_grounding_three_task.sh`
- correction 正式结果：
  `robomme_policy_learning/runs/evaluation/place-causal-mem-observable-fsm-corr-3seed50-260831/`
- strict baseline 正式结果：
  `robomme_policy_learning/runs/evaluation/place-causal-mem-observable-fsm-base-3seed50-260831/`

评测期间 SAPIEN/Vulkan 在约 24 次环境重建后出现 renderer 初始化错误。每次均在同一目录按
`progress.json` 续跑，基础设施异常没有被计为 episode fail；最终六个条件均包含完整的 50 条结果。
