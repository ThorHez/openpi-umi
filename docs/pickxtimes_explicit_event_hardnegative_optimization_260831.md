# PickXTimes 显式事件 MEM 与 on-policy hard-negative 优化

日期：2026-08-31

## 目标

沿用 VideoPlaceOrder 的核心思路，将 PickXTimes 从直接预测完整 recurrent state 改为：

```text
最近 12 帧 RGB + gripper/command/EEF-Z + 前一离散状态
  -> learned event head（hold / pick / place / press）
  -> deterministic monotonic count updater
  -> canonical latent memory [128, 64]
  -> latent-language codebook wrapper
  -> frozen MME action
```

部署时不读取 simulator event、object pose、contact 或 oracle subgoal。Simulator phase 只用于
训练期 rollout 标签和评测后审计。

## 首先修复的 action 接口错误

最初 runner 错误地启动普通 `mme_vla_suite` 配置和官方
`symbolic-grounded-subgoal/79999` 路径。该配置不会启用 Pick latent codebook 的 language
integration，导致纯 GT latent 也是 0/10。

正确接口为：

- checkpoint wrapper：
  `runs/ckpts/mme_vla_oracle_latent_pick_codebook/oracle_codebook/79999`
- policy config：`mme_vla_oracle_latent_pick_codebook`
- 转换：`latent -> canonical prototype -> simple-SG token ids -> frozen MME action`

修复后，同一 10 条 test episode 的纯 GT latent 上界恢复为 4/10。另一个曾得到 5/10 的
诊断同时传入了 oracle subgoal 文本，属于输入污染，不能作为纯 latent 结果。

## 模型与训练

### 基础显式事件头

固定 train/dev/test 划分，专家训练数据包含：

- train：70 episodes，2928 chunks；
- dev：15 episodes，672 chunks；
- test：15 episodes，672 chunks。

基础模型使用成功演示的 privileged event trajectory 作为训练标签，2400 steps，checkpoint
只按 dev 的 `min(transition, hold, final)` 选择。基础离线 test：

| State | Transition | Hold | Sequence | Final |
|---:|---:|---:|---:|---:|
| 99.70% | 100.00% | 99.65% | 86.67% | 100.00% |

### action-aligned causal window

旧实现按非重叠 12 帧更新，而 action 每 16 步重新规划，事件可能延迟一个 action chunk。
修改后只在 action query 前使用最近 12 帧因果滑窗更新，避免 12/16 帧边界错位。

### train-split on-policy hard negatives

成功演示没有“夹爪闭合/张开但目标物体没有真正完成抓放”的负例。为此在 train split 用冻结
action policy 收集 10 条 rollout：

- 374 action-query windows；
- 42 个 oracle state-changing events；
- 78 个模型或 proprio gate 曾提议更新、但 simulator phase 未变化的 hard negatives；
- 不使用 test rollout 训练。

从基础 checkpoint 微调 600 steps，每个 batch 为 50% 专家数据和 50% rollout 数据；rollout
部分对 hard negative、真实 event、普通 hold 分层采样。模型仍只按原 dev split 选点。

微调后：

| Split | State | Transition | Hold | Final | Update P/R |
|---|---:|---:|---:|---:|---:|
| dev | 99.11% | 97.98% | 99.30% | 100.00% | 93.94% / 93.94% |
| locked test | 91.96% | 91.75% | 92.00% | 100.00% | 88.35% / 93.81% |
| train rollout | 99.73% event | 100.00% recall | 99.70% hold | - | 98.72% hard-negative accuracy |

## 10-episode 闭环消融

统一协议：test split 前 10 条、action seed 7、每条最多 1300 steps，不输入 subgoal 文本。
这些是 smoke/机制消融，不是可写主表的 3 seeds × 50 episodes 结果。

| 条件 | 成功率 | 成功 episode |
|---|---:|---|
| 纯 GT latent + 正确 codebook wrapper | **4/10** | 0, 1, 2, 9 |
| 基础 learned head | 1/10 | 1 |
| 基础 head + 强制 proprio/temporal gate | 2/10 | 1, 9 |
| hard-negative 微调 head | **2/10** | 1, 9 |
| 微调 head + 强制 gate | 1/10 | 9 |
| 微调 head + veto-only proprio/temporal gate | **2/10** | 1, 9 |

强制 gate 会在 learned head 判为 hold 时凭夹爪状态创建事件，绕开 hard-negative 学习，因此不应作为
最终结构。当前保留的合理形式是 learned head 提议、proprio/时序只否决不合法事件；但本轮 10 条
尚未带来额外闭环成功。

## 结论

1. 显式事件瓶颈和确定性 updater 解决了非单调回退，但不能自动解决事件真实性。
2. hard-negative 微调将 10 条闭环从 1/10 提升到 2/10，方向正向，但 10 条 train rollout 的覆盖
   不足以跨布局泛化。
3. 主要剩余 MEM 误差是“夹爪动作发生”与“目标颜色物体真正移动到目标”之间的混淆。仅靠 gripper
   gap、command 和时间间隔不能确认 object binding/transport/place。
4. 同批纯 GT latent 上界仅 4/10，说明即使 MEM 完全正确，当前冻结 action backbone 在该小样本
   上仍有显著低层失败。当前 learned 2/10 已达到该 10 条上界的一半，但不宜开始 3×50 主表评测。

## 下一步

优先做训练数据与目标物体运动验证，而不是继续微调 loss 权重：

1. 在 **train split** 用 3 个 action seeds 收集 30–50 条 rollout，覆盖成功、空抓、抓错物体、提前
   放开和恢复轨迹；保留 dev/test 不参与训练。
2. 给 event head 增加通用的 target-conditioned motion evidence：对目标颜色区域建立 before/after
   correspondence，pick 要求目标随 EEF 抬升，place 要求目标在 release 后稳定落到目标区域。
3. 做三项锁定消融：`expert only`、`+ rollout negatives`、`+ target motion evidence`。
4. 只有当 10–20 条 smoke 达到至少 4/10 并接近 GT-latent 上界后，再运行 3 seeds × 50 episodes。

## 关键产物

- 模型：`src/openpi/tasks/robomme/pickxtimes/explicit_event_count_memory.py`
- 基础训练器与 rollout 微调：
  `scripts/mem/train_pickxtimes_explicit_event_count_memory.py`
- 在线推理：`scripts/mem/robomme_pick_explicit_event_inference.py`
- action bridge：
  `../robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- runner：
  `../robomme_policy_learning/scripts/run_pick_semantic_feedback_action.sh`
- 基础 checkpoint：
  `checkpoints/pickxtimes_explicit_event_count_seed260831_260831`
- hard-negative checkpoint：
  `checkpoints/pickxtimes_explicit_event_onpolicy_hardneg_seed260832_260831`
- hard-negative 训练结果：
  `checkpoints/pickxtimes_explicit_event_onpolicy_hardneg_seed260832_260831/result.json`
- 最终 smoke：
  `../robomme_policy_learning/runs/evaluation/pick-explicit-event-hardneg-headonly-smoke10-260831`
  和
  `../robomme_policy_learning/runs/evaluation/pick-explicit-event-hardneg-veto-gate-smoke10-260831`
