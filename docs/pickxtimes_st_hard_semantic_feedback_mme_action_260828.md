# PickXTimes straight-through hard semantic feedback 与 MME action 实验

日期：2026-08-28

## 实验问题

1. straight-through (ST) hard semantic feedback 是否稳定优于现有 soft semantic feedback？
2. 选出的 recurrent MEM 能否通过已验证的 canonical latent codebook，直接驱动官方 MME action backbone？

除反馈离散化外，hard/soft 使用相同数据、70/15/15 划分、loss、batch size、学习率、400 步
teacher forcing + 600 步 free rollout 和 checkpoint 选择规则。每种方法均报告 4 个种子的锁定 test，
不按 test 挑种子。

## ST-hard 实现

每个字段前向先执行 `argmax`，再编码为固定 hard logits；反向使用 identity straight-through
gradient：

```text
hard_logits = one_hot_logits(argmax(soft_logits))
feedback_logits = soft_logits + stop_gradient(hard_logits - soft_logits)
```

hard 的是语义 embedding 输入和下一步 residual update 的状态基线，不只是 readout。

## 离线结果

指标均为 15 条锁定 test 的 dynamic fields free rollout。

| 模式 | State | Transition | Hold | Sequence | Final |
|---|---:|---:|---:|---:|---:|
| soft，4-seed mean | **87.55%** | **89.95%** | **87.16%** | 48.33% | 90.00% |
| ST-hard，4-seed mean | 86.68% | 87.89% | 86.48% | **65.00%** | **95.00%** |

ST-hard 最佳单种子 260875 达到 state 91.70%、transition 92.78%、hold 91.53%、
sequence 93.33%、final 100%，但 hard 的 transition 标准差为 3.04pp，soft 仅 0.85pp。
hard 的四个 checkpoint 均由 dev 规则选在 step 400，说明 hard 前向消除了 teacher-forcing 输入分布差，
但后续 free-rollout 反传没有稳定提升。

预先锁定的选择标准是最大化四种子平均 `min(transition, hold)`，同时避免 sequence 明显下降。
hard 的平均 transition 低 2.06pp、hold 低 0.68pp，且种子方差更大，因此 action 主实验不采用
最佳 hard 单种子，回退到 soft。action 使用 soft seed 260871，其 checkpoint 由 dev 指标自动选择，
不是根据 test 选择。

## MEM 到 action 的接口

使用已有上界验证过的接口：

```text
12-frame RGB + 6-D proprio
  -> soft recurrent semantic feedback MEM
  -> 6-field constrained Pick state
  -> canonical [128,64] latent codebook
  -> frozen official MME symbolic-grounded action backbone
```

六个字段为 target color、required count、completed count、holding、ready、done。部署请求不发送
`simple_subgoal` 或 `grounded_subgoal` 文本；simulator oracle 只写入 trace 做事后审计，不参与
MEM 或 action 输入。合法状态使用 predicted logits 做 constrained MAP，而不是读取 GT 状态。

在线 proprio 与训练保持相同布局：双指状态、是否闭合、上一动作 gripper command、观测 EEF Z、
commanded EEF Z。超过训练的 96 个 chunk 时冻结最后 causal state，使失败轨迹由环境正常报告 timeout。

## 10 条闭环结果

固定验证集前 10 条、action seed 7、同一 MME checkpoint：

| Memory/action 条件 | 成功率 |
|---|---:|
| official exact-oracle simple SG（已有上界） | 7/10 = 70% |
| GT latent codebook + MME action（已有上界） | 6/10 = 60% |
| **soft recurrent semantic MEM + latent codebook + MME action** | **4/10 = 40%** |

成功 episode 为 1、4、8、9；覆盖 1 次、2 次和 3 次 Pick，以及 red/green/blue 目标。
失败 episode 为 0、2、3、5、6、7，其中 episode 2、5 为 timeout。

按 simulator oracle subgoal 事后还原合法 6-field state，在线 416 个 action-query 时刻的 semantic-key
exact 为 233/416 = 56.01%。分组后：

- 成功 episode：118/131 = 90.08%；
- 失败 episode：115/285 = 40.35%。

这说明 action success 与在线语义正确率强相关，MME action 接口不是当前主要瓶颈。

## 失败机制与结论

离线 test 约 90% transition、87% hold，但闭环在线 exact 只有 56%。主要错误为：

1. **提前推进**：真实环境尚未完成 pick/place，MEM 已增加 count 或输出 ready/done，action 因而过早
   进入下一阶段或 press。
2. **非单调回退**：已经正确到达更高 count/ready 后，又被后续异常视觉或 proprio 窗口拉回旧状态。
3. **action-induced distribution shift**：抓取或放置偏离演示后，模型未见过的轨迹导致 holding/readout
   抖动，错误 memory 又进一步改变动作，形成闭环正反馈。

最终结论：ST-hard 是有价值的消融，尤其提升 sequence consistency，但四种子主指标与稳定性不足以
替代 soft。soft MEM 已能真实驱动 MME action 并达到 4/10，证明 latent 接口可行；但尚未达到可稳定
接入 action 的状态。下一优先级应是直接解决闭环 semantic trajectory 的不可逆性与分布偏移，而不是
继续微调离线 readout loss：训练时加入 action-rollout / recovery 轨迹，并对 count、done 施加通用的
合法转移约束或 transition hysteresis，再复测相同 10 条。

## 关键产物

- 模型：`src/openpi/tasks/robomme/unified_semantic_feedback_student.py`
- 训练器：`scripts/mem/train_pickxtimes_semantic_feedback_student.py`
- 在线推理：`scripts/mem/robomme_semantic_feedback_inference.py`
- soft checkpoint：`checkpoints/pickxtimes_unified_semantic_feedback_seed260871_260828/best/params`
- hard checkpoints：`checkpoints/pickxtimes_unified_hard_semantic_feedback_seed260872_260828/`
  至 `seed260875_260828/`
- action 结果：`../robomme_policy_learning/runs/evaluation/semantic-feedback-mme-action-260828/`
  `semantic-feedback-soft-seed260871-val10-final/ckpt79999/seed7/semantic_feedback_latent/`

