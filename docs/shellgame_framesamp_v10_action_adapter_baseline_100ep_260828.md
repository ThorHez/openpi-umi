# ShellGame MME FrameSamp + V10 action adapter 基线（100 episodes）

日期：2026-08-28

## 1. 实验目标

本实验保留 MME-VLA FrameSamp 的记忆结构，只在 action 路径增加一个接口，
并把 action 模型替换为已经验证过动作执行能力的 V10 absolute-EEF7 模型。
目标是把“记忆选杯能力”和“低层动作执行能力”拆开验证。

模型路径如下：

```text
前 60 帧第三人称图像
  -> frozen MME FrameSamp encoder
  -> 512 x 1024 memory tokens
  -> 新增 ParallelSemanticMemoryActionConditioner
  -> frozen V10 Pi0.5 action expert
  -> absolute EEF7 controller
```

V10 原生 tracker 的 action residual 被关闭：
`old_memory_condition_strength=0`。因此选杯信息只能来自 MME FrameSamp
memory，不会从 V10 原生 60 帧 tracker 泄漏。

## 2. 训练配置

- MME checkpoint：`mme_vla_shellgame_framesamp_modul_v1` step 9999。
- V10 checkpoint：absolute-EEF7 V10 step 1000。
- 数据：V10 nominal absolute-EEF raw7，5000 episodes。
- bank：每局前 60 帧均匀采 32 帧，每帧池化为 16 tokens，最终
  `512 x 1024`，float16，共 5.24 GB。
- bank 标签分布：left/middle/right = `1645/1706/1649`。
- action-interface：16 query tokens，hidden width 256，zero-init residual gate。
- 参数：总计 35.318 亿；冻结 35.268 亿（99.9%）；仅训练新增接口
  499.9 万参数。
- 训练：GPU 4--7，batch 8，500 steps，peak LR `1e-4`。
- episode split：4500 train / 500 held-out validation。

训练期间 `val/action_loss`：

| Step | val/action_loss |
|---:|---:|
| 125 | 0.111759 |
| 250 | 0.098134 |
| 375 | 0.096701 |
| 500 final | **0.081856** |

最终 checkpoint 中 356 个参数叶子完整可恢复；新增接口
`gate_delta=-0.00861`，说明接口已经从零初始化状态学开。V10 的 322 个旧
叶子由严格 loader 恢复，并在训练中保持冻结。

## 3. 闭环评测协议

- 100 个唯一 episode，来自 seed42 的 500-episode adapter held-out 集。
- 目标严格平衡：left/middle/right = `34/33/33`。
- diffusion noise 按 `(salt, episode, query_index)` 固定。
- sampling steps：4；replan interval：8。
- 每局最多 **150 policy steps**。失败局完整执行 19 次重规划，避免动作被评测器
  提前停止。
- 主指标：正确选杯且夹爪在 rollout 中接触正确杯。
- 辅助指标：正确选杯、目标杯接触、任意杯接触、目标杯抬升至少 8 cm。
- 四个并行分片，100 个视频全部保存。

曾先运行过一个 95-step 诊断版；其选杯同样是 29/100，但抬杯只有 5/100。
正式 150-step 版本把抬杯提高到 6/100，因此本文只把 150-step 结果作为正式
基线。

## 4. 100-episode 结果

| 指标 | 结果 | 成功率 |
|---|---:|---:|
| 正确选杯且接触正确杯（主指标） | **29/100** | **29%** |
| 正确选杯 | 29/100 | 29% |
| rollout 中曾接触目标杯 | 41/100 | 41% |
| 接触任意杯 | 100/100 | 100% |
| 正确目标杯抬升至少 8 cm | 6/100 | 6% |
| 到目标杯 60 mm 内 | 32/100 | 32% |
| 到目标杯 30 mm 内 | 29/100 | 29% |

正确选杯率 95% Wilson 区间为 `21.0%--38.5%`。与三选一随机水平比较，
双侧 exact binomial test 为 `p=0.397`，没有显著高于随机。

此前 V10 action-only no-MEM 基线为 35/100 正确选杯且接触。本实验是
29/100，描述性 Fisher exact test 为 `p=0.449`；两者 episode/noise 协议并非
完全相同，因此该检验只作参考。结论是这版方法没有显示出优于 no-MEM 的
选杯收益。

按目标杯分解：

| 目标 | Episodes | 正确选杯 | 目标接触 | 目标抬升 |
|---|---:|---:|---:|---:|
| left | 34 | 8 | 12 | 1 |
| middle | 33 | 13 | 18 | 3 |
| right | 33 | 8 | 11 | 2 |

预测分布 left/middle/right = `29/41/30`，有轻微 middle 偏置，但不足以解释
全部误差。混淆矩阵接近随机：

| GT \\ Pred | left | middle | right |
|---|---:|---:|---:|
| left | 8 | 15 | 11 |
| middle | 9 | 13 | 11 |
| right | 12 | 13 | 8 |

## 5. 原因判断

这次替换 V10 action 后，动作执行问题已经基本排除：

- 100/100 都接触了某个杯；
- 所有 29 个正确选杯 episode 都接触了正确杯；
- 150 steps 下有 6 个 episode 完整抬杯；
- 错误 episode 也完整运行到 150 steps，而不是视频提前结束。

瓶颈在 memory 的最终杯身份表达及其监督方式：

1. 对 frozen `512 x 1024` memory 做全局均值 ridge probe，训练准确率 45.2%，
   held-out 只有 **32.8%**，没有可泛化的线性最终杯信号。
2. action loss 从 0.1118 降到 0.0819，但闭环选杯仍为 29%。说明连续动作回归
   可以通过学习通用接近、下降、闭爪时序而下降，并不保证学到离散杯身份。
3. 正确选杯时接触率 100%，而错误选杯时常稳定到达另一个杯；这与 V10
   低层控制失败的模式不同。

因此第一版结论是：**V10 action head 能修复动作完整性，但不能补救当前
FrameSamp memory 中缺失或难以读取的目标身份信息。** 下一版应在 action
接口前增加显式 final-cup/slot 辅助监督，或先把 FrameSamp memory 的 held-out
三分类 probe 提升到明显高于随机，再训练 V10 action 接口。

## 6. 结果资产

- 正式合并结果：
  `evaluation/shellgame/framesamp_v10_adapter_step499_heldout100_max150_260828/result.json`
- 四个 `shard_*` 目录：逐 episode JSON 与 100 个视频。
- 95-step 诊断结果：
  `evaluation/shellgame/framesamp_v10_adapter_step499_heldout100_260828/result.json`
- 训练 checkpoint：
  `checkpoints/pi0_shellgame_framesamp_v10_action_adapter_eef7_v1/framesamp_modul_step9999_v10_adapter_nominal5000_b8_s500_260828/499`

关键实现：

- `src/openpi/training/mem/recipes/shellgame_framesamp_v10_action_adapter.py`
- `scripts/mem/train_shellgame_framesamp_v10_action_adapter.py`
- `examples/shellgame/serve_v10_exact_parallel_semantic_adapter_deterministic.py`
- `scripts/mem/eval_shellgame_framesamp_v10_action_adapter.py`
- `scripts/mem/merge_shellgame_framesamp_v10_action_eval.py`

