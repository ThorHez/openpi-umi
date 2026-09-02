# PickXTimes Event/Hold 双更新路由实验（2026-08-27）

## 结论摘要

独立双更新路由已实现，并完成 4 组 400-step 受控实验。结构能够：

- 从旧 checkpoint 严格等价启动；
- 保留原 write gate 的积分步长；
- 学出 event/hold 不同的更新残差；
- 通过温度把事件与 far-hold 的路由概率明显拉开。

但所有实验的全局最佳 checkpoint 仍为 step 0，训练后的最佳 transition accuracy 最高为 42.42%，未超过原模型的 43.43%。因此当前结果不支持把双更新路由接入 action 主实验。

## 结构

原共享 updater 先生成 `candidate`，独立 event gate 输出事件概率。新增两个零初始化残差分支：

```text
event_residual = EventUpdateAdapter(candidate - memory)
hold_residual  = HoldUpdateAdapter(candidate - memory)

routing_probability = temperature_scale(event_gate)
routed_residual = routing_probability * event_residual
                + (1 - routing_probability) * hold_residual

candidate_routed = candidate + routed_residual
memory_next = memory + write_gate * (candidate_routed - memory)
```

两个 adapter 的输出层均使用全零 kernel/bias 初始化，因此第一次加载旧 checkpoint 时：

```text
event_residual = hold_residual = routed_residual = 0
candidate_routed = candidate
```

单元测试和全尺寸 dev rollout 都验证了初始 memory trajectory 不发生漂移。

新增可训练参数共 8,448 个；实验阶段冻结 visual encoder、recurrent updater、readout、write gate 和 event gate，只训练 `route_*`。

## 对照设置

共同设置：

- Task：PickXTimes；
- 输入：固定 12 帧、非重叠 chunk；
- Resume：`checkpoints/pickxtimes_dual_gate_event_head_seed260829_260827/best/params`；
- 400 steps，batch size 4，seed 260829；
- memory loss 1.0，state loss 0.5，change weight 10，final weight 4；
- 每 25 steps 在固定 15 条 dev episodes 上验证；
- checkpoint objective：final、transition、sequence、all-state 的字典序。

## 结果

基线 step 0：Final 100.00%，Transition 43.43%，Hold 41.33%，All-state 41.63%。

下表报告每组 **step > 0 的最佳 checkpoint**；四组包含 step 0 时的全局 best 均为 step 0。

| 路由 | LR | post-train best step | Final | Transition | Hold | All-state |
|---|---:|---:|---:|---:|---:|---:|
| Soft，T=1.0 | 3e-4 | 350 | 100.00% | 40.40% | 39.29% | 39.45% |
| Soft，T=1.0 | 1e-4 | 250 | 100.00% | **42.42%** | 38.95% | 39.45% |
| Sharpen，T=0.5 | 1e-4 | 250 | 100.00% | **42.42%** | 39.29% | 39.74% |
| Sharpen，T=0.25 | 1e-4 | 200 | 100.00% | **42.42%** | 39.46% | **39.88%** |

锐化和分支学习确实发生：

| 设置 | Routing prob / change | Routing prob / far hold | Event residual norm | Hold residual norm |
|---|---:|---:|---:|---:|
| T=0.5 best post-train | 0.654 | 0.420 | 0.0773 | 0.0659 |
| T=0.25 best post-train | 0.722 | 0.429 | 0.0691 | 0.0578 |

这说明负结果不是因为代码没有启用路由或两个分支完全没有分化。分化后的更新内容仍然降低 hold/all-state，并且没有把 transition 提升到基线之上。

## 归因

1. Event gate 只有约 0.683 AUROC，错误路由会被递归累积；内容路由虽然比乘法调 gate 安全，但仍会向 memory trajectory 注入噪声。
2. 当前 canonical loss 对 event/hold 两个 adapter 没有直接的角色约束。即使温度锐化，两个分支仍同时由 episode-level memory/readout loss 间接训练。
3. 事件后的 state label 离散，而 adapter 修改的是 128 个 latent token。仅 8,448 个 route 参数和 15 条 dev episodes 下，latent 改善未稳定映射为 exact-state 改善。
4. 结果与上一轮乘法 gate 实验一致：保留原 recurrent trajectory 比让当前 event detector 主动干预更可靠。

## 决策

- 保留代码和 `strength=0` 的独立 event confidence 输出，用于分析和未来辅助 loss；
- PickXTimes 当前部署仍使用无调制、无 update routing 的 memory checkpoint；
- 不把本轮 route checkpoint 接入 action smoke test；
- 下一步若继续优化，应先提高 event detector 的 oracle-window AUPRC/AUROC，或用 privileged state transition 直接监督 event/hold residual 的目标 latent delta，而不是继续调 routing temperature/LR。

## 产物

- Soft T=1.0, LR 3e-4：`checkpoints/pickxtimes_dual_update_route_lr3e4_seed260829_260827`
- Soft T=1.0, LR 1e-4：`checkpoints/pickxtimes_dual_update_route_lr1e4_seed260829_260827`
- T=0.5, LR 1e-4：`checkpoints/pickxtimes_dual_update_route_t05_lr1e4_seed260829_260827`
- T=0.25, LR 1e-4：`checkpoints/pickxtimes_dual_update_route_t025_lr1e4_seed260829_260827`

## 验证

- 单元测试：9 passed；
- Python compile：通过；
- `git diff --check`：通过；
- 测试覆盖路由关闭、零初始化严格等价、padding 冻结和温度锐化。
