# RoboMME Fig. 2 核心 memory-token 消融（3 seeds，fixed step 2000）

日期：2026-09-01

## 目的

补做与 Fig. 2 和 `src/openpi/models/siglip_mem_semantic.py` 真正一致的核心消融。此前的统一诊断携带的是额外 64D causal-evidence hidden state，并未检验 `128×64` memory tokens 本身的递归必要性。本实验明确关闭该 hidden state，也不使用滑动窗口或显式 event trigger。

核心更新为：

```text
M_i = M_{i-1} + alpha_i * (U(M_{i-1}, z_i) - M_{i-1})
```

其中 `M_i ∈ R^(128×64)`，`U` 直接调用 `siglip_mem_semantic.MemoryUpdateBlock`。每个 seed 的一个统一 checkpoint 同时服务四项 RoboMME 任务。

## 严格匹配协议

- 数据、visual encoder、goal initializer、teacher readout、batch size、optimizer和训练步数完全一致；
- 连续、不重叠的 12 帧 causal chunks；
- seeds：260951、260952、260953；
- 表中全部使用固定 step 2000 checkpoint，不使用 best-checkpoint 选择；
- held-out test：60 episodes；
- 所有 A/B/C/D 均由当前同一训练入口重新训练；
- inference 不接收 GT event、state、boundary 或 gate 标签。

四个条件：

1. **A — token carry + soft gate**：递归携带完整 `128×64` memory，学习标量 `alpha_i`；
2. **B — reset token + soft gate**：每个 chunk 都从 goal-initialized `M_0` 更新，禁止输出 token carry；
3. **C — token carry + unconditional write**：保留完整 token carry，固定 `alpha_i=1`；
4. **D — A without trajectory teacher**：结构与 A 相同，但仅监督最终 queried answer，中间 teacher memory/state 只用于评测。

## 主结果

数值为 3 个训练 seed 的 mean ± sample SD（%）。

| Variant | Field | State | Change | Hold | Final | Answer | Sequence |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: token carry + soft gate | **90.2 ± 0.1** | **46.4 ± 1.1** | **23.5 ± 0.9** | **50.0 ± 1.3** | 12.2 ± 6.9 | 18.3 ± 6.0 | **2.2 ± 2.5** |
| B: reset token + soft gate | 82.8 ± 0.4 | 26.8 ± 1.3 | 15.0 ± 4.4 | 28.6 ± 1.7 | **17.2 ± 5.9** | 25.0 ± 7.3 | 0.0 ± 0.0 |
| C: token carry + unconditional write | 82.0 ± 0.8 | 31.6 ± 1.5 | 11.8 ± 2.2 | 34.6 ± 1.5 | 7.8 ± 3.5 | 18.9 ± 8.4 | 1.1 ± 1.9 |
| D: A w/o trajectory teacher | 27.5 ± 2.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **45.0 ± 1.7** | 0.0 ± 0.0 |

配对差值：

- **完整 token carry 的作用（A−B）**：State `+19.65 ± 0.29`，Change `+8.48 ± 3.49`，Hold `+21.36 ± 0.41`；三个 seed 的方向一致。
- **soft gate 的作用（A−C）**：State `+14.89 ± 2.53`，Change `+11.76 ± 1.44`，Hold `+15.37 ± 2.74`；三个 seed 的方向一致。
- **trajectory teacher 的作用（A−D）**：State `+46.44 ± 1.09`，Change `+23.51 ± 0.93`，Hold `+49.95 ± 1.32`。

但是，A 相对 B 的 Final 为 `−5.00 ± 4.41`，Answer 为 `−6.67 ± 1.67`。因此本实验只证明核心 token recurrence 改善了**稠密状态轨迹和保持**，不能声称它改善最终 query，更不能单独作为 action-ready 证据。D 的 Answer 很高而完整轨迹为零，说明 terminal-only 模型学习了 endpoint shortcut。

## 视觉因果干预（A）

| Input | State | Change | Final | Answer | Mean gate |
|---|---:|---:|---:|---:|---:|
| Normal | 46.4 ± 1.1 | 23.5 ± 0.9 | 12.2 ± 6.9 | 18.3 ± 6.0 | 0.019 ± 0.001 |
| Zero video | 36.7 ± 0.2 | 20.4 ± 2.4 | 15.0 ± 4.4 | 22.8 ± 2.5 | 0.029 ± 0.002 |
| Reverse chunks | 45.5 ± 1.1 | 22.9 ± 1.8 | 14.4 ± 6.9 | 22.8 ± 5.4 | 0.020 ± 0.001 |
| Within-task episode shuffle | 45.0 ± 1.1 | 22.8 ± 1.0 | 11.7 ± 7.6 | 16.7 ± 8.8 | 0.019 ± 0.001 |

这组干预否定了“核心 A 已经学到强视觉因果状态机”的解释：

- Zero video 只使 State 下降 9.7 点，Change 下降 3.1 点；
- Reverse chunks 只使 State 下降 0.9 点；
- Answer 在 Zero/Reverse 下反而更高，不能作为视觉因果证据；
- gate 收敛到约 0.019，更像共享 updater 的小步长积分器，而非事件概率。

最合理解释是：固定 chunk 数、goal、encoder bias/相对位置输出和共享递归更新允许模型形成 **elapsed-step / task-prior clock shortcut**。因此 A−B 的正向稠密轨迹结果是真实的 recurrence 效应，但其中相当一部分是“递归保存并积分时间先验”，而不是“从 RGB 稳定解析事件后更新语义状态”。

## 论文可写结论

可以写：

> 在严格匹配的 3-seed 消融中，递归携带完整 `128×64` memory tokens 相对逐窗重置将全状态准确率提高 19.6 点，soft gate 相对无条件写入再提高 14.9 点；两项增益在三个 seed 上方向一致。去掉轨迹教师后完整状态轨迹归零，尽管终点查询仍可通过 shortcut 达到 45.0%。

必须紧接限制：

> 然而 zero-video 和 reversed-order 干预只造成有限下降，且 token recurrence 未提高最终 query。这表明当前统一 recipe 的 recurrent gain 部分来自 elapsed-step prior，而非充分的视觉事件 grounding；核心 recurrence 已被证明有助于轨迹保持，但其视觉因果性和 action readiness 尚未成立。

## 产物

- 汇总 JSON：`checkpoints/robomme_core_memory_token_ablation_3seed_step2000.json`
- 训练脚本：`scripts/mem/run_core_memory_token_ablation_3seed.sh`
- 干预脚本：`scripts/mem/run_core_memory_token_interventions_3seed.sh`
- 汇总/合同审计：`scripts/mem/summarize_core_memory_token_ablation.py`
