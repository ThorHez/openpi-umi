# RoboMME ordinal binding × multi-target completeness 消融（2026-08-29）

## 结论

在 `pooled_soft_causal` recurrent MEM 上完成 2×2 消融、每个配置 3 个 seed。
**只有 ordinal binding loss 值得进入 action 测试**；completeness loss 单独收益不稳定，
与 ordinal loss 联用存在明显负交互。

| 配置 | Unmask final episode | Swap final episode | Place final episode | Mean final query | Transition | Hold | All-state |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 64.44±10.18 | 64.44±10.18 | 48.89±3.85 | 62.56±3.49 | 69.55±6.70 | 73.12±8.68 | 72.64±8.39 |
| **ordinal only** | **68.89±7.70** | **71.11±7.70** | **55.56±10.18** | **68.40±5.22** | **69.82±1.98** | 71.87±2.35 | 71.59±2.24 |
| completeness only | 73.33±11.55 | 62.22±15.40 | 46.67±6.67 | 64.79±8.13 | 68.77±7.14 | 72.87±3.76 | 72.33±4.10 |
| ordinal + completeness | 57.78±10.18 | 60.00±6.67 | 51.11±16.78 | 59.80±5.73 | 64.30±3.72 | 71.75±0.92 | 70.75±0.73 |

数值为 test free-rollout 的三 seed 均值±样本标准差，单位为百分比。

## 消融设计

四组模型使用完全相同的：

- causal anchor evidence state；
- explicit event bottleneck；
- shared soft semantic updater；
- 12-frame causal chunks；
- 1600 steps，其中前 800 steps 为 operation pretraining；
- batch size 12，checkpoint selection rule 和基础损失权重。

新增项都直接监督共享 semantic table，不增加任务专用模型 head：

1. `ordinal binding loss`（weight=2.0）：仅对 VideoPlaceOrder 最终状态中被查询的
   `ordered_cell_k` 做 region CE；
2. `multi-target completeness loss`（weight=0.5）：仅对有两个目标颜色的
   Unmask/Swap episode，惩罚请求颜色仍停留在 `none`，不监督具体 region。

Baseline 复用之前已经完成的三个完全同配方 checkpoint；其余 9 次训练使用 GPU 0–3
并行完成。

## 配对结果

Ordinal-only 相对 baseline：

- Place final episode exact：**+6.67 pp**；逐 seed 差值为 0、+13.33、+6.67 pp；
- mean final query：**+5.84 pp**，三个 seed 均提升；
- Unmask：+4.44 pp；Swap：+6.67 pp；
- transition：+0.27 pp，基本不变；
- hold：-1.25 pp；all-state：-1.05 pp，退化小于预设的 2 pp 容忍线。

因此 ordinal-only 满足接 action 的离线门槛：目标任务提升、另外两项任务不退化、
状态转移不退化，hold/all-state 仅有轻微代价。

## 为什么 completeness 没有效果

在离线 test 上，baseline 的 Unmask 和 Swap requested-color missing rate 已经都是 0%。
它们的错误主要是“写到了错误 region”，而不是“保持 none”。因此 completeness loss
监督了一个已经饱和的子问题：

- completeness-only 虽将 Unmask exact 提升 8.89 pp，但 Swap 降低 2.22 pp，Place
  降低 2.22 pp，跨任务不稳定；
- 它只鼓励 non-none，不约束 region 正确，可能把“不确定”变成错误的强制写入；
- 与 ordinal 联用后，mean final query 相对 ordinal-only 下降 8.60 pp，transition
  下降 5.52 pp，说明共享 executor 中存在明显梯度竞争。

所以不应把 completeness loss 接入最终模型。多目标缺失的 action 个例更可能是
episode/domain-specific 的视觉解析错误，应从 event evidence 或置信度校准处理，而不是
增加全局 non-none 损失。

## Action 测试候选

推荐 checkpoint：

`/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/robomme_explicit_event_ordinal_only_seed260909_260829`

选择原因：它在 ordinal-only 三个 seed 中具有最高的最差任务 final episode exact：

- Unmask 73.33%；
- Swap 66.67%；
- Place 66.67%；
- mean final query 72.31%；
- transition 71.65%，hold 72.43%，all-state 72.33%；
- best step 1400。

该 checkpoint 已通过在线 predictor 预检，输出 table `[T+1,7,5]` 和 latent memory
`[T+1,128,64]`，与现有 GroundSG action bridge 完全兼容，无需修改 action 模型。
Launcher 现可通过 `ROBOMME_CAUSAL_TRAINING_DIR` 显式选择 checkpoint。

## 是否接 Action

结论：**适宜**。建议只测试 ordinal-only 候选，不测试 completeness-only 或 combined。

下一轮采用与已有 baseline 完全相同的配对协议：seed 7、三任务各 10 条、1300 步上限、
官方 `symbolic-grounded-subgoal/79999` action checkpoint。主要判据为：

- 总成功率是否高于当前 causal baseline 的 25/30；
- VideoPlaceOrder 是否高于 7/10；
- VideoUnmask/Swap 是否保持不低于 9/10；
- 逐 episode semantic region 与 action success 是否继续一一对应。

如果只达到持平，则 ordinal loss 可作为离线机制消融但不进入最终 action 模型；如果
Place 提升且总成功率不下降，则保留 ordinal-only 配方并扩展到三 action seeds。

## 后续 Action 验证更新

上述 action 准入测试已经完成。Ordinal-only 得到 Unmask 7/10、Swap 6/10、Place
7/10，总计 20/30；原 causal baseline 为 9/10、9/10、7/10，总计 25/30。因此
**离线准入假设被闭环实验否定**，ordinal-only 不进入最终 action 模型。详见
`docs/robomme_ordinal_event_mem_action_test10_260829.md`。
