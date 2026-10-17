# PickXTimes Oracle Boundary 视觉语义监督实验

日期：2026-08-27

## 结论

环境特权标签足够，当前主要缺口是**从 12 帧冻结 SigLIP 特征中区分 Pick 与 Place**，而不是 recurrent MEM 不会学习状态机。

在 simulator oracle boundary 已知的前提下：

- 真实视觉事件头在 locked test 上达到 `64.95%` event accuracy、`70.95%` macro recall；
- `press_complete` 与 `pick_complete` 有明显可学习信号，但 `place_complete` recall 只有 `46.34%`；
- 由预测事件语义驱动的 recurrent MEM 达到 `93.33%` final exact、`60.00%` full-sequence exact；
- 将同一个已训练模型的 test 视觉打乱后，final exact 降到 `20.00%`；将视觉清零后降到 `0%`；
- 但从头使用全零视觉训练的对照，仅靠 oracle boundary 的事件次数，就能达到 `100%` state/sequence/final exact。

因此：

1. SigLIP 特征中确实存在可泛化的事件语义，不是完全没有视觉信号；
2. 该信号尚不足以稳定区分 Pick/Place；
3. Pick 成功示范的 canonical event 顺序过于规则，模型可以绕过视觉、仅按 boundary 序号完成计数；
4. 当前视觉 recurrent MEM 的瓶颈在 observation-to-event 接口，不在环境 GT 信息量或递归容量。

## 实验目的

前一项信息充分性 probe 已证明：

```text
goal + binary oracle boundary -> recurrent symbolic state
```

可在 PickXTimes test 上达到 100%。本实验进一步验证：

1. 当前冻结视觉特征能否在正确选窗后识别 `pick/place/press`；
2. 不使用 teacher latent 时，预测出来的显式事件 token 能否驱动 recurrent MEM；
3. 之前 direct teacher delta 失败是否主要来自 latent basis mismatch。

## 数据与严格隔离

固定 episode-disjoint 划分：

| split | episodes | native events |
|---|---:|---:|
| train | 70 | 438 |
| dev | 15 | 99 |
| locked test | 15 | 97 |

事件类别统计：

| split | Pick | Place | Press |
|---|---:|---:|---:|
| train | 184 | 184 | 70 |
| dev | 42 | 42 | 15 |
| test | 41 | 41 | 15 |

选窗来源为 H5 原生 `info/is_subgoal_boundary`；每个 boundary 与一个 canonical `pick_complete/place_complete/press_complete` 一一对应。输入视觉为该 boundary 所在的非重叠 12 帧 chunk 的冻结 SigLIP `4 x 4 x 1152` patch tokens。

该实验明确不使用：

```text
teacher latent memory
teacher checkpoint/readout
Qwen prediction
learned event gate 选窗
action loss
```

只使用 GT event/state 作为分类监督。

## 模型结构

```text
oracle boundary
  -> 12-frame frozen SigLIP patch tokens
  -> shared visual encoder
  -> visual-only event head
  -> P(pick, place, press)
  -> weighted semantic event token
  -> recurrent memory updater
  -> newly initialized shared state readout
  -> required_count / completed_count / holding / ready / done
```

关键约束是：recurrent updater **不能直接读取视觉 token**，只能读取三类事件概率形成的 semantic token。因此 state loss 是否成功与事件语义接口直接相关，不能通过拟合 teacher full latent 取得捷径。

训练损失：

```text
L = class-balanced event CE + direct symbolic state CE
```

模型从随机初始化训练 1500 steps，batch size 8。按 dev 的 sequence exact、final exact、event macro recall 依次选择 checkpoint。

## Locked test 结果

### 真实视觉模型及输入破坏对照

| Test 输入 | Event acc | Event macro recall | State exact | Sequence exact | Final exact |
|---|---:|---:|---:|---:|---:|
| Native visual | **64.95%** | **70.95%** | **80.36%** | **60.00%** | **93.33%** |
| Permuted visual | 38.14% | 34.31% | 55.36% | 13.33% | 20.00% |
| Zero visual | 42.27% | 33.33% | 66.96% | 0% | 0% |

最佳 visual checkpoint：step 500。

真实视觉的逐事件 recall：

| Event | Recall |
|---|---:|
| pick_complete | 73.17% |
| place_complete | **46.34%** |
| press_complete | 93.33% |

Confusion matrix（行是真值，列是预测 Pick/Place/Press）：

```text
[[30, 10,  1],
 [22, 19,  0],
 [ 1,  0, 14]]
```

主要错误是 `place -> pick`。这与现有视觉 student transition 较弱的表现一致。

### 从头训练的全零视觉对照

| 输入与训练方式 | Event acc | Macro recall | State exact | Sequence exact | Final exact |
|---|---:|---:|---:|---:|---:|
| Zero visual，从头训练 | 42.27% | 33.33% | **100%** | **100%** | **100%** |

最佳 zero checkpoint：step 700。事件头退化为单类预测，但 recurrent updater 仍可根据：

```text
goal required_count + oracle boundary 次数 + recurrent step ordinal
```

恢复完整 Pick 状态机。这个对照再次证明环境信息和 recurrent 表达容量都够用，同时警告：在 canonical 成功示范上，state accuracy 可能掩盖模型并未理解视觉事件。

## 结果解释

### 1. 视觉 evidence 是真实有效的

Native visual 相比 permuted/zero visual：

- event macro recall 提升约 `+36.6/+37.6 pp`；
- final state 提升 `+73.3/+93.3 pp`；
- sequence exact 提升 `+46.7/+60.0 pp`。

所以视觉模型不是仅靠类别频率工作；真实窗口和事件标签之间存在可泛化对应关系。

### 2. 当前视觉 evidence 仍不够稳定

训练 event loss 接近 0，而 dev/test event accuracy 约 63%–65%，说明主要问题是小数据上的泛化，而不是训练不足。继续增加相同训练步数不会解决，step 500 以后 dev sequence 反而下降。

最困难的是 Pick 与 Place。可能原因包括：

- native boundary 在 12 帧 chunk 内的位置不固定，chunk 可能包含大量事件前画面；
- `4 x 4` spatial pooling 对夹爪—物体接触和是否持有物体过粗；
- 仅 RGB 缺少 gripper command/state、EEF 高度、接触等低维物理谓词；
- 只有 70 个训练 episode，视觉背景、物体颜色和姿态容易被记忆；
- non-overlapping chunk 未围绕 boundary 对称取事件前后证据。

### 3. Direct symbolic supervision 比 latent delta 更可诊断

此前 native oracle + direct teacher delta 的 test transition 为 34.02%，且 correction prediction RMS 远小于 target RMS。本实验完全移除 teacher latent 后，事件后 state exact 达到 77.32%，final 达到 93.33%。

这支持以下判断：full-latent basis mismatch 是旧流程的重要问题；显式 event/state supervision 更适合作为 privileged-to-MEM 的第一阶段目标。

但它还不是可部署模型，因为推理仍依赖 oracle boundary，且 Pick/Place event accuracy 不足。

## 下一步决策

最合适的下一项不是继续加训练步数，而是做 boundary-centered 多模态事件 probe：

1. 以 native boundary 为中心取 `6 pre + 6 post` 帧，而不是使用 boundary 所在的固定 chunk；
2. 在 RGB 特征外加入 simulator `gripper state/command` 和 EEF Z，分别做 RGB-only、proprio-only、RGB+proprio 消融；
3. 继续预测统一的 `pick/place/press`，不增加任务定制 action head；
4. 目标是先把 locked-test Pick 与 Place recall 都提升到 80% 以上，再蒸馏 oracle boundary 给视觉 soft gate。

这能进一步判断当前缺的是时间对齐，还是 RGB 本身不包含足够的接触/持有信息。

## 产物

- 实验脚本：`scripts/mem/probe_pickxtimes_oracle_boundary_visual_semantics.py`
- Visual run：`checkpoints/pickxtimes_oracle_boundary_visual_semantics_visual_seed260833_260827`
- Zero control：`checkpoints/pickxtimes_oracle_boundary_visual_semantics_zero_seed260833_260827`

本实验为单 seed 的机制 probe；它足以定位当前接口瓶颈，但论文正式结果仍应补 3 seeds。
