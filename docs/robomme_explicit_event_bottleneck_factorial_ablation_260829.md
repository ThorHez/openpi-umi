# RoboMME 显式事件瓶颈与 updater 结构消融（2026-08-29）

## 目标

验证以下三个结构改动中，究竟哪一项能够缩小 visual ceiling 与 recurrent MEM 的差距：

1. 将视觉输入从 early/late pooling 换成逐 anchor 时间关系编码；
2. 将 soft-mixture semantic executor 换成 straight-through hard 确定性 executor；
3. 在固定 chunk 之间维护轻量 causal visual evidence state。

所有模型使用同一个显式事件接口：`event_type`、`write_entity`、`write_region`、`swap_pair`。semantic updater 只执行 write/swap 代数，不再额外学习状态转移规则。训练均为 1600 step：前 800 step 只学习事件和 payload，后 800 step加入 recurrent state loss 与 teacher-forcing curriculum。

## 主消融：同一种子 260908

下面均为锁定 test 的真实 free rollout，单位为百分数。

| Variant | Params | Train | FPR ↓ | Full update ↑ | Final ↑ | Transition ↑ | Hold ↑ | All-state ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled + soft | 204,002 | 3.8 min | 3.13 | 27.52 | 40.77 | **44.88** | 50.06 | **49.37** |
| pooled + hard deterministic | 204,002 | 3.8 min | 5.91 | 46.98 | 40.77 | 33.86 | 40.51 | 39.62 |
| relational + soft | 271,714 | 22.1 min | 4.21 | 28.19 | 39.15 | 35.43 | **51.15** | 49.06 |
| relational + hard deterministic | 271,714 | 22.1 min | 9.27 | 37.58 | 49.32 | 39.37 | 38.57 | 38.68 |
| pooled + soft + causal state | 232,994 | 4.0 min | **1.42** | **61.07** | 60.85 | 64.57 | 64.93 | 64.88 |
| relational + soft + causal state | 300,706 | 21.1 min | 1.65 | 54.36 | **64.02** | 59.84 | 66.38 | 65.51 |

### 模块归因

- **hard deterministic updater 单独无效。** pooled 条件下 transition 下降 11.02 pp、all-state 下降 9.75 pp。视觉事件仍有误差时，硬提交会把局部误判变成不可恢复的状态跳变；递归训练还会反向破坏事件 parser。
- **relational temporal encoder 单独无效。** soft 条件下 transition 下降 9.45 pp、final 下降 1.62 pp，all-state 基本不变；参数增加 33%，训练耗时约 5.8 倍。
- **causal evidence state 是唯一显著有效模块。** 在 pooled-soft 上，transition +19.69 pp、final +20.09 pp、all-state +15.51 pp，同时 FPR -1.71 pp。
- **relational 与 causal 没有足够的正交增益。** 相对 pooled causal，它只在单种子上提高 final 3.17 pp，却使 transition -4.72 pp、full-update -6.71 pp，并显著增加参数和耗时，因此不保留。

## causal state 三随机种子配对验证

| Variant | FPR ↓ | Full update ↑ | Final ↑ | Min final ↑ | Transition ↑ | Hold ↑ | All-state ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled + soft | 12.00 | 33.11 | 43.65 ± 2.62 | 15.56 | 41.47 ± 4.61 | 48.17 | 47.27 |
| pooled + soft + causal state | **1.31** | **57.72** | **62.56 ± 3.49** | **48.89** | **69.55 ± 6.70** | **73.12** | **72.64** |
| paired delta | **-10.69** | **+24.61** | **+18.92** | **+33.33** | **+28.08** | **+24.95** | **+25.37** |

每个种子的 paired final/transition 都提升：

| Seed | Final delta | Transition delta | All-state delta | FPR delta |
|---:|---:|---:|---:|---:|
| 260908 | +20.09 | +19.69 | +15.51 | -1.71 |
| 260909 | +22.31 | +40.94 | +38.16 | -27.97 |
| 260910 | +14.36 | +23.62 | +22.43 | -2.39 |

## 分任务结果（三种子均值）

| Task | Baseline final | Causal final | Baseline full update | Causal full update |
|---|---:|---:|---:|---:|
| VideoUnmask | **66.67** | 64.44 | 32.22 | **35.56** |
| VideoUnmaskSwap | 48.72 | **74.36** | 43.88 | **71.31** |
| VideoPlaceOrder | 15.56 | **48.89** | 12.50 | **47.50** |

causal state 的主要收益来自需要跨 chunk 保留操作进展的 Swap 和 PlaceOrder。VideoUnmask 的 final 略降 2.23 pp，但 full-update recall 略升；说明简单可见性任务不依赖长期 evidence，而复杂事件显著依赖它。

## 结论

原计划中的“显式事件瓶颈 + 确定性 updater”不能整体接受：显式事件 schema 是有用的可解释接口，但 hard updater 在当前视觉准确率下有害。真正解决当前结构问题的是：

```text
12-frame pooled anchor evidence
          ↓
lightweight causal per-anchor evidence state
          ↓
explicit event tuple
          ↓
soft semantic executor
          ↓
recurrent semantic table
```

下一版主模型应采用 `pooled_soft_causal`，不采用 hard executor，也不采用当前 relational temporal attention。下一步优化应围绕 causal state 做状态可视化、gate 诊断和 action 接口验证，而不是继续扩大单窗口或增加辅助 head。

## 产物

- 模型：`src/openpi/tasks/robomme/explicit_event_bottleneck_memory.py`
- 单元测试：`src/openpi/tasks/robomme/explicit_event_bottleneck_memory_test.py`
- 训练器：`scripts/mem/train_robomme_explicit_event_bottleneck_ablation.py`
- 结果：`checkpoints/robomme_explicit_event_{variant}_seed{260908,260909,260910}_260829/`

