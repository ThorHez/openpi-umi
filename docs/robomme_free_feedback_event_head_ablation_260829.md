# RoboMME 自由递归事件 Head 消融（2026-08-29）

## 目的

检验给 recurrent MEM 增加独立事件完成 head 和阶段（phase）head，是否能在真实自由递归反馈中稳定改善记忆，而不只是在 teacher-forced 或独立窗口分类中提高辅助指标。

## 实现

在 `AnchorConditionedTransitionMemory` 中加入可选的三个输出：

- `completion_logits`：当前 micro event 是否应提交；
- `event_kind_logits`：提交后是 write 还是 swap；
- `phase_logits`：当前 chunk 所处的观察/操作阶段。

递归更新使用 straight-through hard commit。默认旧结构保持不变；只有显式设置 `use_auxiliary_heads=True` 才创建新 head。三组消融使用相同参数树、相同初始化、相同数据与训练预算：

1. `joint`：原始三分类 `hold/write/swap`；
2. `completion`：独立 completion + event-kind；
3. `phase`：completion + event-kind，并增加 phase 辅助损失。

训练共 1600 step：前 800 step 只训练事件和载荷解析，后 800 step 加入 recurrent table loss 和 teacher-forcing curriculum。每组运行种子 260905、260906、260907。checkpoint 只用 dev free rollout 选择；commit threshold 只在 dev 校准；下面报告锁定 test 的 free rollout。

## 主要结果

### 开发集固定阈值 0.5（三种子均值）

| Head | Hold FPR ↓ | Full-update recall ↑ | Mean final ↑ | Transition ↑ | Hold state ↑ | All-state ↑ |
|---|---:|---:|---:|---:|---:|---:|
| joint | 6.42 | 45.58 | **51.95** | **45.38** | 42.40 | 42.76 |
| completion | 5.73 | 42.86 | 44.58 | 42.56 | **43.20** | **43.12** |
| completion + phase | **5.70** | 42.40 | 44.61 | 41.28 | 41.60 | 41.56 |

单位均为百分数。相同 operating point 下，phase head 没有提高 final、transition 或 all-state；它与 completion-only 的 final 基本相同，并使 transition/all-state 略降。

### Dev 校准后锁定测试集（三种子均值）

校准目标为 dev hold FPR ≤ 0.5%。

| Head | Test FPR ↓ | Full-update recall ↑ | Mean final ↑ | Transition ↑ | Hold state ↑ | All-state ↑ | Phase acc. ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 4.42 | **44.07** | **43.50 ± 8.68** | **38.58** | **46.03** | **45.04** | 36.48 |
| completion | **0.04** | 0.45 | 1.48 ± 1.28 | 1.05 | 36.15 | 31.48 | 36.37 |
| completion + phase | 0.32 | 8.95 | 15.81 ± 13.81 | 10.50 | 37.89 | 34.24 | **76.90** |

phase head 相对 completion-only 的均值确实更高（final +14.33 pp、transition +9.45 pp、full-update recall +8.50 pp），但该增益不稳定：种子 260906 在校准后完全不提交事件，final/transition 都为 0。joint 基线虽然误触发过高，仍明显保留了最多有效状态转移。

### 每种子部署结果

| Head | Seed | Threshold | FPR | Full update | Mean final | Transition | All-state | Phase acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint | 260905 | 0.70 | 4.26 | 46.98 | 53.50 | 42.52 | 47.38 | 44.23 |
| joint | 260906 | 0.95 | 2.73 | 38.93 | 37.86 | 37.80 | 43.61 | 51.57 |
| joint | 260907 | 0.50 | 6.25 | 46.31 | 39.15 | 35.43 | 44.13 | 13.63 |
| completion | 260905 | 0.99 | 0.11 | 0.67 | 2.22 | 1.57 | 31.66 | 41.19 |
| completion | 260906 | 0.99 | 0.00 | 0.00 | 0.00 | 0.00 | 31.24 | 56.71 |
| completion | 260907 | 0.99 | 0.00 | 0.67 | 2.22 | 1.57 | 31.55 | 11.22 |
| completion + phase | 260905 | 0.95 | 0.45 | 15.44 | 25.47 | 16.54 | 35.95 | 76.52 |
| completion + phase | 260906 | 0.99 | 0.00 | 0.00 | 0.00 | 0.00 | 31.24 | 77.04 |
| completion + phase | 260907 | 0.95 | 0.51 | 11.41 | 21.97 | 14.96 | 35.53 | 77.15 |

## 结论与结构决策

**当前不能证明特定 phase head 对 recurrent MEM 有稳定正向影响。** 它稳定学会了阶段分类，却没有在相同阈值下改善状态转移；低 FPR 校准后的 memory 收益又对随机种子和阈值高度敏感。这说明辅助 phase supervision 主要形成了一个可解码旁路，尚未真正改变 completion gate 的因果决策。

因此：

- 新 head 保留为显式可选的实验分支，便于后续研究；
- 默认模型仍使用原 joint head，不改变已有 checkpoint 的参数树；
- 不用当前 phase 版本接 action，也不在论文中宣称其带来稳定提升；
- 如果继续这一方向，下一次应该测试“phase 条件化 completion”（让 phase 表征直接参与 commit logit），而不是继续加大独立 phase loss。

## 产物

- 模型：`src/openpi/tasks/robomme/anchor_conditioned_transition_memory.py`
- 单元测试：`src/openpi/tasks/robomme/anchor_conditioned_transition_memory_test.py`
- 消融训练器：`scripts/mem/train_robomme_free_feedback_head_ablation.py`
- checkpoint/result：`checkpoints/robomme_free_feedback_staged_{joint,completion,phase}_seed{260905,260906,260907}_260829/`

