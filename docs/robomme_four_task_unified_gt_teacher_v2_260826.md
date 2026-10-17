# RoboMME 四任务统一 GT Teacher V2

日期：2026-08-26

## 结论

已经实现并训练一套与 Qwen 完全解耦的四任务统一 teacher。teacher 不读取视频、不读取 Qwen JSON，只接收场景 GT goal/state/event。四个任务共享：

- 同一个 `128 x 64` memory 空间；
- 同一套 goal/event embedding；
- 同一个 recurrent updater；
- 同一个 19-field query readout；
- 不使用任务专用 head 或任务路由器。

最终用于学生 MEM 蒸馏的是 **canonical GT-state memory**。在锁定 test 的 60 个 episode 上，它对四个任务均达到：

- field accuracy：100%；
- state exact：100%；
- full state sequence exact：100%；
- final state exact：100%。

因此 Qwen 可以独立继续优化，teacher checkpoint 和 student 蒸馏 target 不会随 Qwen 版本改变。

## 数据

固定 episode-disjoint 划分：

| split | 总 episode | 每任务 |
|---|---:|---:|
| train | 280 | 70 |
| dev | 60 | 15 |
| test | 60 | 15 |

最长 GT event sequence 为 11，统一 padding 到 12 个事件；state 序列包含初始状态，因此缓存维度为 13。

训练事件统计：

| event | 数量 |
|---|---:|
| target_visible | 199 |
| target_covered | 199 |
| swap_complete | 134 |
| pick_complete | 184 |
| place_complete | 397 |
| press_complete | 70 |

标签来源：

- PickXTimes：`choice_action + subgoal boundary + gripper edge`；
- VideoUnmask：可见颜色/位置与 visible-to-covered phase；
- VideoUnmaskSwap：初始颜色/位置与经过 grounded-pick 审计的 motion-derived swap pair；
- VideoPlaceOrder：placement cell 和目前已有的 target-relevant hard swap。

数据构建结果：

```text
artifacts/robomme_four_task_gt_teacher_sequences_v1_260826/
  train.npz
  dev.npz
  test.npz
  train.jsonl
  dev.jsonl
  test.jsonl
  summary.json
```

`summary.json` 明确记录：

```text
teacher_uses_qwen_predictions = false
```

## 统一状态合同

teacher 使用一个共享 readout 解码 19 个字段：

```text
task
target_color_0, target_color_1
red_cell, green_cell, blue_cell
ordered_cell_0..3
covered, completed_swap_count
written_count
required_count, completed_count
holding, ready_to_press, done
queried_ordinal
```

各任务只通过 `state_field_mask` 决定哪些字段参与 loss。例如 PickXTimes 监督 count/holding/done，VideoUnmaskSwap 监督目标颜色对应的 cell/covered/swap count。所有字段仍经过同一组 query 和同一个六分类 projection。

统一事件输入保持四字段合同：

```text
event, entity, region_a, region_b
```

`no_completed_event`、`incomplete_event` 和 `insufficient_evidence` 被硬门控为 exact no-op，不允许改变 teacher memory。

## 为什么增加 canonical GT-state memory

第一版只训练：

```text
goal + previous memory + GT event -> next teacher memory -> state readout
```

它在 test 上得到：

| 任务 | final state exact |
|---|---:|
| PickXTimes | 100% |
| VideoUnmask | 100% |
| VideoPlaceOrder | 93.3% |
| VideoUnmaskSwap | 6.7% |

VideoUnmaskSwap 的失败来自未见过的“目标颜色 × 当前 region × swap pair”组合。纯神经 updater 在 70 个训练 episode 上记住了常见组合，但没有稳定学会离散置换规则。这样的 latent 不能作为干净 teacher target。

V2 增加：

```text
GT state_t
  -> shared field/class state encoder
  -> canonical teacher memory_t [128,64]
  -> shared readout
```

同时保留 event rollout 作为辅助分支：

```text
goal + GT events
  -> recurrent rollout memory_t
  -> shared readout
  -> align to stopgrad(canonical memory_t)
```

训练 loss：

```text
L = L_rollout_state
  + L_canonical_state
  + cosine(rollout_memory, stopgrad(canonical_memory))
  + 0.1 * MSE(rollout_memory, stopgrad(canonical_memory))
```

canonical memory 是学生使用的 teacher target；event rollout 仅用于诊断共享 updater 的组合泛化，不能替换 canonical target。

## V2 锁定 test 结果

### Canonical teacher target

| 任务 | state exact | sequence exact | final state exact |
|---|---:|---:|---:|
| PickXTimes | 100% | 100% | 100% |
| VideoUnmask | 100% | 100% | 100% |
| VideoUnmaskSwap | 100% | 100% | 100% |
| VideoPlaceOrder | 100% | 100% | 100% |
| Overall | 100% | 100% | 100% |

### Diagnostic event-only rollout

| 任务 | state exact | sequence exact | final state exact |
|---|---:|---:|---:|
| PickXTimes | 100% | 100% | 100% |
| VideoUnmask | 100% | 100% | 100% |
| VideoPlaceOrder | 89.1% | 80.0% | 80.0% |
| VideoUnmaskSwap | 54.3% | 6.7% | 20.0% |
| Overall | 84.0% | 71.7% | 75.0% |

这说明 canonical teacher 已适合提供蒸馏目标，但不能把辅助 event-only updater 宣称为四任务可部署 symbolic updater。

## Checkpoint 和缓存

训练目录：

```text
checkpoints/robomme_four_task_unified_gt_teacher_canonical_v2_260826/
```

按 dev 上 canonical 与 rollout strict final 的较小值选择：

```text
best step = 1700
best params = checkpoints/robomme_four_task_unified_gt_teacher_canonical_v2_260826/best/params
```

学生蒸馏缓存：

```text
artifacts/robomme_four_task_gt_teacher_memory_v2_260826/
  train.npz
  dev.npz
  test.npz
  summary.json
```

每个 split 的主要字段：

```text
teacher_memory              # canonical target, float16 [N,13,128,64]
diagnostic_rollout_memory   # 仅诊断，不作为 student target
task_id
episode_index
source
step_mask
state_targets
state_field_mask
```

缓存生成时再次验证了 train/dev/test canonical full-sequence exact 均为 100%。

## 与 Qwen 的关系

当前正式训练链路：

```text
场景 GT state/event -> canonical teacher memory
视频窗口 + previous student memory -> student memory
student memory -> 对齐 canonical teacher memory
```

Qwen 独立链路：

```text
视频窗口 -> Qwen unified JSON
```

Qwen 只用于：

- 评估通用视觉事件理解；
- 后续无 GT 数据的伪标注；
- “人工 GT 比例 / Qwen 伪标签比例”标注效率消融。

Qwen 不进入 teacher checkpoint 训练，也不进入 canonical memory cache。

## 下一步：训练统一视觉 student MEM

建议第一版 student：

```text
goal token + previous student memory + 12-frame local window
  -> shared visual event encoder
  -> shared recurrent updater
  -> student memory_t [128,64]
```

目标：

```text
L_student = cosine(student_memory_t, stopgrad(teacher_memory_t))
          + 0.1 * MSE(student_memory_t, stopgrad(teacher_memory_t))
          + 0.25 * masked_state_readout_loss
          + event_gate/type/argument losses
```

训练必须按 episode sequence 展开，不能把所有窗口当成彼此独立样本；否则 PickXTimes count 和 swap/place 状态更新无法学习误差累积。

第一轮应继续保持四任务均衡采样，不接 action。验收顺序：

1. teacher cache decode 100%（已通过）；
2. student teacher-memory cosine/MSE；
3. 每任务 full state sequence / final state；
4. dense sliding-window no-op、incomplete 和重复触发；
5. 达标后再接 action expert。

## 当前限制

VideoPlaceOrder 的 100% 是针对当前 GT 合同：所有 placement 和已有的 target-relevant hard swap。当前数据并不包含 hard 场景所有无关目标的完整 swap pair。因此不能把该结果表述成“VideoPlaceOrder 全部物体完整世界状态 100%”。如需完整 hard-scene teacher，仍需从 simulator replay 或稳定 motion tracking 恢复所有 swap pair。

## 代码入口

```text
src/openpi/tasks/robomme/unified_gt_teacher.py
src/openpi/tasks/robomme/unified_gt_teacher_test.py
scripts/mem/build_robomme_four_task_gt_teacher_sequences.py
scripts/mem/train_robomme_four_task_gt_teacher.py
scripts/mem/cache_robomme_four_task_gt_teacher_memories.py
```

