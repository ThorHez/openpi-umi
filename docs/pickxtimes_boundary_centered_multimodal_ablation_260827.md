# PickXTimes Boundary-Centered 多模态事件消融

日期：2026-08-27

## 结论

目标已经在 proprio 模态上达到：`Pick` 和 `Place` locked-test recall 均超过 80%，实际达到 100%。

四组结果表明：

- 仅将 RGB 改为 boundary-centered `6 pre + 6 post`，不能同时解决 Pick/Place；
- gripper state/command 单独即可在 test 上达到三类事件 100%；
- EEF-Z 单独也在当前数据分布上达到 100%，但收敛更慢，可能使用了控制器轨迹的细微模式；
- RGB-only 的 3-seed Pick/Place recall 均值只有 `60.16%/53.66%`，且两类间明显摇摆；
- RGB+proprio 在三个 seed 中都于 50 steps 达到 dev/test 三类 100%，是最稳妥的接口；
- 当前缺少的不是环境标签，而是 student 没有读取机器人自身已知的低维物理状态。

## 实验设置

### Boundary-centered 窗口

每个原生 simulator `is_subgoal_boundary` 完成事件使用：

```text
[boundary - 6, ..., boundary - 1,
 boundary, ..., boundary + 5]
```

共 12 帧。全量审计：

| split | episodes | events | 最小前向余量 | 最小后向余量 |
|---|---:|---:|---:|---:|
| train | 70 | 438 | 101 帧 | 32 帧 |
| dev | 15 | 99 | 103 帧 | 33 帧 |
| test | 15 | 97 | 104 帧 | 31 帧 |

因此所有 centered window 都是真实帧，无 padding 或越界。

注意：`+5` 帧表示该 probe 在 boundary 后有 5 帧观测延迟，不是 boundary 时刻严格零延迟的 causal detector。若执行频率为 20 Hz，相当于最多约 250 ms lookahead/确认延迟。

### 四组输入

1. `RGB`
   - 冻结 SigLIP `4 x 4 x 1152` patch tokens；
   - 共享时空视觉 encoder。
2. `Gripper`
   - `obs/gripper_state[0:2]`；
   - `obs/is_gripper_close`；
   - 已执行/下发的 `action/eef_action[6]` gripper command。
3. `EEF-Z`
   - `obs/eef_state_raw/pose[2]`；
   - `action/eef_action_raw/pose[2]` target Z。
4. `RGB+proprio`
   - RGB、Gripper 和 EEF-Z 全部融合。

低维特征只使用 train 统计量标准化，dev/test 统计量不参与训练。

### 标签与选择指标

三分类标签：

```text
pick_complete
place_complete
press_complete
```

训练按类别均衡采样。checkpoint 首先最大化：

```text
min(Pick recall, Place recall)
```

然后比较 macro recall 和 accuracy，避免模型通过牺牲其中一类获得较高 overall accuracy。

为防止 learned recurrent updater 只靠事件序号作弊，事件预测还驱动一个固定的 Pick 状态机：

```text
predicted event sequence
 -> completed_count / holding / ready / done
```

symbolic sequence/final 只有在预测事件足以恢复真实轨迹时才算正确。

## Locked test 结果

### 三 seed 视觉稳定性

| 模态 | Seed | Best step | Event acc | Macro recall | Pick recall | Place recall | Press recall | Symbolic sequence/final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centered RGB | 260834 | 350 | 60.82% | 67.70% | 65.85% | 43.90% | 93.33% | 6.67% |
| Centered RGB | 260835 | 150 | 62.89% | 70.73% | 56.10% | 56.10% | 100% | 13.33% |
| Centered RGB | 260836 | 200 | 65.98% | 73.17% | 58.54% | 60.98% | 100% | 20.00% |
| **RGB mean ± std** | 3 seeds | — | **63.23 ± 2.12%** | **70.53 ± 2.24%** | **60.16 ± 4.15%** | **53.66 ± 7.18%** | **97.78 ± 3.14%** | **13.33 ± 5.44%** |
| **RGB+proprio** | 3 seeds | 50 | **100 ± 0%** | **100 ± 0%** | **100 ± 0%** | **100 ± 0%** | **100 ± 0%** | **100 ± 0%** |

RGB+proprio 的三个 seed（260834/260835/260836）均得到完全相同的 locked-test 结果，不是单次幸运初始化。

### 单模态 proprio 机制消融

| 模态 | Seed | Best step | Event acc | Pick recall | Place recall | Press recall | Symbolic sequence/final |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gripper | 260834 | 50 | **100%** | **100%** | **100%** | **100%** | **100%** |
| EEF-Z | 260834 | 800 | **100%** | **100%** | **100%** | **100%** | **100%** |

### Centered RGB confusion matrix

行是真值，列是预测 Pick/Place/Press：

```text
[[23, 18,  0],
 [18, 23,  0],
 [ 0,  0, 15]]
```

这是 seed 260835 的代表性 confusion matrix。三个 seed 的共同现象是 Press 基本可分，而 Pick/Place 决策边界不稳定；有的 run 偏向 Pick，有的偏向 Place。

相比此前任意位置 fixed chunk 的 `Pick 73.17% / Place 46.34% / Press 93.33%`，centered RGB 三 seed 均值为：

- Press：`97.78%`，较容易识别；
- Place：`53.66%`，有小幅改善；
- Pick：`60.16%`，有所下降。

窗口居中改善了时间对齐，但没有真正解决 Pick/Place 的视觉歧义；任一 RGB seed 都没有同时达到两类 80%。

## 为什么 proprio 能达到 100%

全量 train/dev/test 统计呈现一致的物理模式：

| Event | Gripper observation | Gripper command | EEF-Z 行为 |
|---|---|---|---|
| Pick complete | 闭合、夹物体后约 0.018 | -1 | 回升到约 0.148 m |
| Place complete | 张开、约 0.040 | +1 | 回升到约 0.149 m |
| Press complete | 几乎完全闭合、约 0.000 | -1 | 从约 0.062 降到 0.027 m |

这些字段不是未来任务答案，而是机器人执行器自身可观测状态和已下发控制命令。部署时 controller/VLA 本来就知道 gripper command，并能读取 gripper state 与 EEF pose，因此可以合法作为 MEM updater 的 observation。

## 必须保留的风险判断

### 1. 100% 不等于真实闭环已经解决

当前数据全部来自官方成功 controller，gripper 和 EEF 轨迹非常规范。在动作失败、抓空、滑落、重复闭合或 Z 轨迹抖动时，这些模式会改变。当前 100% 证明的是：

```text
现有特权/机器人状态字段足以无歧义监督成功示范中的事件语义
```

它不证明在 on-policy failure 分布上仍为 100%。

### 2. EEF-Z 可能存在 controller shortcut

Pick 和 Place 的平均 Z 很接近，EEF-Z-only 模型需要约 800 steps 才达到最佳结果，可能利用了轨迹速度、target Z 或控制器模板的细节。相比之下，gripper 状态直接对应抓取/释放物理谓词，解释性和预期鲁棒性更好。

### 3. Oracle boundary 仍未被移除

本实验只解决：

```text
正确窗口 -> 这是什么事件？
```

尚未解决：

```text
连续视频/机器人状态 -> 事件何时完成？
```

因此它仍是 event-semantic 上界，不是可部署 action 模型。

## 对 MEM 结构的建议

下一版 updater 的 evidence 不应只有 RGB：

```text
visual token
+ gripper state token
+ previous/issued gripper command token
+ EEF-Z token
 -> shared soft event gate
 -> pick/place/press semantic logits
 -> recurrent MEM update
```

建议先采用 `RGB+proprio`，而不是 EEF-Z-only：

- gripper 提供直接、可解释的 Pick/Place 证据；
- EEF-Z 区分 Press 和普通抓放阶段；
- RGB 保留颜色、物体身份和空间位置，后续 VideoPlaceOrder 等任务仍需要；
- 结构保持统一，不需要 PickXTimes 定制 action head。

下一项验证应加入 failure/retry 数据，并把 oracle boundary 替换为基于 proprio edge + RGB 的 causal soft gate。只有在抓空、重试和滑落样本上仍能保持较高 precision/recall，才适合接 action。

## 产物

- 数据构建：`scripts/mem/build_pickxtimes_boundary_centered_multimodal_events.py`
- 训练/评估：`scripts/mem/probe_pickxtimes_boundary_centered_modalities.py`
- 数据缓存：`artifacts/pickxtimes_boundary_centered_multimodal_events_v1_260827`
- RGB：`checkpoints/pickxtimes_centered_modalities_rgb_seed260834_260827`
- RGB seeds：同前缀 `seed260834/260835/260836`
- Gripper：`checkpoints/pickxtimes_centered_modalities_gripper_seed260834_260827`
- EEF-Z：`checkpoints/pickxtimes_centered_modalities_eef_z_seed260834_260827`
- RGB+proprio：`checkpoints/pickxtimes_centered_modalities_rgb_proprio_seed260834_260827`
- RGB+proprio seeds：同前缀 `seed260834/260835/260836`

RGB 与 RGB+proprio 已完成 3 seeds；Gripper 与 EEF-Z 是单 seed 机制消融。论文报告仍应补 failure-distribution，尤其不能只在官方成功 controller 的规范轨迹上报告 100%。
