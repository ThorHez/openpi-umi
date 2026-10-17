# ShellGame V10 action-only no-MEM baseline（100 episodes）

日期：2026-08-28

这份文档记录 ShellGame 的严格无记忆端到端基线。主指标采用放宽后的
成功定义：**模型选择正确杯子，并且夹爪与该杯发生物理接触**。该指标用于
区分“能够移动到杯子附近”和“能够在没有时序记忆时选对目标”这两种能力。

## 1. 模型与 no-MEM 隔离定义

使用 checkpoint：

```text
checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/
  absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/
  1000
```

严格 no-MEM 配置如下：

- 不输入外接 Qwen/semantic MEM。
- `old_memory_condition_strength=0`，关闭 V10 原生 tracker 对 action token
  的条件残差。
- 推理时跳过 tracker encoder 和旧的 action-memory cross-attention。
- 不向模型提供真实 60 帧历史。checkpoint 的 transform 仍要求 61 个旧图像
  键，因此所有 61 个槽均复制当前帧，只用于满足输入 shape 契约。
- 模型仅保留当前 base/wrist 图像、当前机器人状态和通用抓取 prompt。

注意：这是对联合训练 V10 checkpoint 的严格 action-only/no-MEM 消融，
不是一个从头单独训练的 current-only V10 checkpoint。

## 2. 闭环评测协议

- 100 个唯一 episode，编号 `0..99`，环境 seed `260813`。
- 三个杯子，每回合三次交换。
- 四个并行分片，每个分片 25 回合；每回合使用新的 MuJoCo/EGL 进程。
- action：absolute EEF7 raw controller command。
- diffusion sampling steps：4。
- replan interval：8。
- 每回合最多 150 个 policy steps。
- 不启用 simulator XY-before-Z guard。
- 夹爪—杯子接触由 MuJoCo body contact 判定，并在完整 rollout 的每一步
  在线累计，不局限于末尾 physics-debug 窗口。
- 本次服务使用随机 diffusion noise（`deterministic_noise=False`）；环境 episode
  序列固定，但没有为每次 policy query 固定 diffusion seed。

目标杯分布为 left/middle/right = `37/39/24`；模型选择分布为
`35/31/34`。

## 3. 成功定义

主指标：

```text
relaxed_end_to_end_success
  = cup_selection_correct AND selected_target_cup_contacted_by_gripper
```

辅助指标：

- `any_cup_contact`：夹爪碰到任意杯子，只衡量到达/接触能力。
- `target_cup_contact`：rollout 中曾碰到目标杯，不要求 selection vote 正确；
  机械臂可能先后碰到多个杯子，因此该指标不能替代主指标。
- `target_lift_success`：正确目标杯抬升至少 8 cm。
- `any_cup_lift_success`：任意杯抬升至少 8 cm，可能抬错杯。

## 4. 100-episode 结果

| 指标 | 结果 | 成功率 |
|---|---:|---:|
| 正确选杯且夹爪接触正确杯（主指标） | **35/100** | **35%** |
| 正确选杯 | 35/100 | 35% |
| rollout 中曾接触目标杯 | 45/100 | 45% |
| 接触任意杯 | 100/100 | 100% |
| 正确目标杯抬升至少 8 cm | 1/100 | 1% |
| 任意杯抬升至少 8 cm | 2/100 | 2% |

主指标的 95% Wilson 置信区间为 `26.4%--44.7%`。与三杯随机水平
`1/3 = 33.3%` 比较，双侧 exact binomial test 为 `p=0.751`，没有显著
高于随机水平。

两次完整抬杯中，episode 34 正确选择并抬起目标杯；episode 77 抬起了错误
杯子。因此正确选杯并抬起的严格成功率为 `1/100 = 1%`。

## 5. 结论与使用方式

该结果作为 ShellGame 的 **V10 action-only no-MEM 端到端基线**：

- 夹爪能够稳定到达并接触某个杯子（100% any-cup contact）。
- 没有时序记忆时，正确选杯且接触的 35% 与三选一随机水平一致。
- V10 action head 不能在切断原生 tracker 条件后保持原来的完整抓取能力；
  该 checkpoint 的目标和阶段条件来自联合训练的 tracker residual。

后续带 MEM 方法应至少同时报告：正确选杯率、正确选杯且接触率、正确目标
抬杯率。主要 MEM 方法应与本基线的 `35%` 宽松端到端成功率比较；
`100%` 任意杯接触率不能作为 ShellGame 任务成功率。

此前得到的 `20/20` 选杯、`13/20` 抬杯结果保留了 V10 原生 60 帧 tracker，
属于完整 V10 policy 结果，**不是** no-MEM baseline，不能与本文结果混用。

## 6. 结果资产

合并结果：

```text
evaluation/shellgame/
  v10_action_only_no_mem_current_only_step1000_replan8_100ep_260828/result.json
```

人工摘要：

```text
evaluation/shellgame/
  v10_action_only_no_mem_current_only_step1000_replan8_100ep_260828/summary.md
```

四个 `shard_*` 目录包含 100 个视频、100 份 physics log 和分片结果。合并时
已验证 episode `0..99` 无缺失、无重复，100 个 episode seed 均唯一，全部
JSON 可正常解析。

关键实现入口：

- `examples/shellgame/serve_v10_exact_parallel_semantic_adapter_deterministic.py`
  中的 `v10_action_no_memory` mode。
- `examples/shellgame/main_absolute_eef_fixed_history.py` 中的
  `_current_only_compat_policy_input`。
- `examples/shellgame/eval_absolute_eef_fixed_history_xy_before_z_isolated.py`
  中的 `--current-only-no-memory-input` 和全 rollout 接触聚合。

