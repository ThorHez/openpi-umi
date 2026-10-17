# ShellGame 多阶段训练步骤说明

> 目标：提供一份可以直接交给合作方执行的训练步骤。  
> OpenPI目录：`/data2/hzl_workspace_for_pi_mem/openpi-umi`  
> 推荐GPU：`0,1,2,3,4,5`，对应 `--fsdp-devices 6`  
> 当前推荐最终模型：V10 step1000

## 1. 训练阶段总览

```text
M1  初始球杯分类
 ↓
M2  三次交换关系分类
 ↓
M3  递归memory组合
 ↓
M4  全视觉memory验证
 ↓
M5  合并完整tracker
 ↓
M6  memory接入Pi0.5 action expert
 ↓
A0  nominal absolute-EEF动作训练（从头复现时执行）
 ↓
A1  使用已经验证的V6-5999作为动作基线
 ↓
A2  V10混合数据训练0～500步
 ↓
A3  恢复完整train state，续训到2000步
 ↓
闭环筛选step499/1000/1500/1999，正式评测最佳checkpoint
```

如果合作方只研究action训练，从A1开始即可，不需要重新执行M1～M6。

## 2. 训练前准备

### 2.1 数据

确认以下目录存在：

```text
# M1～M6使用
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint

# A0和V10 nominal数据
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7

# V10行为保留数据
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816

# V10时序纠偏数据
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819
```

V9目录还必须包含：

```text
safe_balanced_recovery_v9_oracle_supervision_audit.json
xy_sampling_metrics_v9.npz
```

检查命令：

```bash
test -d /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint
test -d /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7
test -d /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819/xy_sampling_metrics_v9.npz
df -h /data2
```

训练前建议至少保留100 GiB可用空间。

### 2.2 统一输入输出契约

- 历史图像固定使用episode frame `0..59`，共60帧，`stride=1`。
- frame 60之后只更新当前帧，不滚动替换历史。
- 总输入为60帧历史加1帧当前图像。
- EEF state：`xyz(3) + rot6d(6) + gripper(1)`，共10维。
- EEF action：`xyz(3) + rotation-vector(3) + gripper(1)`，共7维。
- action horizon：16。
- 训练prompt必须与数据转换和测试一致。

所有命令默认从以下目录执行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
```

## 3. Memory训练

### M1：初始球杯分类

目的：验证frame 0图像足以判断球最开始位于左、中、右哪个杯子。

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup uv run python examples/shellgame/train_frame0_initial_cup_probe.py \
  --exp-name frame0_initial_cup_<TAG> \
  --steps 300 \
  --batch-size 60 \
  --num-workers 8 \
  --fsdp-devices 1 \
  --eval-interval 50 \
  --eval-batches 10 \
  > train_m1_<TAG>.log 2>&1 &
```

通过条件：held-out初始杯分类接近100%。

参考checkpoint：

```text
checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params
```

### M2：三段交换关系分类

目的：分别识别三段交换属于 `left-middle`、`left-right` 还是 `middle-right`。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_pair_fixed_grid_probe.py \
  --exp-name swap_pair_full600_<TAG> \
  --steps 600 \
  --warmup-steps 50 \
  --peak-lr 3e-4 \
  --batch-size 18 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 200 \
  --eval-batches 100 \
  > train_m2_<TAG>.log 2>&1 &
```

通过条件：三个交换关系及联合held-out准确率达到100%。

参考checkpoint：

```text
checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params
```

### M3：递归memory组合

目的：验证128个64维memory tokens能够依次组合三次交换，并在一次history read后得到最终杯位。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_oracle_single_history_read_adapter_probe.py \
  --exp-name oracle_single_read_600_<TAG> \
  --steps 600 \
  --warmup-steps 20 \
  --peak-lr 1e-3 \
  --batch-size 72 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 25 \
  --eval-batches 9 \
  --memory-tokens 128 \
  --memory-width 64 \
  --memory-depth 2 \
  --memory-heads 4 \
  --adapter-heads 4 \
  --pair-mode correct \
  > train_m3_<TAG>.log 2>&1 &
```

通过条件：正常输入held-out为100%；打乱交换关系后准确率应明显下降。

参考checkpoint：

```text
checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params
```

### M4：全视觉memory验证

目的：把M1初始杯识别、M2交换识别和M3递归memory组合起来，确认模型仅凭历史图像得到最终杯位。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py \
  --exp-name fully_visual_memory_eval_<TAG> \
  --init-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params \
  --initial-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params \
  --memory-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params \
  --steps 0 \
  --batch-size 72 \
  --num-workers 2 \
  --fsdp-devices 6 \
  --eval-batches 9
```

通过条件：初始杯、三段交换和最终杯位均为100%。此外应确认打乱frame 0或交换历史后准确率显著下降。

### M5：合并完整tracker

目的：把M1～M3权重合并成单个tracker checkpoint，并用小动作头确认memory能影响动作方向。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_fully_visual_joint_action_probe.py \
  --exp-name fully_visual_joint_action_<TAG> \
  --init-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params \
  --initial-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params \
  --memory-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params \
  --steps 300 \
  --warmup-steps 30 \
  --peak-lr 3e-4 \
  --batch-size 72 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10 \
  > train_m5_<TAG>.log 2>&1 &
```

参考checkpoint：

```text
checkpoints/pi0_shellgame_three_swap_fully_visual_joint_action_260810/fully_visual_joint_action_head300_b72_260810/299/params
```

### M6：memory接入Pi0.5 action expert

目的：用learned queries读取raw memory，并通过cross-attention条件化Pi0.5 action expert。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_query_crossattn_pi_joint_action_probe.py \
  --exp-name query_crossattn_pi_action_<TAG> \
  --tracker-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_fully_visual_joint_action_260810/fully_visual_joint_action_head300_b72_260810/299/params \
  --steps 300 \
  --warmup-steps 30 \
  --peak-lr 3e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10 \
  --query-tokens 16 \
  --query-width 256 \
  --query-depth 2 \
  --query-heads 4 \
  --action-cross-attention-heads 8 \
  > train_m6_<TAG>.log 2>&1 &
```

第一次训练这个接口时不要加入 `--restore-memory-interface`。

通过条件：正常memory下的动作选杯明显高于shuffle memory和zero memory。参考结果约为normal 98%、shuffle 43%、zero 35%。

参考checkpoint：

```text
checkpoints/pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/query_crossattn_pi_flow_action300_b12_260810/299/params
```

## 4. Absolute-EEF action训练

### A0：Nominal EEF训练

仅在从M6完整复现action链时执行。tracker和memory保持冻结，只训练action expert以及action/time projection。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_old_tracker_full_absolute_eef.py \
  --exp-name absolute_eef_nominal_<TAG> \
  --init-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/query_crossattn_pi_flow_action300_b12_260810/299/params \
  --steps 2000 \
  --warmup-steps 300 \
  --peak-lr 3e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 1000 \
  --gripper-loss-weight 4.0 \
  > train_a0_<TAG>.log 2>&1 &
```

### A1：使用V6-5999初始化

当前动作实验统一从这个已验证基线开始：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
```

V6已有正式结果为67/100，选杯为100/100。合作方不需要重新训练V6，除非研究任务本身就是复现V6。

### A2：V10训练0～500步

目的：保留V6已有动作能力，同时修正下降和闭爪的chunk前部时序。

数据采样比例：

- nominal：60%；
- V6 replay：30%；
- V9 timing：10%。

启动命令：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

GPU_IDS=0,1,2,3,4,5 STEPS=500 \
nohup bash examples/shellgame/run_train_eef_v10_timing_diag_260820.sh \
  > launcher_v10_500.log 2>&1 &
```

关键参数：batch 12、6 GPU、warmup 50、peak LR `3e-6`、save interval 250。

开始训练后检查日志必须显示：

```text
source_mass=nominal:0.60,v6_preservation:0.30,v9_timing:0.10
init=.../v6.../5999/params
frozen=tracker+memory
```

### A3：从500步续训到2000步

必须恢复完整train state，不能只加载step499的params重新启动。

```bash
GPU_IDS=0,1,2,3,4,5 STEPS=2000 \
nohup bash examples/shellgame/run_train_eef_v10_continue_to2k_260820.sh \
  > launcher_v10_continue_2k.log 2>&1 &
```

续训保持原experiment目录和optimizer count，step500以后LR固定在 `3e-7`。

需要保留并测试：

- step499；
- step1000；
- step1500；
- step1999。

已有20 episode结果分别为14/20、15/20、14/20、13/20。因此当前使用step1000，不使用最新step1999。

## 5. 闭环测试

### 5.1 20 episode筛选

```bash
bash examples/shellgame/run_eval_eef_v10_continue_replan8_sweep20_260821.sh
```

统一评测参数：

- seed：260813；
- replan：8；
- maximum policy steps：150；
- 60帧固定历史；
- absolute EEF raw7；
- unguarded；
- 每个episode使用独立MuJoCo/EGL进程。

同时检查：

- `cup_selection_correct`；
- `lift_successes`；
- 视频是否有雪花或截断；
- physics trace中的XY、Z和gripper时序。

### 5.2 100 episode正式测试

当前推荐checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/1000
```

```bash
nohup bash examples/shellgame/run_eval_eef_v10_step1000_replan8_100ep_260821.sh \
  > launcher_eval_v10_step1000_100ep.log 2>&1 &
```

参考结果：

```text
cup_selection_correct = 100/100
lift_successes = 73/100
```

## 6. 每阶段交付内容

合作方完成一个阶段后应提供：

```text
1. 完整启动命令
2. 数据路径
3. 初始化checkpoint
4. 输出checkpoint路径
5. 训练日志
6. eval指标
7. 是否通过该阶段门槛
8. 闭环结果、result.json和视频路径
```

不要仅根据训练loss决定进入下一阶段。Memory阶段必须看分类与shuffle消融，action阶段必须看固定seed闭环成功率。
