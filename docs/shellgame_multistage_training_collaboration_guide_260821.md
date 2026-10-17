# ShellGame MEM + Absolute EEF 多阶段训练协作手册

> 日期：2026-08-21  
> 工作区：`/data2/hzl_workspace_for_pi_mem`  
> OpenPI：`/data2/hzl_workspace_for_pi_mem/openpi-umi`  
> Robosuite：`/data2/hzl_workspace_for_pi_mem/robosuite`  
> 当前推荐部署模型：V10 step1000  
> 本文用途：交给合作方后，可明确知道每个阶段为什么做、需要什么数据、从哪个 checkpoint 初始化、运行什么命令，以及什么结果才允许进入下一阶段。

## 1. 先读结论：不要把整条链当成一次端到端训练

当前有效方案由两条相互独立、最后再连接的训练链组成：

```text
Joint视觉数据
  ├─ M1：frame 0 初始球杯识别
  ├─ M2：三段交换关系识别
  └─ M3：Oracle语义驱动的递归memory组合
          ↓
      M4：全视觉三次交换组合验证
          ↓
      M5：合并tracker，并训练确定性动作诊断头
          ↓
      M6：raw memory → learned queries → Pi0.5 action接口
          ↓
Absolute-EEF数据
  ├─ A0：nominal完整动作训练
  ├─ A1：V6持续纠偏基线（当前作为已验证基础资产）
  ├─ A2：V10 60/30/10行为保留与时序训练，0→500步
  └─ A3：恢复完整train state，500→2000步，按闭环选择checkpoint
          ↓
      replan=8、100 episode正式闭环评测
```

核心原则：

1. M1～M4证明“历史视觉确实能得到球最终在哪个杯子”，不以 action loss 代替这个验证。
2. M5～M6证明 memory 信息能够进入通用 action expert，而不是只进入场景专用分类头。
3. A0～A3固定已经验证的 tracker/memory，只训练 Pi0.5 action expert、action projection 和 time projection。
4. 最终 checkpoint 由闭环成功率选择，不由 validation loss 最低或训练步数最多选择。
5. 合作方若只做动作改进，应直接从已验证 V6-5999 或 V10 checkpoint 开始，不要重复 M1～M6。

## 2. 两种协作方式

### 2.1 推荐：快速协作路径

适合研究采样、纠错数据、动作时序或控制策略的合作方。

直接接收以下资产：

- 已验证 tracker/memory/action-interface checkpoint；
- V6-5999 基线 checkpoint；
- nominal、V6、V9 三份审计通过的 LeRobot 数据；
- 当前代码快照，而不只是 Git commit ID；
- 固定评测脚本和 seed。

然后只执行 A2、A3 和正式闭环评测。这样一次候选实验的主要成本是约 500～2000 个 action fine-tune step，而不是重跑全部 memory 预训练。

### 2.2 完整结构复现路径

适合验证 memory 结构本身的合作方。

顺序执行 M1 → M2 → M3 → M4 → M5 → M6，再用 absolute-EEF nominal 数据执行 A0。M4 的全视觉验证未达到门槛时，不应继续训练 action。

V6 是经过多轮历史数据迭代得到的稳定行为基线。当前 V6 recipe 的原始初始化是 V5-2999，而 V5 又依赖更早的 V4。因此，在没有完整历史资产时，不要把 A0 checkpoint 静默替换成 V6 的初始化；那会成为一个新的实验条件，结果不能与现有 V6/V10 直接比较。

## 3. 统一训练与数据契约

### 3.1 视觉时间轴

- 固定历史：episode 的 frame `0..59`，共 60 帧，`stride=1`。
- 当前帧：frame 60 之后随闭环状态动态更新。
- 模型总输入：60 个固定历史帧 + 1 个当前帧，即 `num_frames=61`。
- 禁止在抓取阶段把最近 60 帧滚动替换成历史；那会改变 memory 的语义。

### 3.2 Prompt

观察阶段：

```text
Observe the ball moving under a cup and remember which cup contains it.
```

动作阶段：

```text
The shell game has ended. Grasp and lift the cup containing the ball.
```

训练、转换和评测必须一致。Prompt 不一致不是无关紧要的字符串差异，会改变 Pi0.5 的语言条件。

### 3.3 Absolute-EEF 表示

- observation state：10 维，`xyz(3) + rot6d(6) + gripper(1)`；
- action：7 维，`world xyz(3) + rotation-vector(3) + gripper(1)`；
- action horizon：16；
- observation position frame：`absolute`；
- rotation convention：`openpi`；
- OSC input：`absolute`；
- raw action 对齐：`observation[i] -> controller_action[i+1]`；
- rotation-vector 在接近 `±π` 时必须使用 canonical branch；
- episode 末端用最后一个 absolute controller command 做 hold padding。

state 使用 rot6d 是为了提供连续、易学习的姿态观察；action 使用 rotation-vector 是因为它必须匹配 robosuite absolute OSC controller 的命令接口。两者维度不同是有意设计，不是数据错误。

### 3.4 GPU/FSDP

六卡训练的标准设置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
--fsdp-devices 6
```

`--fsdp-devices` 表示进程内可见设备数，不要求是 2 的幂；六张可见 GPU 时可以且应该配置为 6。不要让 `CUDA_VISIBLE_DEVICES` 暴露 8 张卡、同时把 `fsdp_devices` 写成 6，这会给设备映射和其他使用者带来歧义。

数据生成和闭环评测才需要：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

普通训练不需要设置 EGL。

### 3.5 代码版本

当前仓库基线 commit 是：

```text
94761fd789d0d417a2da20ced3b70b6c454c00b5
```

但有效 ShellGame 训练、数据和评测脚本中存在大量尚未提交的修改和新文件。因此这个 commit **不足以复现实验**。交付给合作方之前必须完成以下任一项：

- 建立实验分支并提交当前有效文件；或
- 打包完整工作树与依赖锁文件；或
- 提供可应用的 patch，并记录 patch 的 SHA256。

只发送 checkpoint 和上述 commit，合作方无法得到相同的数据过滤、固定历史、V10采样和评测行为。

## 4. 数据资产总表

| 数据 | 规模 | 路径 | 用途 |
|---|---:|---|---|
| absolute joint LeRobot | 6000 ep / 930000 frames | `/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint` | M1～M6 |
| nominal absolute EEF raw7 | 5000 ep / 775000 frames | `/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7` | A0、V6、V10；V10占60% |
| V6 low-stage gated | 1200 ep / 186000 frames | `/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816` | V6行为保留；V10占30% |
| V9 safe balanced recovery | 1200 ep / 186000 frames | `/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819` | V10时序与纠偏；V10占10% |

开始训练前至少检查：

```bash
df -h /data2

test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint/conversion_summary.json
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7/conversion_summary.json
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816/conversion_summary.json
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819/safe_balanced_recovery_v9_oracle_supervision_audit.json
test -f /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819/xy_sampling_metrics_v9.npz
```

建议训练前保留至少 100 GiB 空间。当前 checkpoint 一次异步保存会短时同时占用旧、新两份文件，不能只按最终单个 checkpoint 大小估算。

## 5. 数据准备

已经共享上述四个审计通过的数据目录时，可以跳过本节，直接进入第6节。

### 5.1 Joint数据：生成与LeRobot转换

在 robosuite 环境中执行：

```bash
cd /data2/hzl_workspace_for_pi_mem/robosuite

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python robosuite/scripts/render_shellgame_phase_instruction_joint_dataset_parallel.py \
  --output outputs/shellgame_absolute_joint_dataset \
  --num-episodes 6000 \
  --workers 10 \
  --gpu-ids 0,1,2,3,4,5 \
  --width 224 \
  --height 224 \
  --fps 10 \
  --min-swaps 3 \
  --max-swaps 3 \
  --no-attach-cup-to-gripper
```

这个 joint 生成器保存真正的 absolute joint command，并检查 measured joint/action 语义；不要用 OSC 执行后的 measured `q` 冒充 `JOINT_POSITION` command。

转换：

```bash
python robosuite/scripts/convert_shellgame_to_lerobot_absolute_joint.py \
  --input outputs/shellgame_absolute_joint_dataset \
  --output outputs/shellgame_lerobot_absolute_joint \
  --image-size 224 \
  --fps 10 \
  --action-horizon 16 \
  --phase-instructions \
  --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --workers 16 \
  --png-compress-level 1
```

验收 `conversion_summary.json`：6000 episodes、930000 frames、state/action dim 均为8、horizon为16，且 left/middle/right 不应严重失衡。

### 5.2 Nominal absolute-EEF数据

原始数据生成：

```bash
cd /data2/hzl_workspace_for_pi_mem/robosuite

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python robosuite/scripts/render_shellgame_phase_instruction_dataset_parallel.py \
  --output outputs/shellgame_absolute_eef_phase_instruction_dataset \
  --num-episodes 5000 \
  --workers 10 \
  --gpu-ids 0,1,2,3,4,5 \
  --width 224 \
  --height 224 \
  --fps 10 \
  --min-swaps 3 \
  --max-swaps 3 \
  --osc-input-type absolute \
  --action-representation controller \
  --no-attach-cup-to-gripper
```

转换：

```bash
python robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py \
  --input outputs/shellgame_absolute_eef_phase_instruction_dataset \
  --output outputs/shellgame_lerobot_absolute_eef_raw7 \
  --image-size 224 \
  --fps 10 \
  --action-horizon 16 \
  --observation-position-frame absolute \
  --rot6d-convention openpi \
  --phase-instructions \
  --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --workers 12 \
  --png-compress-level 1
```

验收：5000 episodes、775000 frames、action dim 7、horizon 16、`osc_input_type=absolute`、`rotation_vector_canonicalization=near_pi_dominant_axis_positive_equivalent_branch`。

### 5.3 V6/V9纠错数据的共同原则

纠错数据不是把模型失败动作也放进监督。生成流程是：

1. 模型正常运行到一个 switch state；
2. 隐藏的定位/扰动命令只用于制造状态，不保存为训练 action；
3. 从 switch 后的 observation 开始，只保存连续 Oracle action；
4. 每个可训练 observation 保存完整连续的 horizon=16 Oracle chunk；
5. `action_mask=true` 的行必须全部来自 Oracle；
6. gripper 只有在 XY 对准且到达正确高度后才闭合；
7. left/middle/right、偏移方向、偏移大小和下降高度必须按设计配额平衡。

### 5.4 重新生成V6时的推荐流程

V6数据由V5-2999策略产生闭环前缀，再从switch observation开始保存gated Oracle suffix。先启动V5 policy server：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
nohup uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
  --checkpoint-dir /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v5_260816/absolute_eef7_mixed_correction_v5_balanced1200_60_30_5_5_b12_3k_6gpu_260816/2999 \
  --port 8000 \
  --num-sampling-steps 4 \
  > serve_v6_data.log 2>&1 &
```

在另一终端生成固定配额的1200 episodes：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
nohup uv run python examples/shellgame/generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel.py \
  --host 127.0.0.1 \
  --port 8000 \
  --robosuite-root ../robosuite \
  --output /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_low_stage_gated_v6_balanced1200_260816 \
  --num-episodes 1200 \
  --max-attempts 43200 \
  --workers 6 \
  --dataset-seed 260819 \
  --policy-checkpoint-label v5_2999 \
  --replan-steps 3 \
  --prefix-steps 30,36,42 \
  --width 224 \
  --height 224 \
  --fps 10 \
  > generate_v6_balanced1200.log 2>&1 &
```

原始数据审计：

```bash
uv run python examples/shellgame/audit_onpolicy_eef_low_stage_gated_dataset_v6.py \
  /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_low_stage_gated_v6_balanced1200_260816 \
  --expected-episodes 1200
```

期望配额为left/middle/right各400；high/mid/late为120/240/840；8个方向各150。审计通过后再转换：

```bash
cd /data2/hzl_workspace_for_pi_mem/robosuite

python robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py \
  --input outputs/shellgame_onpolicy_eef_low_stage_gated_v6_balanced1200_260816 \
  --output outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816 \
  --image-size 224 \
  --fps 10 \
  --action-horizon 16 \
  --observation-position-frame absolute \
  --rot6d-convention openpi \
  --phase-instructions \
  --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --grasp-phase-ids 8,9,10,11 \
  --workers 12 \
  --png-compress-level 1
```

转换后再次核对 `conversion_summary.json`：1200 episodes、186000 frames、action dim 7、horizon 16。V6没有像V9 wrapper那样在转换时自动执行全部严格审计，因此原始数据audit不能跳过。

### 5.5 重新生成V9时的推荐流程

V10会消费V6与V9，但一般不需要反复重建它们。只有数据研究任务才执行本节。

先在独占GPU上启动产生状态所需的V6 policy server：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
nohup uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
  --checkpoint-dir /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999 \
  --port 8000 \
  --num-sampling-steps 4 \
  > serve_v9_data.log 2>&1 &
```

再生成1200个固定设计槽。若某些槽重试耗尽，保持同一输出目录并再次运行；脚本默认 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
nohup uv run python examples/shellgame/generate_onpolicy_eef_safe_balanced_recovery_dataset_v9_parallel.py \
  --host 127.0.0.1 \
  --port 8000 \
  --robosuite-root ../robosuite \
  --output /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819 \
  --num-episodes 1200 \
  --max-attempts 86400 \
  --workers 6 \
  --dataset-seed 260825 \
  --policy-checkpoint-label v6_5999 \
  --prefix-steps 30,36,42 \
  --width 224 \
  --height 224 \
  --fps 10 \
  > generate_v9_balanced1200.log 2>&1 &
```

严格审计原始数据：

```bash
uv run python examples/shellgame/audit_onpolicy_eef_safe_balanced_recovery_dataset_v9.py \
  /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819 \
  --expected-episodes 1200 \
  --require-complete-quota
```

转换脚本会再次执行审计，并检查每个 horizon window：

```bash
uv run python examples/shellgame/convert_shellgame_onpolicy_safe_balanced_recovery_v9_to_lerobot_raw_action.py \
  --input /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819 \
  --output /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819 \
  --image-size 224 \
  --fps 10 \
  --action-horizon 16 \
  --observation-position-frame absolute \
  --rot6d-convention openpi \
  --phase-instructions \
  --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --grasp-phase-ids 8,9,10,11 \
  --workers 12 \
  --png-compress-level 1
```

最后构造V10采样依赖的逐行真实XY误差sidecar：

```bash
uv run python examples/shellgame/build_eef_xy_sampling_metrics_v9.py
```

必须生成：

```text
safe_balanced_recovery_v9_oracle_supervision_audit.json
xy_sampling_metrics_v9.npz
```

训练V10前，V6也必须通过：

```bash
uv run python examples/shellgame/audit_onpolicy_eef_low_stage_gated_dataset_v6.py \
  /data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_onpolicy_eef_low_stage_gated_v6_balanced1200_260816 \
  --expected-episodes 1200
```

## 6. Memory多阶段训练

以下命令均从 OpenPI 根目录执行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
```

每次运行把 `<TAG>` 替换为唯一实验标识，例如合作方姓名和日期。不要对已有正式实验使用 `--overwrite`。

### M1：frame 0初始球杯分类

目的：证明单帧图像中球的位置视觉信号可被当前SigLIP patch embedding读取。它排除“小球太小、视觉前端完全看不到”的可能性。

输入：absolute-joint LeRobot 数据的 frame 0。  
训练：冻结视觉主干，只训练轻量分类probe。  
门槛：held-out `initial_cup_accuracy` 应达到接近100%；否则停止后续memory训练。

```bash
CUDA_VISIBLE_DEVICES=0 \
nohup uv run python examples/shellgame/train_frame0_initial_cup_probe.py \
  --exp-name frame0_initial_cup_linear_<TAG> \
  --steps 300 \
  --batch-size 60 \
  --num-workers 8 \
  --fsdp-devices 1 \
  --eval-interval 50 \
  --eval-batches 10 \
  > train_m1_frame0_<TAG>.log 2>&1 &
```

已验证参考checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params
```

### M2：三段交换关系视觉分类

目的：把每段10帧交换片段分类为 `left-middle`、`left-right` 或 `middle-right`。此阶段保留空间网格，使用固定2×2 pooling后的K=64 tokens和factorized spatial/temporal Transformer。

输入：历史frame 20..49，按三个10帧片段处理。  
门槛：三个relation accuracy和三段联合held-out accuracy均为100%。只在训练集100%不算通过。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_pair_fixed_grid_probe.py \
  --exp-name swap_pair_full600_b18_<TAG> \
  --steps 600 \
  --warmup-steps 50 \
  --peak-lr 3e-4 \
  --batch-size 18 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 200 \
  --eval-batches 100 \
  > train_m2_swap_pair_<TAG>.log 2>&1 &
```

参考checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params
```

### M3：递归compact memory和一次history read

目的：在Oracle初始杯和Oracle交换关系输入下，验证 `M=128, D=64` 的递归memory updater能连续组合三次交换，并可由一次显式history read解码最终杯位。它单独验证memory容量和更新接口，不混入视觉relation误差。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_oracle_single_history_read_adapter_probe.py \
  --exp-name oracle_single_read_for_visual_relation_600_<TAG> \
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
  > train_m3_oracle_memory_<TAG>.log 2>&1 &
```

门槛：normal held-out为100%；把交换关系设为 `roll` 或 `shuffle_batch` 后应显著崩溃，而不是仍维持100%。

参考checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params
```

### M4：全视觉组合验证（零步训练）

目的：把M1的初始杯识别、M2的三段交换关系、M3的递归memory真正串起来。这里不训练action，也不允许使用ground-truth final slot。

先运行normal：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py \
  --exp-name fully_visual_relation_normal_<TAG> \
  --init-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params \
  --initial-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params \
  --memory-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params \
  --steps 0 \
  --batch-size 72 \
  --num-workers 2 \
  --fsdp-devices 6 \
  --eval-batches 9
```

然后至少运行两个反事实消融：

```bash
# 打乱frame 0，其他输入保持不变
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py \
  --exp-name fully_visual_relation_shuffle_initial_<TAG> \
  --steps 0 \
  --batch-size 72 \
  --num-workers 2 \
  --fsdp-devices 6 \
  --eval-batches 9 \
  --initial-mode shuffle_batch

# 打乱三段交换视频，其他输入保持不变
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py \
  --exp-name fully_visual_relation_shuffle_swaps_<TAG> \
  --steps 0 \
  --batch-size 72 \
  --num-workers 2 \
  --fsdp-devices 6 \
  --eval-batches 9 \
  --video-mode shuffle_swaps
```

注意：消融命令若不使用默认参考checkpoint，必须显式传入与normal完全相同的三个checkpoint。

通过门槛：normal初始杯、三个relation、最终杯位均100%；shuffle frame0约回到33%，shuffle swaps接近随机三关系组合水平。参考结果为normal 100%、shuffle frame0约33%、shuffle swaps约4.17%。

### M5：合并tracker并训练确定性动作诊断头

目的：把M1/M2/M3参数合成单一tracker checkpoint，并用 `final-slot probability + state` 预测frame59之后的首个joint action chunk。这个head是链路诊断，不是最终通用action architecture。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_fully_visual_joint_action_probe.py \
  --exp-name fully_visual_joint_action_head300_b72_<TAG> \
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
  --cup-eval-interval 50 \
  --cup-eval-episodes 24 \
  > train_m5_joint_diagnostic_<TAG>.log 2>&1 &
```

门槛：memory分类仍为100%，并确认预测的首段FK端点受memory控制、靠近正确杯。不要把这个分类概率head当成最终action部署接口。

参考checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_fully_visual_joint_action_260810/fully_visual_joint_action_head300_b72_260810/299/params
```

### M6：raw memory接入Pi0.5 action expert

目的：用16个learned query从 `[128,64]` raw memory读取信息，经action-suffix cross-attention影响Pi0.5 flow action expert。这个阶段去掉场景专用final-slot动作头，建立可继续迁移到joint或EEF action的通用接口。

第一次建立接口时不要使用 `--restore-memory-interface`，否则会错误地假设接口参数已经存在。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_three_swap_query_crossattn_pi_joint_action_probe.py \
  --exp-name query_crossattn_pi_flow_action300_<TAG> \
  --tracker-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_fully_visual_joint_action_260810/fully_visual_joint_action_head300_b72_260810/299/params \
  --steps 300 \
  --warmup-steps 30 \
  --peak-lr 3e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10 \
  --cup-eval-interval 50 \
  --cup-eval-episodes 24 \
  --query-tokens 16 \
  --query-width 256 \
  --query-depth 2 \
  --query-heads 4 \
  --action-cross-attention-heads 8 \
  > train_m6_memory_to_pi_<TAG>.log 2>&1 &
```

门槛不只是normal离线误差下降，还必须做memory反事实：

- normal endpoint/cup约98%；
- shuffled memory约43%；
- zero memory约35%；
- 闭环只执行approach段时，正确选杯约18～19/20。

若normal、shuffle、zero几乎相同，说明action expert没有真正读memory，不能进入A0。

参考checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/query_crossattn_pi_flow_action300_b12_260810/299/params
```

## 7. Absolute-EEF action训练

### A0：nominal完整动作训练（完整复现时执行）

目的：从M6的memory-to-action接口迁移到absolute EEF7，学习approach、持续下降、闭爪和抬升。tracker、memory、query resampler和memory cross-attention冻结；只训练Pi0.5 action expert、action projection和time projection。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_old_tracker_full_absolute_eef.py \
  --exp-name absolute_eef7_old_tracker_phase_balanced_b12_2k_<TAG> \
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
  > train_a0_nominal_eef_<TAG>.log 2>&1 &
```

这里的eval loss只验证teacher-forced动作拟合。至少要做20 episode闭环测试，分别统计selection、接近杯、下降、闭爪、抬升，不要直接根据loss进入下一阶段。

### A1：V6行为基线

当前V6由历史V5-2999初始化，再用60/30/5/3/2动态phase-aware recipe训练3000步并以低LR续训到5999。为了让不同合作实验共享同一起点，建议把V6-5999作为版本化基础资产直接分发：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
```

若确实需要复现V6，并且V5-2999已提供：

```bash
nohup bash examples/shellgame/run_train_eef_v6_dynamic_phase_260816.sh \
  > launcher_train_v6_<TAG>.log 2>&1 &
```

续训到6000总步数：

```bash
nohup bash examples/shellgame/run_train_eef_v6_continue_to6k_260816.sh \
  > launcher_continue_v6_<TAG>.log 2>&1 &
```

V6已有正式基线为67/100，selection为100/100。换初始化或改数据后应使用新实验名，不能继续称为V6复现。

### A2：V10初始500步

目的：从V6-5999保留已有抓取行为，同时只用少量V9数据修正chunk前部下降/闭爪时序。

V10全局采样质量：

- nominal 60%；
- V6 preservation replay 30%；
- V9 timing replay 10%。

其中V9的10%再拆分为hard、low、aligned、front-3 descent、close-within-3和early lift。采样器只选择已有Oracle行，不编辑或合成action。

复现当前正式实验时可直接运行已有的带磁盘检查runner：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

GPU_IDS=0,1,2,3,4,5 STEPS=500 \
nohup bash examples/shellgame/run_train_eef_v10_timing_diag_260820.sh \
  > launcher_train_v10_formal_reproduction.log 2>&1 &
```

合作方创建新实验时必须使用唯一exp name。下面的直接命令与runner训练参数一致，并避免覆盖正式目录：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

V10_EXP=absolute_eef7_v10_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_<TAG>

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
nohup uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag.py \
  --exp-name "${V10_EXP}" \
  --init-checkpoint /data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params \
  --steps 500 \
  --warmup-steps 50 \
  --peak-lr 3e-6 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 125 \
  --eval-batches 20 \
  --save-interval 250 \
  --keep-period 250 \
  --gripper-loss-weight 4.0 \
  > train_v10_<TAG>.log 2>&1 &
```

实际关键超参数：batch 12、6 GPU、warmup 50、peak LR `3e-6`、500 steps、eval每125步、save每250步、gripper loss weight 4。

训练启动后检查日志中必须出现：

```text
source_mass=nominal:0.60,v6_preservation:0.30,v9_timing:0.10
init=.../v6.../5999/params
frozen=tracker+memory
```

如果数据source mass、初始化或冻结范围不同，应立即停止，这不是同一个V10实验。

### A3：恢复完整train state，续训到2000步

目的：不是重新从step500权重启动一个新optimizer，而是恢复step499下的完整model、optimizer和step count。500步后固定使用原schedule的terminal LR `3e-7`。

复现正式实验时，已有runner会绑定正式A2的exp name并检查step499 train state：

```bash
GPU_IDS=0,1,2,3,4,5 STEPS=2000 \
nohup bash examples/shellgame/run_train_eef_v10_continue_to2k_260820.sh \
  > launcher_continue_v10_formal_reproduction.log 2>&1 &
```

合作方新实验必须复用A2完全相同的 `${V10_EXP}`，直接执行resume脚本。不要加 `--overwrite`，不要重新warmup：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

V10_EXP=absolute_eef7_v10_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_<TAG>

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
nohup uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_continue.py \
  --exp-name "${V10_EXP}" \
  --steps 2000 \
  --warmup-steps 50 \
  --peak-lr 3e-6 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 1 \
  --gripper-loss-weight 4.0 \
  > continue_v10_<TAG>.log 2>&1 &
```

候选checkpoint测试顺序：

1. step499；
2. step1000；
3. step1500；
4. step1999。

已有同协议20-episode结果：14/20、15/20、14/20、13/20。因此当前选step1000，而不是最新step1999。训练越久不保证闭环越好。

## 8. 闭环评测

### 8.1 20 episode checkpoint筛选

评测必须使用：

- `seed=260813`；
- `replan_steps=8`；
- `max_policy_steps=150`；
- fixed history frame 0..59；
- `num_frames=61, stride=1`；
- absolute EEF raw7；
- unguarded，即不读取模拟器真值杯位做XY-before-Z控制；
- 每个episode独立MuJoCo/EGL进程；
- 保存视频、physics trace和result.json。

已有批量筛选runner：

```bash
bash examples/shellgame/run_eval_eef_v10_continue_replan8_sweep20_260821.sh
```

评测先看两个指标：

- `cup_selection_correct`：memory/条件链是否选对杯；
- `lift_successes`：action/control是否完成抓取。

如果selection下降，优先查history、prompt和memory restore；如果selection仍100%而lift下降，优先查EEF action、下降时XY漂移、闭爪和抬升时序。

### 8.2 100 episode正式评测

当前推荐checkpoint：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/1000
```

运行：

```bash
nohup bash examples/shellgame/run_eval_eef_v10_step1000_replan8_100ep_260821.sh \
  > launcher_eval_v10_step1000_100ep.log 2>&1 &
```

参考结果：

```text
lift_successes        = 73/100
cup_selection_correct = 100/100
```

结果文件：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/eef7_v10_continue_step1000_replan8_isolated100_seed260813_260821/result.json
```

注意：V10的73/100相比V6既有67/100是正向趋势，但现有配对检验 `p=0.263`，不能宣称统计显著。论文级比较还应补做V6在完全相同replan=8协议下的独立100 episode复测。

## 9. 每阶段交付物与Go/No-Go门槛

| 阶段 | 必交付 | Go条件 | No-Go条件 |
|---|---|---|---|
| 数据 | conversion summary、audit、数据路径 | 数量/维度/prompt/action对齐全部通过 | 缺episode、雪花帧、action mask含模型动作、horizon错位 |
| M1 | checkpoint、log | held-out初始杯接近100% | 仅train高、held-out接近随机 |
| M2 | checkpoint、log、relation指标 | 三段及联合held-out 100% | 任何relation不能泛化 |
| M3 | checkpoint、normal/roll/shuffle结果 | normal 100%，反事实显著下降 | shuffle后仍100%或normal不稳定 |
| M4 | 三组完整eval结果 | normal全视觉最终杯100%，两种shuffle崩溃 | 全视觉约33%或不依赖历史 |
| M5 | 合并checkpoint、FK/动作诊断 | memory正确且动作端点受其控制 | 只分类正确，动作不随memory变 |
| M6 | checkpoint、normal/shuffle/zero结果 | normal显著高于shuffle/zero | action expert忽略memory |
| A0/A1 | checkpoint、分段离线指标、20ep闭环 | selection保持，抓取链可运行 | loss下降但不下降/不闭爪/selection退化 |
| A2/A3 | 每个候选step的20ep结果 | 按闭环选最佳step | 仅根据loss或最新step选模型 |
| 正式评测 | result.json、100视频、physics trace | 无雪花/截断，100ep完整 | 复用损坏renderer、缺episode或控制参数不同 |

## 10. 合作实验的命名和记录模板

每个实验至少记录：

```text
experiment_id:
owner:
date:
code_commit:
working_tree_patch_sha256:
init_checkpoint:
dataset_roots:
dataset_audit_files:
train_script:
full_command:
visible_gpus:
fsdp_devices:
steps:
learning_rate_schedule:
frozen_modules:
changed_variable:
control_experiment:
checkpoint_candidates:
20ep_selection:
20ep_lift_success:
100ep_selection:
100ep_lift_success:
result_json:
video_root:
known_failures:
conclusion:
```

实验名建议：

```text
<stage>_<single-change>_<batch>_<steps>_<gpu-count>_<owner>_<yymmdd>
```

一次实验只改变一个主变量。改变数据比例、初始化checkpoint、replan和控制guard中的任意两项后，结果就不再是严格控制变量对照。

## 11. 监控与常见问题

### 11.1 后台进程看起来有两个

`uv run python ...`通常会有一个uv父进程和一个实际Python子进程，这不表示启动了两次训练。用父子PID和checkpoint目录判断：

```bash
ps -eo user,pid,ppid,etime,cmd | grep -E 'uv run python|train_old_tracker|train_three_swap' | grep -v grep
```

### 11.2 eval loss下降但闭环变差

这是本项目已经多次观察到的现象。teacher-forced action loss平均了大量容易的nominal行，不能表示低位视觉纠偏、闭爪时机或长期闭环稳定性。解决方式是保存中间checkpoint并做固定20ep闭环筛选。

### 11.3 选杯正确但抓取失败

这不表示memory没影响action。memory使approach方向选择正确；后续失败来自action suffix的持续XY对准、下降、闭爪和抬升。应检查physics trace中的EEF-to-target XY error、Z、gripper command和cup lift，而不是重新训练杯位分类器。

### 11.4 视频后半段雪花

雪花通常来自renderer/context生命周期或并行评测资源复用。正式协议使用每episode独立MuJoCo/EGL进程，并保存physics trace。出现雪花的episode和统计结果都应标为无效并重跑，不能只相信result计数。

### 11.5 checkpoint恢复失败或step重新从0开始

续训必须存在：

```text
<experiment>/<step>/train_state
```

只加载`params`是权重初始化，不是完整resume。V10 A3必须恢复optimizer count，否则会重启warmup并改变实验。

### 11.6 硬盘不足

训练前看 `/data2` 剩余空间，优先删除已确认无用的中间checkpoint、JAX编译缓存和可重建的HF缓存；不要删除当前最佳checkpoint、其train_state、正式评测视频/trace和四个训练数据根目录。任何清理都先生成目标清单并由负责人确认。

## 12. 建议的合作任务拆分

| 角色 | 负责内容 | 不应自行改变 |
|---|---|---|
| 数据负责人 | 生成、转换、审计、配额和sidecar | prompt、action对齐、history边界 |
| Memory负责人 | M1～M4和反事实消融 | action数据与控制器 |
| 接口负责人 | M5～M6、memory shuffle/zero | 显式final-slot场景trick |
| Action负责人 | A0～A3、采样比例、checkpoint筛选 | tracker/memory权重与固定历史 |
| 评测负责人 | 独立进程闭环、视频、trace、统计 | seed、replan、guard和成功定义 |

每个负责人交付上游checkpoint或数据时，必须同时交付门槛结果；下游不能只凭“训练完成”开始运行。

## 13. 当前推荐资产清单

```text
M1 initial cup:
checkpoints/pi0_shellgame_frame0_initial_cup_probe_260807/frame0_initial_cup_linear_260807/299/params

M2 swap relation:
checkpoints/pi0_shellgame_three_swap_pair_fixed_grid_probe_260809/swap_pair_full600_b18_260809/599/params

M3 recurrent memory:
checkpoints/pi0_shellgame_three_swap_oracle_single_history_read_adapter_260809/oracle_single_read_for_visual_relation_600_b72_260810/599/params

M5 combined tracker:
checkpoints/pi0_shellgame_three_swap_fully_visual_joint_action_260810/fully_visual_joint_action_head300_b72_260810/299/params

M6 generic memory/action interface:
checkpoints/pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/query_crossattn_pi_flow_action300_b12_260810/299/params

V6 stable EEF baseline:
checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params

V10 recommended deployment checkpoint:
checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/1000
```

上述相对路径均以 `/data2/hzl_workspace_for_pi_mem/openpi-umi` 为根目录。

## 14. 关联文档

模型迭代、失败版本、所有消融和阶段结论详见：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/docs/shellgame_mem_absolute_eef_stage_summary_260821.md
```

本手册负责“如何交接和执行”；阶段总结负责“为什么最终采用这条路径，以及历史实验给出了什么证据”。
