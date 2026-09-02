# ShellGame V10 架构：真机标准示范训练与部署 Recipe

> 适用范围：使用真机采集的标准成功示范，训练 ShellGame 的视觉记忆与抓取策略，并部署回同一真机。  
> 本文只保留原始离线训练主线，不包含仿真阶段的纠错数据、on-policy suffix 或混合纠错训练。  
> 仓库根目录：`/data2/hzl_workspace_for_pi_mem/openpi-umi`

## 1. 先明确要复现的内容

这里复用的是 V10 阶段已经验证过的模型结构和输入输出契约：

```text
固定 60 帧历史 RGB
  ├─ frame 0：识别球最初位于左/中/右杯
  ├─ frame 20..29：识别第一次交换
  ├─ frame 30..39：识别第二次交换
  └─ frame 40..49：识别第三次交换
          ↓
  recurrent compact memory [128, 64]
          ↓
  16 个 learned queries + memory cross-attention
          ↓
最新当前帧 + 10D EEF state + prompt
          ↓
Pi0.5 action expert
          ↓
16 × 7 absolute EEF action chunk
```

最终输入输出必须保持：

- 历史图像：固定 `frame 0..59`，闭环期间永远不滑动；
- 当前图像：第 61 个视觉位置，每次推理替换为最新真机图像；
- 图像：`224×224 RGB uint8`；
- 频率：参考 recipe 为 `10 Hz`；
- state：`xyz(3) + rot6d(6) + gripper_width(1)`，共 10 维；
- action：`world xyz(3) + world rotation-vector(3) + gripper_command(1)`，共 7 维；
- action horizon：16；
- 模型内部 action pad 到 32 维，但 loss 只作用于前 7 维；
- gripper loss weight：4.0；
- 推理采样步数：4；
- 推荐闭环方式：预测 16 步，先执行前 8 步，再用最新观测重新规划。

## 2. 推荐执行路线

```text
R0  固定相机、坐标系和控制器语义
 ↓
R1  采集标准成功示范并记录完整标签
 ↓
R2  把每条 episode 规范化到 V10 固定时间轴
 ↓
R3  转为 LeRobot v2.1，生成真机 norm_stats
 ↓
R4  数据审计、回放和 episode-level 划分
 ↓
M0  在真机 held-out 视频上检查现有 tracker 是否可迁移
 ├─ 通过：直接进入 A0，只训练真机 action expert
 └─ 不通过：执行 M1～M6，使用真机视频重训 memory 链
 ↓
A0  只用真机标准示范训练 absolute-EEF action
 ↓
E0  离线验证、memory 消融、影子推理
 ↓
D0  限速空载测试 → 杯上方测试 → 低速抓取 → 正式闭环
```

## 3. R0：采集前冻结真机契约

训练前先写一份不可随意改动的 `robot_contract.json`，至少包含：

```json
{
  "control_hz": 10,
  "image_size": [224, 224],
  "base_camera_serial": "<SERIAL>",
  "wrist_camera_serial": "<SERIAL>",
  "world_frame": "<ROBOT_BASE_OR_CALIBRATED_WORLD>",
  "eef_frame": "<TOOL_FRAME>",
  "rotation_state": "rot6d_openpi",
  "rotation_action": "world_rotation_vector",
  "action_semantics": "absolute_eef_target",
  "gripper_open_value": -1.0,
  "gripper_close_value": 1.0,
  "action_horizon": 16
}
```

以下内容必须在采集、训练、policy client 和机器人控制器中完全一致：

1. `xyz` 使用哪个坐标系；
2. 末端姿态是 tool frame 还是 flange frame；
3. 四元数顺序是 `xyzw` 还是 `wxyz`；
4. rot6d 的生成约定；
5. rotation-vector 的分支约定；
6. gripper 数值的开/合方向；
7. action 是 absolute pose、delta pose，还是速度；
8. observation 和 action 的时间戳含义。

本 recipe 只适用于 **absolute EEF target**。如果真机控制器接收 delta 或速度，不要只改字段名称；必须在执行端把模型输出的绝对目标安全地转换为控制器命令，或重新定义数据与模型 action contract。

## 4. R1：真机数据采集

### 4.1 每条 episode 必须保存的内容

建议保存原始高精度时间戳，不要只依赖数组下标：

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `base_rgb` | `[T,H,W,3] uint8` | 固定第三人称相机，memory 的主要视觉来源 |
| `wrist_rgb` | `[T,H,W,3] uint8` | 腕部相机，供当前帧动作控制使用 |
| `camera_timestamp` | `[T] float64` | 相机时间戳 |
| `eef_pos_world` | `[T,3] float32` | 当前 measured EEF world xyz |
| `eef_quat_world` | `[T,4] float32` | 当前 measured EEF world quaternion，并记录顺序 |
| `gripper_width` | `[T,1] float32` | 当前 measured gripper width |
| `commanded_eef_pos_world` | `[T,3] float32` | 实际下发的绝对 EEF xyz 目标 |
| `commanded_eef_rotvec_world` | `[T,3] float32` | 实际下发的绝对旋转向量目标 |
| `commanded_gripper` | `[T,1] float32` | 实际下发的夹爪命令 |
| `joint_pos` | `[T,J] float32` | M5/M6 或诊断所需，建议始终记录 |
| `phase_id` | `[T] int64` | reveal/cover/swap/settle/approach/descend/grasp/lift |
| `initial_ball_slot` | scalar | `left/middle/right` |
| `swap_pairs` | 3 项 | 每次交换的杯对 |
| `final_ball_slot` | scalar | 三次交换后的空间杯位 |
| `success` | bool | 是否抓住并抬起正确杯子 |
| `failure_reason` | string | 采集失败原因，仅用于剔除和审计 |

相机、state 和 command 应按时间戳同步；建议记录原始频率后统一重采样到 10 Hz，而不是采集时静默丢帧。

### 4.2 episode 的规范时间轴

若直接复用当前 V10 tracker 和训练脚本，每条标准化后的 episode 必须是 155 帧：

| 帧范围 | 帧数 | 内容 |
|---|---:|---|
| `0..9` | 10 | reveal：球和初始杯位清晰可见 |
| `10..19` | 10 | cover：用杯盖住球并稳定 |
| `20..29` | 10 | swap 1，完整动作必须落在本窗口 |
| `30..39` | 10 | swap 2 |
| `40..49` | 10 | swap 3 |
| `50..59` | 10 | settle，三个杯停止运动；frame 59 是 memory/action 交界 |
| `60..88` | 29 | approach/选择目标杯 |
| `89..108` | 20 | descend |
| `109..118` | 10 | grasp |
| `119..153` | 35 | lift/hold |
| `154` | 1 | terminal hold |

真机动作持续时间不一致时，应先按 phase 切段，再在每一段内部按时间重采样到表中的帧数。不要把整条视频直接均匀压成 155 帧，否则三段交换很可能错过硬编码窗口。

### 4.3 数据平衡

最低要求：

- 初始球位 left/middle/right 尽量均衡；
- 最终空间杯位 left/middle/right 尽量均衡；
- 每个 swap stage 都覆盖 `left-middle / left-right / middle-right`；
- 光照、杯子起始微偏、机器人初始 pose 和操作速度要有真实但安全的变化；
- train/val/test 必须按 episode 划分，不能把同一 episode 的不同 action row 分到不同集合。

仿真 nominal recipe 使用了 5000 条成功示范。真机不必机械照搬这个数量，但应分批推进：先做小规模 pipeline smoke test，再持续增加成功示范，直到 held-out memory 和闭环指标不再明显受数据量限制。任何阶段都不要为了凑数量保留时间错位、遮挡不完整或控制器异常的 episode。

## 5. R2/R3：时间对齐与 LeRobot v2.1 转换

### 5.1 先判断是否需要 `+1` 对齐

这是仿真转换器最容易被误用到真机的地方。

仿真数据记录的是“执行本行 action 后得到本行 observation”，所以原转换器使用：

```text
observation[i] -> action[i+1]
```

真机采集器必须通过时间戳确定语义：

- 如果 row `i` 保存的是下发 command 前的 observation，则训练配对通常是 `observation[i] -> command[i]`；
- 如果 row `i` 保存的是执行 command 后的 observation，则使用 `observation[i] -> command[i+1]`；
- 如果相机、机器人 state 和 command 异步，按时间戳匹配“该 observation 之后第一个真正执行的 command”，不要凭数组长度猜测。

转换后随机抽 20 条样本，把 `observation + 未来16步 action` 可视化或在离线控制器中回放。若第一步经常指向已经到达的旧 pose，通常就是 action 对齐错了一帧。

### 5.2 目标 LeRobot schema

真机数据应转换为与 nominal absolute EEF 数据一致的核心字段：

```text
observation.robot0_eef_pos                    [2,3] float32
observation.robot0_eef_rot_axis_angle         [2,6] float32
observation.robot0_eef_rot_axis_angle_wrt_start [2,6] float32
observation.robot0_gripper_width              [2,1] float32
actions                                       [16,7] float32
observation.left_wrist_0_rgb_0                [224,224,3] image  # wrist
observation.left_wrist_0_rgb_1                [224,224,3] image  # base
timestamp                                     scalar float32
frame_index                                   scalar int64
episode_index                                 scalar int64
index                                         scalar int64
task_index                                    scalar int64
action_mask                                   scalar bool
phase_id                                      scalar int64
```

单臂 state 实际使用第 0 行：

```text
state10 = [world_xyz(3), world_rot6d(6), gripper_width(1)]
```

`actions[t]` 是从当前 observation 对齐后的未来 16 步：

```text
[world_x, world_y, world_z,
 world_rotvec_x, world_rotvec_y, world_rotvec_z,
 gripper_command]
```

episode 尾部 padding 对 absolute action 的正确语义是重复最后一个绝对 pose，并保留最终 gripper intent；不能补零。

### 5.3 姿态和夹爪规范化

- observation 姿态统一转成 OpenPI rot6d；
- action 姿态统一为 world rotation-vector；
- 接近 `pi` 的等价 rotation-vector 应固定到单一连续分支，避免数值跳变约 `2*pi`；
- gripper command 统一映射到 `[-1,1]`，推荐 `-1=open, +1=close`；
- 真机执行端再把 `[-1,1]` 映射回夹爪硬件单位。

### 5.4 prompt

训练与部署必须使用完全相同的文本：

```text
Observe the ball moving under a cup and remember which cup contains it.
Grasp and lift the cup containing the ball.
```

frame 0..59 使用 observe prompt；从 action 阶段开始使用 grasp prompt。若数据转换发生 action `+1` shift，prompt 也必须做相同 shift。

### 5.5 norm stats

必须用真机训练集重新生成并保存：

```text
<REAL_LEROBOT_ROOT>/norm_stats.json
```

不要沿用仿真的 state/action norm stats。生成后至少检查：

- state/action 无 NaN、Inf；
- xyz 范围位于真机安全工作空间；
- rot6d 每帧合法且连续；
- action rotation-vector 无异常 `2*pi` 跳变；
- gripper 符号和范围正确；
- q01/q99 不被少量坏帧支配。

## 6. R4：训练前数据门槛

训练前必须全部通过：

```text
[ ] 每条 episode 恰好 155 帧，frame_index=0..154
[ ] history 0..59 完整，三次 swap 分别落在规定窗口
[ ] base/wrist/state/action 时间戳对齐
[ ] action horizon=16，action dim=7
[ ] action 是 absolute world EEF target，不是 measured pose 冒充 command
[ ] terminal padding 是最后 pose hold
[ ] left/middle/right 初始位和最终位分布可接受
[ ] 三个 swap stage 的三类 pair 都有覆盖
[ ] train/val/test 按 episode 隔离
[ ] norm_stats 只由 train episodes 计算
[ ] 随机可视化至少 20 条 action chunk，无一帧错位
[ ] 选取 5～10 条示范做低速 command replay，轨迹和夹爪语义正确
```

注意：当前 `train_old_tracker_full_absolute_eef.py` 硬编码了仿真数据路径和 `LAST_EPISODE_FRAME=154`。真机训练前应复制出 real 版本或将数据根目录参数化；不要覆盖原脚本，也不要直接修改已有实验数据。

## 7. M0：先测现有 V10 memory 能否迁移到真机

在训练 action 之前，先用真机 held-out 历史做以下检查：

1. frame 0 初始杯分类；
2. 三段 swap pair 分类；
3. 最终杯位分类；
4. shuffle history / wrong-episode history 消融。

推荐进入 A0 的门槛：

```text
initial cup accuracy >= 95%
each swap-pair accuracy >= 95%
final cup accuracy >= 95%
shuffle/wrong-history 后 final cup accuracy 明显下降
```

若达标，说明现有 tracker 对真机视觉域可用，可以冻结 tracker、compact memory、query resampler 和 memory cross-attention，只训练真机 action expert。

若不达标，不要靠 action loss 期待模型自动修复 memory；按 M1～M6 使用真机数据重新训练。

## 8. M1～M6：真机 memory 原始分阶段 recipe

以下命令来自已验证的原始分阶段 recipe。执行前需把各脚本中的数据根目录和 label 路径参数化为 `<REAL_LEROBOT_ROOT>` / `<REAL_RAW_ROOT>`；标签来自真机采集记录，不依赖 simulator pose。

所有命令默认：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
```

### M1：frame 0 初始杯分类

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/shellgame/train_frame0_initial_cup_probe.py \
  --exp-name real_frame0_initial_cup \
  --steps 300 \
  --batch-size 60 \
  --num-workers 8 \
  --fsdp-devices 1 \
  --eval-interval 50 \
  --eval-batches 10
```

门槛：episode-held-out 初始杯分类不低于 95%，目标是接近 100%。

### M2：三段 swap pair 分类

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/train_three_swap_pair_fixed_grid_probe.py \
  --exp-name real_swap_pair_full600 \
  --steps 600 \
  --warmup-steps 50 \
  --peak-lr 3e-4 \
  --batch-size 18 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 200 \
  --eval-batches 100
```

门槛：三个 stage 各自和 27-way 联合 held-out 准确率均不低于 95%；shuffle swap clip 后应明显下降。

### M3：递归 compact memory

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/train_three_swap_oracle_single_history_read_adapter_probe.py \
  --exp-name real_oracle_single_read_600 \
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
  --pair-mode correct
```

门槛：正常输入 final-slot held-out 不低于 95%；打乱 relation 后回到明显更低水平。

### M4：组合成全视觉 memory 并验证

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py \
  --exp-name real_fully_visual_memory_eval \
  --init-checkpoint <M2_PARAMS> \
  --initial-checkpoint <M1_PARAMS> \
  --memory-checkpoint <M3_PARAMS> \
  --steps 0 \
  --batch-size 72 \
  --num-workers 2 \
  --fsdp-devices 6 \
  --eval-batches 9
```

门槛：初始杯、三个 swap 和最终杯位均不低于 95%，并通过 frame-0、swap-clip、wrong-episode 三类消融。

### M5：合并 tracker，并训练诊断动作头

M5 的目的只是确认正确 memory 能影响动作目标，不是最终部署模型。原脚本使用 absolute joint 轨迹，因此真机采集时应保留 measured `joint_pos`，并为真机数据准备同 schema 的 action target。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/train_three_swap_fully_visual_joint_action_probe.py \
  --exp-name real_fully_visual_joint_action \
  --init-checkpoint <M2_PARAMS> \
  --initial-checkpoint <M1_PARAMS> \
  --memory-checkpoint <M3_PARAMS> \
  --steps 300 \
  --warmup-steps 30 \
  --peak-lr 3e-4 \
  --batch-size 72 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10
```

门槛：正常 memory 的目标杯动作正确率显著高于 shuffle/zero memory。

### M6：memory 接入 Pi0.5 action expert

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
uv run python examples/shellgame/train_three_swap_query_crossattn_pi_joint_action_probe.py \
  --exp-name real_query_crossattn_pi_action \
  --tracker-checkpoint <M5_PARAMS> \
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
  --action-cross-attention-heads 8
```

首次构建真机 memory interface 时不要加 `--restore-memory-interface`。门槛仍是 normal memory 明显优于 shuffle/zero memory。

## 9. A0：只用真机标准示范训练 absolute EEF action

### 9.1 训练内容

A0 保持以下模块冻结：

- visual tracker；
- recurrent compact memory；
- raw-memory query resampler；
- action-memory cross-attention 接口。

只训练：

- Pi0.5 action expert；
- action input/output projection；
- time MLP projection。

这正是原始 nominal absolute-EEF recipe，不混入其他数据源。

### 9.2 准备真机训练入口

从 `examples/shellgame/train_old_tracker_full_absolute_eef.py` 复制一个真机版本，例如：

```text
examples/shellgame/train_real_tracker_full_absolute_eef.py
```

至少替换：

```python
LEROBOT_ABSOLUTE_EEF7_ROOT = "<REAL_LEROBOT_ROOT>"
CONFIG_NAME = "pi0_shellgame_real_tracker_full_absolute_eef7"
```

如果真机规范化后的 episode 不是 155 帧，还必须同步修改：

- `LAST_EPISODE_FRAME`；
- phase-balanced row sampler；
- terminal temporal mask；
- fixed-prefix dataset 的历史边界。

推荐保持 155 帧，避免同时改变模型、数据和采样逻辑。

### 9.3 启动训练

若 M0 已证明仿真 tracker 可迁移，`<INIT_CHECKPOINT>` 可使用现有已验证 M6 checkpoint。若执行了真机 M1～M6，则使用真机 M6 checkpoint。

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
nohup uv run python examples/shellgame/train_real_tracker_full_absolute_eef.py \
  --exp-name real_absolute_eef_nominal_v1 \
  --init-checkpoint <INIT_CHECKPOINT> \
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
  > train_real_absolute_eef_nominal_v1.log 2>&1 &
```

训练开始后必须从日志确认：

```text
dataset=<REAL_LEROBOT_ROOT>
action_horizon=16
real_action_dim=7
fixed_history=60, total_frames=61
freeze=tracker+memory+memory-interface
init=<INIT_CHECKPOINT>
```

保留并分别评估 step 499、999、1499、1999。不要默认最后一步最好，也不要只按 validation loss 选 checkpoint。

### 9.4 phase-balanced sampling

原始 A0 对五个 action 阶段做 20%/20%/20%/20%/20% 平衡：

```text
selection : frame 59
approach  : frame 60..88
descend   : frame 89..108
grasp     : frame 109..118
lift      : frame 119..153
```

真机数据若沿用规范时间轴可直接复用。这样可以避免 selection 或长 lift 阶段按原始帧数压倒短暂但关键的 grasp 阶段。

## 10. E0：checkpoint 选择

每个候选 checkpoint 至少做四层评估。

### 10.1 数据和 loss 检查

- train/eval loss 是否有限且稳定；
- gripper loss 是否单独下降；
- temporal valid fraction 是否符合预期；
- predicted action 是否位于真机 train q01/q99 附近；
- absolute pose 是否连续，无瞬时大跳变。

### 10.2 memory 检查

在完全独立的 test episodes 上报告：

```text
initial cup accuracy
swap 1/2/3 pair accuracy
final cup accuracy
normal / shuffle / zero / wrong-episode memory action 对比
```

正确 memory 与错误 memory 的动作目标必须产生可解释差异，否则即使离线 action loss 很低也不能部署。

### 10.3 离线 action 指标

分别按 `selection / approach / descend / grasp / lift` 报告：

- xyz MAE；
- rotation geodesic error；
- gripper sign accuracy 和 close transition timing；
- 16 步 chunk 前 1/3/8 步误差；
- 预测轨迹超出 workspace 的比例。

### 10.4 影子推理

连接真实相机和机器人 state，但不下发 action：

1. 完整记录 60 帧 shell-game 历史；
2. 每个控制周期请求 action chunk；
3. 将预测 EEF 轨迹叠加到相机画面或 3D 可视化；
4. 检查目标杯、下降时机、闭爪时机、抬升方向；
5. 所有安全检查通过后才允许真机执行。

## 11. D0：真机部署

### 11.1 启动 policy server

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0 \
uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
  --checkpoint-dir <SELECTED_CHECKPOINT_DIR> \
  --port 8000 \
  --num-sampling-steps 4
```

若真机训练入口改变了 config name 或模型类，应复制出对应 real server，并用完全相同的 real training config 重建模型后加载 checkpoint；不能用形状碰巧一致但 transform/norm stats 不同的 server。

### 11.2 在线输入

episode 开始时缓存规范化后的 60 帧历史：

```text
fixed_history = [frame_0, ..., frame_59]
```

闭环每次推理：

```text
model_video = fixed_history + [latest_current_frame]
state       = latest measured world xyz + rot6d + gripper width
prompt      = "Grasp and lift the cup containing the ball."
```

不要使用滑动 61 帧窗口。历史一旦滑动，frame 0 和三段 swap 会被推出窗口，破坏 tracker 的固定时间语义。

### 11.3 action 执行

server 返回 `[16,7]` action chunk。执行端流程：

```text
反归一化（server policy 正常应已完成）
  ↓
检查有限值、时间戳和 watchdog
  ↓
absolute xyz workspace clamp
  ↓
单步 xyz / rotation / gripper 变化限幅
  ↓
碰撞、力矩、速度、奇异位形检查
  ↓
转换为真机控制器目标
  ↓
以 10 Hz 执行前 N 步，然后重规划
```

初始建议 `N=1` 做限速验证；确认闭环稳定后再测试 `N=4`，最后评估是否使用 V10 参考设置 `N=8`。训练 horizon=16 不代表必须一次执行完 16 步。

### 11.4 分级放行

按以下顺序逐级通过，每一级失败都停止，不直接跳到完整抓取：

1. policy server + 假 observation smoke test；
2. 真机连接但 action 不下发；
3. action 下发到仿真/数字孪生或控制器 dry-run；
4. 真机空载、低速、远离桌面；
5. 只允许在三个杯上方做 XY approach，禁止下降和闭爪；
6. 允许下降，但保持夹爪打开；
7. 允许闭爪但限制抬升高度；
8. 完整抓取和抬升；
9. 固定 test episode 协议的正式评测。

必须具备硬件急停、软件 watchdog、通信超时 hold、workspace 边界、最大笛卡尔步长、最大旋转步长、最大速度/加速度和碰撞/力矩阈值。policy server 断连时只能停止或保持安全 pose，不能继续消费旧 action chunk。

## 12. 最终评测与交付

正式结果至少分开报告：

```text
1. final cup selection accuracy
2. approach success
3. grasp success
4. lift success
5. left / middle / right 分槽成功率
6. 每个 episode 的 seed/配置/视频/机器人 trace
7. 推理延迟、控制周期超时率、急停次数
```

每次训练交付应保存：

```text
robot_contract.json
采集代码版本和相机/机器人标定文件
原始 episode manifest
数据审计报告
train/val/test episode 列表
LeRobot meta/info.json 和 norm_stats.json
完整训练命令与日志
初始化 checkpoint 和最终候选 checkpoints
离线评估结果
影子推理记录
真机 result.json、trace 和视频
```

## 13. 最短执行清单

```text
[ ] 冻结真机 absolute-EEF、坐标系、姿态、夹爪和时间戳契约
[ ] 采集成功示范，同时保存双相机、measured state、commanded action 和标签
[ ] 按 phase 重采样为 155 帧，确保 swaps 位于 20..49
[ ] 根据真机日志语义选择 obs[i]->action[i] 或 obs[i]->action[i+1]
[ ] 转为 LeRobot v2.1，action=[16,7]，重新计算真机 norm_stats
[ ] episode-level 划分并完成回放/可视化审计
[ ] 先测现有 memory 的真机迁移；不通过再做 M1～M6
[ ] A0 只用真机 nominal 数据，冻结 tracker/memory/interface
[ ] 比较 499/999/1499/1999，不按最后一步或最低 loss 自动选模型
[ ] normal/shuffle/zero/wrong-history 消融通过
[ ] 影子推理和分级安全放行通过
[ ] 固定 60 帧历史，动态更新第 61 帧，低 N 起步闭环部署
```

## 14. 当前仓库中的参考实现

- 原始 nominal EEF 训练：`examples/shellgame/train_old_tracker_full_absolute_eef.py`
- 固定历史 dataset：`examples/shellgame/fixed_prefix_current_video_dataset.py`
- 固定历史在线输入：`examples/shellgame/main_absolute_eef_fixed_history.py`
- policy server：`examples/shellgame/serve_old_tracker_full_absolute_eef.py`
- M1：`examples/shellgame/train_frame0_initial_cup_probe.py`
- M2：`examples/shellgame/train_three_swap_pair_fixed_grid_probe.py`
- M3：`examples/shellgame/train_three_swap_oracle_single_history_read_adapter_probe.py`
- M4：`examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py`
- M5：`examples/shellgame/train_three_swap_fully_visual_joint_action_probe.py`
- M6：`examples/shellgame/train_three_swap_query_crossattn_pi_joint_action_probe.py`
- 仿真 nominal 转换器（仅用于参考 schema 和对齐逻辑）：`../robosuite/robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py`
- V10 阶段总结：`docs/shellgame_mem_absolute_eef_stage_summary_260821.md`

最后强调三条：

1. 真机训练首先要保证 action 是实际下发 command，而不是把 measured pose 当作 action；
2. 固定历史和 swap 时间窗是当前 V10 tracker 的结构契约，不是可随意改变的数据增强；
3. memory 指标和抓取指标必须分开验证，只有两者都通过才部署。
