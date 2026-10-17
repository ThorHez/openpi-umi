# 真机 ShellGame 306 条数据：仅 Stage-2、与真机推理一致的训练 Recipe

## 固定契约

- 初始化 MEM：`/data2/hzl_workspace_for_pi_mem/4999/params`
- 固定视觉历史：episode frame `0..240`，共 241 帧
- 当前视觉：action 时刻的 wrist frame，总模型输入 242 帧
- state：`T_ep0^-1 @ T_link6_measured[t]`，表示为 `xyz(3)+rot6d(6)+gripper(1)`
- action：16 个 future target 均为 `T_link6_measured[t]^-1 @ T_link6_target[t+h+1]`，每步 10D
- Pi0.5 内部 action pad 到 32D，loss 只作用于前 10D，gripper index 为 9
- Stage-2 冻结视觉 MEM tracker；训练 query resampler、memory cross-attention、Pi0.5 action expert 和 action/time projections

这与 `/data2/hzl_workspace_for_pi_mem/umi-arx-kian/scripts/eval_arx5_pi_hzl.py` 完全一致。该脚本把模型输出按同一个当前 `link6` anchor 解码：

```text
T_link6_target_world = T_link6_current_world @ T_model_current_relative
T_eef_target_world = link6_to_eef(T_link6_target_world, eef_offset=0.145)
```

不要把 action 改成 episode-first，也不要让 16 个 waypoint 使用不同 anchor。episode-first 只用于输入 state。

其他精确 I/O：

- 输入图像 key：`left_wrist_0_rgb_0` ... `left_wrist_0_rgb_241`，共 242 张 RGB `uint8 (224,224,3)`；前 241 张固定，最后一张为当前 wrist；
- 输入 state key：`robot0_eef_pos (3)`、`robot0_eef_rot_axis_angle (6D)`、`robot0_gripper_width (1)`；
- 训练 prompt 与推理默认 prompt 相同：`The shell game has ended. Grasp and lift the cup containing the ball.`；
- 输出：`actions (16,10)`，布局为 `xyz(3)+row-rot6d(6)+Direct gripper width(1)`；
- 推理端先把 ZMQ eef 减去 TCP-Z `0.145m` 得到 Direct link6，再构造 state/解码 action；夹爪 observation 从 ZMQ `0.085m` 量程映射到训练的 Direct `0.11m` 量程，action 下发前做逆映射。

## 1. 数据审计

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
PYTHONPATH=src .venv/bin/python scripts/mem/convert_real_shellgame_stage2_epfirst.py \
  --audit-only \
  --input ../replay_buffer_merged_306_degap.zarr.zip \
  --labels ../labels_merged_306_degap.jsonl
```

当前审计结果保存在：

```text
artifacts/shellgame_real_306_stage2_conversion_audit.json
```

关键结果：

- 306 episodes / 118,807 frames；
- action suffix 中 272 帧（0.604%）raw command 无效或相对 measured pose 偏差超过 5cm，转换时回退到对应 measured EEF；
- 按真机推理 decoder 做 SE(3) round-trip，最大位置误差 `1.67e-16 m`；
- initial cup 分布 `107/102/97`，final cup 分布 `95/115/96`；
- episode 251..305 的 55 条 label `n_frames` 比 Zarr episode boundary 多 1..23 帧。固定事件前缀仍为 `0..240`，action terminal mask 必须使用 Zarr 的实际 episode length。

## 2. LeRobot 数据

正式数据已经完成转换：306 episodes / 118,807 frames，约 9.2GiB：

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10
```

目录包含 `meta/info.json`、`meta/episodes.jsonl`、`norm_stats.json` 和 `conversion_audit.json`。如需从原始 zip 重新生成：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
PYTHONPATH=src .venv/bin/python scripts/mem/convert_real_shellgame_stage2_epfirst.py \
  --input ../replay_buffer_merged_306_degap.zarr.zip \
  --labels ../labels_merged_306_degap.jsonl \
  --output data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10
```

转换器同时生成真实数据自己的 `norm_stats.json`，不会复用仿真或 MEM checkpoint 的 action stats。

## 3. Stage-2 训练

配置名：

```text
pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2
```

默认训练参数：4 GPUs、global batch 4、21,000 steps、warmup 500、peak LR `3e-5`、cosine decay 与总步数同步、每 250 steps 验证 64 batches、每 500 steps 保存。

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
GPU_IDS=0,1,2,3 \
EXP_NAME=real306_currentrel_full80_interface_pi05_seed42_v1 \
nohup scripts/run_shellgame_real_stage2_epfirst.sh \
  > launch_real_stage2_epfirst.log 2>&1 &
```

launcher 默认把 Hugging Face/Arrow 缓存放在
`openpi-umi/.cache/shellgame_real_stage2/huggingface`。不要改到 `/tmp` 或
`/dev/shm`；首次索引 118,807 个带图像的 rows 会超过这些小分区容量。

训练日志必须看到以下梯度非零：

```text
grad/semantic_query_resampler_l2
grad/semantic_action_cross_attn_l2
grad/action_expert_l2
grad/action_projection_l2
```

若前两个始终为零，立即停止；这意味着又复现了失败训练中的 frozen interface 问题。

## 4. checkpoint 选择

不要默认部署最后一个 checkpoint。至少扫描 step 500 起每个保存点，并使用反归一化物理指标：

- XYZ RMSE 必须低于 zero-action baseline；
- early-grasp XYZ cosine 必须大于 0，建议大于 0.5；
- step-0 方向正确；
- normal memory 明显优于 zero/shuffled memory；
- 三个 final-cup 类别均不能退化到单一动作偏置。

在这些条件通过前，只做 observe-only / shadow inference，不下发正式抓取动作。
