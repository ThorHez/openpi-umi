# ShellGame cup_0903 混合 M5/M6 训练

## 数据与接口

- 旧域：`data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10`
- 新域：`data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10`
- 固定历史：episode frame `0..240`，当前动作帧从 `241` 开始。
- state：相对 episode 第一帧的 EEF pose10 + gripper。
- action：未来 16 步、相对当前帧的 EEF pose10 + gripper；与真机 eval 解码接口一致。
- 新数据 episode id 在模型输入中增加 306，避免与旧数据的 `0..305` 冲突。
- 新域固定划分为 70 train / 15 validation / 15 final test；final test 不进入训练或验证。
- 训练源概率为旧域 25%、新域 75%，每个域和 split 内部再做 left/middle/right 等量下采样。

转换审计确认 cup_0903 共 100 个 episode、40,688 帧；frame241 后原始 command fallback 为 0，EEF 合约往返误差约 `1e-16`。标签中的 `n_frames` 与 Zarr `episode_ends` 不一致，因此转换始终以 Zarr 为准。

## 初始化

- 适配后的 MEM：
  `checkpoints/pi0_mem_shellgame_real_relation_adapt_new75_old25/cup0903_new75_old25_relation_only_lr1e5_b32_seed42_v1/500/params`
- 已验证的旧域 H16 动作策略：
  `checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_stage1/real306_m6_direction_stage1_frame241_dirloss010_b32_seed42_v1_best_direction/1199/params`

M5 从适配后的 MEM 初始化，只重新初始化并训练确定性 action probe。M6 保留旧 H16 checkpoint 的动作专家和 memory/action interface，仅从适配后的 MEM checkpoint 覆盖 `swap_relation_classifier`，随后继续训练；不会丢掉已有动作能力。

## 一键无人值守运行

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
tmux new-session -d -s shellgame_cup0903_m5_m6 \
  '.venv/bin/python -u scripts/mem/run_shellgame_real_m5_m6_mixed_cup0903.py \
    --run-name cup0903_mixed25_75_h16_seed42_v1 \
    --batch-size 32 \
    --m5-steps 1000 \
    --m6-stage1-max-steps 2000 \
    --m6-stage1-interval 250 \
    --m6-full-steps 10000'
```

流水线顺序如下：

1. M5 oracle frame241 训练，并在旧域/新域分别测试。
2. M5 memory frame241 训练，并测试动作是否跟随 MEM 分类。
3. M6 frame241 使用 `flow loss + 0.1 * direction loss`；每 250 step 在新域强制 left/middle/right，按方向跟随率选择 checkpoint。无改善或方向无效会提前停止。
4. 通过方向门槛后，M6 在完整 `frame>=241` 渐进动作段训练；普通 checkpoint 间隔为 5000 step。
5. 训练结束立即在两个域运行强制方向测试，并启动 cached policy 完整动作段测试。cached 测试用 `--prompt-from-memory`，方向 prompt 来自 MEM 预测而不是标签。

状态文件：

```text
evaluation/shellgame_real/cup0903_mixed25_75_h16_seed42_v1/pipeline_state.json
```

主要日志：

```text
evaluation/shellgame_real/cup0903_mixed25_75_h16_seed42_v1/train_m5_oracle.log
evaluation/shellgame_real/cup0903_mixed25_75_h16_seed42_v1/train_m5_memory.log
evaluation/shellgame_real/cup0903_mixed25_75_h16_seed42_v1/train_m6_stage1.log
evaluation/shellgame_real/cup0903_mixed25_75_h16_seed42_v1/train_m6_full.log
```

## 分阶段命令

M5：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python scripts/mem/train_shellgame_real_m5_action_probe_mixed_cup0903.py \
  --semantic-source memory \
  --exp-name cup0903_mixed_m5_memory_b32_seed42_v1 \
  --steps 1000 --batch-size 32 --eval-batch-size 32 --fsdp-devices 8 \
  --save-interval 5000
```

M6 frame241：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python scripts/mem/train_shellgame_real_m6_direction_stage1_mixed_cup0903.py \
  --exp-name cup0903_mixed_m6_stage1_dirloss010_b32_seed42_v1 \
  --steps 2000 --batch-size 32 --eval-batch-size 32 --fsdp-devices 8 \
  --direction-loss-weight 0.1 --eval-interval 50 --save-interval 5000
```

M6 完整渐进动作段（checkpoint 必须替换成上一阶段按方向指标选出的路径）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python scripts/mem/train_shellgame_real_m6_mixed_cup0903.py \
  --exp-name cup0903_mixed_m6_full_b32_seed42_v1 \
  --checkpoint /absolute/path/to/selected_stage1_checkpoint \
  --steps 10000 --batch-size 32 --eval-batch-size 32 --fsdp-devices 8 \
  --save-interval 5000
```

## 方向保持版完整后缀训练

普通完整后缀 flow-only 训练会削弱 left/middle/right prompt 的动作响应。方向保持版同时使用：

- 完整 `frame>=241` 数据学习渐进靠近、下降和抓取；
- `frame241..245` 平衡锚点计算方向 loss，避免对后续下降帧施加错误的横向约束；
- 每 1000 step 在旧域和新域全部 validation episode 上各采样两次强制 left/middle/right；
- 按全验证集的最弱域、最弱方向和 MEM prompt 跟随率选择 checkpoint；
- 新域方向门槛 80%，旧域方向门槛 70%，连续无改善时提前停止；
- 临时 checkpoint 自动裁剪，只保留当前与最佳模型。

启动命令（`--checkpoint` 应指向通过全验证比较后选出的 M6 Stage1 checkpoint）：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u scripts/mem/run_shellgame_real_m6_direction_full_guarded_cup0903.py \
  --run-name cup0903_mixed25_75_h16_seed42_directionfull \
  --checkpoint /absolute/path/to/selected_stage1_checkpoint \
  --steps 10000 --interval 1000 --batch-size 32 \
  --direction-loss-weight 0.1 --anchor-fraction 0.5 \
  --new-direction-floor 0.80 --old-direction-floor 0.70 --patience 3 \
  --port 18047 \
  > run_cup0903_mixed25_75_h16_seed42_directionfull.log 2>&1 < /dev/null &
```
