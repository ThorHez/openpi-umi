# ShellGame 真实场景 M5 + M6 完整训练命令

本文档用于将当前真实 ShellGame 的 M5、M6 训练完整交给其他人执行。
推荐顺序如下：

1. 可选：训练并测试 M5-oracle，验证方位标签能够驱动左/中/右动作；
2. 必需：训练并测试 M5-memory，验证冻结 MEM 的分类能够驱动方向动作；
3. 必需：用 M5-memory step 999 初始化 M6，训练带方向 prompt 的端到端动作模型；
4. M6 正常完成后立即运行 held-out 离线测试。

M5-oracle 只是上界诊断，不会传给 M6。M6 使用的是
`real306_m5_memory_seed42_v1/999/params`。

## 1. 数据和模型契约

- 工程：`/data2/hzl_workspace_for_pi_mem/openpi-umi`
- 数据：`data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10`
- 标签：`/data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl`
- M5 初始 MEM checkpoint：`/data2/hzl_workspace_for_pi_mem/4999/params`
- M5 历史帧：0 到 240，共 241 帧
- M5 当前帧：241
- M5 动作目标：command 242 到 257，共 16 步
- state：相对于 episode 第一帧的 EEF10
- action：相对于当前帧 241 的 delta EEF10
- M6 历史/当前帧、state 和 action 契约与 M5 一致
- M6 训练 prompt：标注的最终方位 left/middle/right
- M6 部署 prompt：冻结 MEM 的最终方位预测
- 数据划分：seed 42，275 个训练 episode，31 个验证 episode

## 2. 启动前检查

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

test -x .venv/bin/python
test -f /data2/hzl_workspace_for_pi_mem/4999/params/_METADATA
test -d data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10
test -f /data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl
nvidia-smi
df -h /data2/hzl_workspace_for_pi_mem
```

M5-memory 和 M6 的离线测试还会读取已经生成的 MEM 分类结果：

```bash
test -f evaluation/shellgame_real/real306_currentrel_full80_interface_pi05_seed42_v1_step20999/memory_classifier_validation.json
```

如果 `.venv` 不存在，需要先安装 `uv`，然后在工程根目录执行：

```bash
uv sync --frozen
```

## 3. 可选诊断：M5-oracle 训练

M5-oracle 使用 GT 最终杯子方位作为语义输入，只训练
`HistorySemanticJointActionReadout`。它用于确认 action 标签和坐标契约本身可以学习，
不是 M6 的初始化 checkpoint。

使用 GPU 0、1、2、3：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/mem/train_shellgame_real_m5_action_probe.py \
  --semantic-source oracle \
  --exp-name real306_m5_oracle_seed42_v1 \
  --checkpoint /data2/hzl_workspace_for_pi_mem/4999/params \
  --steps 1000 \
  --warmup-steps 30 \
  --peak-lr 3e-4 \
  --batch-size 4 \
  --fsdp-devices 4 \
  --num-workers 8 \
  --eval-interval 50 \
  --eval-batches 8 \
  --save-interval 100 \
  2>&1 | tee train_shellgame_real_m5_oracle_seed42_v1.log
```

M5 只保留最新 checkpoint，最终输出为：

```text
checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_oracle_seed42_v1/999
```

### 测试 M5-oracle

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/mem/eval_shellgame_real_m5_oracle_action_probe.py \
  --checkpoint checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_oracle_seed42_v1/999 \
  --output evaluation/shellgame_real/real306_m5_oracle_seed42_v1_step999/m5_oracle_action_validation.json \
  2>&1 | tee eval_shellgame_real_m5_oracle_seed42_v1.log
```

重点查看 `normal_oracle_accuracy` 和
`counterfactual_forced_class_accuracy`。只有 oracle prompt 能稳定改变动作方向，才值得继续
M5-memory 和 M6。

## 4. 必需：M5-memory 训练并自动测试

M5-memory 使用冻结 MEM 的预测概率作为语义输入，同样只训练确定性的 action readout。
其 step 999 checkpoint 是 M6 的初始化模型。

推荐直接运行现有训练后自动测试脚本：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
bash scripts/run_shellgame_real_m5_memory_and_eval.sh
```

断开终端运行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
setsid -f bash scripts/run_shellgame_real_m5_memory_and_eval.sh \
  >> run_shellgame_real_m5_memory_and_eval.log 2>&1
```

该脚本等价于下面的训练命令：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/mem/train_shellgame_real_m5_action_probe.py \
  --semantic-source memory \
  --exp-name real306_m5_memory_seed42_v1 \
  --checkpoint /data2/hzl_workspace_for_pi_mem/4999/params \
  --steps 1000 \
  --warmup-steps 30 \
  --peak-lr 3e-4 \
  --batch-size 4 \
  --fsdp-devices 4 \
  --num-workers 8 \
  --eval-interval 50 \
  --eval-batches 8 \
  --save-interval 100 \
  2>&1 | tee train_shellgame_real_m5_memory_seed42_v1.log
```

训练成功的必要输出：

```text
checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_memory_seed42_v1/999/params/_METADATA
```

自动测试结果：

```text
evaluation/shellgame_real/real306_m5_memory_seed42_v1_step999/m5_memory_action_validation.json
```

重点查看 `action_follows_memory_accuracy`。M5-memory 是诊断模型，不是真机部署用的
Pi flow policy。

## 5. 必需：M6 训练并自动测试

只有 M5-memory 的 step 999 完整保存后才启动 M6：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
test -f checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_memory_seed42_v1/999/params/_METADATA
```

推荐运行训练后自动测试脚本。它使用 GPU 0、1、2、3，支持从已有最新 checkpoint
续训；M6 每 5000 step 保存一次，并且只保留最新一份。

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
bash scripts/run_shellgame_real_m6_direction_prompt_and_eval.sh
```

断开终端运行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
setsid -f bash scripts/run_shellgame_real_m6_direction_prompt_and_eval.sh \
  >> run_shellgame_real_m6_direction_prompt_and_eval.log 2>&1
```

M6 的实际训练参数为：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
.venv/bin/python scripts/mem/train_shellgame_real_m6_direction_prompt.py \
  --exp-name real306_m6_direction_prompt_seed42_v1 \
  --checkpoint checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_memory_seed42_v1/999/params \
  --steps 21000 \
  --warmup-steps 500 \
  --peak-lr 3e-5 \
  --batch-size 4 \
  --fsdp-devices 4 \
  --num-workers 8 \
  --eval-interval 250 \
  --eval-batches 64 \
  --save-interval 5000 \
  --resume \
  2>&1 | tee -a train_shellgame_real_m6_direction_prompt_seed42_v1.log
```

M6 checkpoint 保存节点为 `5000、10000、15000、20000、20999`。由于只保留最新
一份，最终输出为：

```text
checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt/real306_m6_direction_prompt_seed42_v1/20999
```

自动测试结果为：

```text
evaluation/shellgame_real/real306_m6_direction_prompt_seed42_v1_step20999/m6_direction_prompt_validation.json
```

重点查看：

- `deployment_action_follows_memory_prompt_accuracy`
- `deployment_action_ground_truth_accuracy`
- `oracle_prompt_action_ground_truth_accuracy`
- `counterfactual_prompt_following_accuracy`
- `mean_deployment_xyz_rmse_mm`

## 6. M5-memory → M6 连续自动执行

如果不运行可选的 M5-oracle，可以用下面的命令自动依次完成：

1. M5-memory 训练；
2. M5-memory 离线测试；
3. M6 训练；
4. M6 离线测试。

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

setsid -f bash -c '
set -euo pipefail
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
bash scripts/run_shellgame_real_m5_memory_and_eval.sh
bash scripts/run_shellgame_real_m6_direction_prompt_and_eval.sh
' >> run_shellgame_real_m5_m6_pipeline.log 2>&1
```

任一阶段失败时，`set -euo pipefail` 会阻止后续阶段启动。不要同时手工启动相同的
M5 或 M6 实验。

## 7. 查看进度

查看 M5/M6 进程：

```bash
pgrep -af 'run_shellgame_real_m5|train_shellgame_real_m5|run_shellgame_real_m6|train_shellgame_real_m6'
```

查看 M5 最新指标：

```bash
grep -aEo 'Step [0-9]+: action_loss=[0-9.eE+-]+|val/action_loss=[0-9.eE+-]+|val/loss=[0-9.eE+-]+' \
  train_shellgame_real_m5_memory_seed42_v1.log | tail -n 30
```

查看 M6 最新指标：

```bash
grep -aEo 'Step [0-9]+: action_loss=[0-9.eE+-]+|val/action_loss=[0-9.eE+-]+|val/loss=[0-9.eE+-]+' \
  train_shellgame_real_m6_direction_prompt_seed42_v1.log | tail -n 30
```

查看 M6 已保存 checkpoint：

```bash
find checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt/real306_m6_direction_prompt_seed42_v1 \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
```

## 8. 重要注意事项

- M5-oracle 和 M5-memory 都从 `/data2/hzl_workspace_for_pi_mem/4999/params` 初始化，
  不是从真实 Stage2 step 20999 初始化。
- M6 只从 M5-memory step 999 初始化，不使用 M5-oracle checkpoint。
- M5 只验证“语义方位能否控制动作方向”；M6 才训练可供端到端推理使用的
  Pi action policy。
- M6 训练时方向 prompt 来自 GT 标签；部署时方向 prompt 必须来自冻结 MEM 预测。
- 不要对需要保留或续训的实验添加 `--overwrite`。
- M5/M6 的 checkpoint 都包含完整冻结 backbone；当前 M5 单份约 5.2 GiB，M6
  单份约 9.1 GiB。异步保存时会短暂同时存在旧 checkpoint 和临时新 checkpoint，
  建议至少预留 25 GiB。
