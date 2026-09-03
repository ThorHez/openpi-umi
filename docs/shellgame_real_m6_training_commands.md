# ShellGame 真实场景 M6 训练命令说明

> 完整的 M5 + M6 训练交接请使用
> `docs/shellgame_real_m5_m6_training_commands.md`；本文档仅保留 M6 单阶段说明。

本文档用于把当前 M6 训练交给其他人直接执行。当前 recipe 使用 4 张 GPU，
每 5000 step 保存一次 checkpoint，并在训练正常完成后立即运行离线测试。

## 1. 训练配置与数据契约

- 工程目录：`/data2/hzl_workspace_for_pi_mem/openpi-umi`
- 训练数据：`data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10`
- 初始化模型：M5 `real306_m5_memory_seed42_v1/999/params`
- 历史图像：episode 的第 0 到 240 帧
- 当前帧：从第 241 帧开始采样
- state：相对于 episode 第一帧的 EEF10
- action：16 步、相对于当前帧的 delta EEF10
- 训练 prompt：使用标注的最终方位（left/middle/right）
- 部署 prompt：使用冻结 MEM 给出的方位分类
- 训练步数：21000，最终 checkpoint 为 step 20999
- checkpoint 间隔：5000；保存节点为 5000、10000、15000、20000、20999
- checkpoint 保留策略：只保留最新一份；保存切换期间需要能容纳新旧两份，建议至少预留 25 GiB

## 2. 启动前检查

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

test -x .venv/bin/python
test -f checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_memory_seed42_v1/999/params/_METADATA
nvidia-smi
df -h /data2/hzl_workspace_for_pi_mem
```

如果 `.venv` 不存在，需要先在工程根目录安装 `uv`，然后执行：

```bash
uv sync --frozen
```

## 3. 推荐：训练完成后自动测试

一键脚本固定使用 GPU 0、1、2、3。它既支持首次启动，也支持在实验目录已有
checkpoint 时自动从最新 checkpoint 续训。训练正常结束后，会自动使用最终
step 20999 执行 31 个 held-out episode 的方向 prompt 测试。

前台运行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
bash scripts/run_shellgame_real_m6_direction_prompt_and_eval.sh
```

推荐的断开终端运行方式：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
setsid -f bash scripts/run_shellgame_real_m6_direction_prompt_and_eval.sh \
  >> run_shellgame_real_m6_direction_prompt_and_eval.log 2>&1
```

不要同时启动两次该脚本，否则两个进程会写入同一个实验目录。

## 4. 仅启动训练

首次训练时使用下面的命令。若要创建另一个实验，必须把 `--exp-name` 改成新的
唯一名称。

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
  2>&1 | tee train_shellgame_real_m6_direction_prompt_seed42_v1.log
```

这条命令只训练，不会自动运行最终方向测试。

## 5. 从最新 checkpoint 续训

续训命令与首次训练相同，但必须保持相同的 `--exp-name`，并增加 `--resume`。
日志使用 `tee -a` 追加，避免覆盖之前的训练记录。

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

不要对需要续训的实验使用 `--overwrite`；该参数会清空已有实验目录。

## 6. 单独运行最终测试

只有 step 20999 checkpoint 完整保存后才能运行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
.venv/bin/python scripts/mem/eval_shellgame_real_m6_direction_prompt.py \
  --checkpoint checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt/real306_m6_direction_prompt_seed42_v1/20999 \
  --output evaluation/shellgame_real/real306_m6_direction_prompt_seed42_v1_step20999/m6_direction_prompt_validation.json \
  --samples-per-prompt 2 \
  2>&1 | tee eval_shellgame_real_m6_direction_prompt_seed42_v1.log
```

## 7. 查看运行状态

查看训练和自动测试 wrapper 是否仍在运行：

```bash
pgrep -af 'run_shellgame_real_m6_direction_prompt_and_eval|train_shellgame_real_m6_direction_prompt'
```

查看最新训练 step 和验证 loss：

```bash
grep -aEo 'Step [0-9]+: action_loss=[0-9.eE+-]+|val/action_loss=[0-9.eE+-]+|val/loss=[0-9.eE+-]+' \
  train_shellgame_real_m6_direction_prompt_seed42_v1.log | tail -n 30
```

查看已经完整保存的 checkpoint：

```bash
find checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt/real306_m6_direction_prompt_seed42_v1 \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
```

## 8. 输出位置

- 训练日志：`train_shellgame_real_m6_direction_prompt_seed42_v1.log`
- wrapper 日志：`run_shellgame_real_m6_direction_prompt_and_eval.log`
- 测试日志：`eval_shellgame_real_m6_direction_prompt_seed42_v1.log`
- checkpoint：`checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt/real306_m6_direction_prompt_seed42_v1/`
- 最终测试结果：`evaluation/shellgame_real/real306_m6_direction_prompt_seed42_v1_step20999/m6_direction_prompt_validation.json`
