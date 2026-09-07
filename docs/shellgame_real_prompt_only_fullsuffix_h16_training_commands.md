# ShellGame prompt-only full-suffix H16 模型训练命令

本文档复现当前真机 H16 实验使用的动作模型。当前已验证模型为：

```text
config: pi0_mem_shellgame_real_m6_prompt_only_full_suffix_mixed
run:    freshmem1800_prompt_only800_cachedsingleframe_fullsuffix_anchor30_b72_s20k_save5k_seed42_v4
best:   step 14999
```

对应 checkpoint：

```text
checkpoints/pi0_mem_shellgame_real_m6_prompt_only_full_suffix_mixed/
  freshmem1800_prompt_only800_cachedsingleframe_fullsuffix_anchor30_b72_s20k_save5k_seed42_v4_fullsuffix_best_guarded/14999
```

## 1. 训练契约

- action horizon 固定为 16，训练监督为当前帧之后的 16 个 EEF10 target。
- 16 个 target 都相对于同一个当前帧 EEF anchor，不是逐步累积 delta。
- action 分支只读取当前单帧图像和方位 prompt，不向 action token 注入 MEM feature。
- 训练 prompt 的左/中/右来自数据标签；部署时由冻结 MEM 分类结果生成 prompt。
- action 数据仅使用 old306 和 cup0903，采样比例为 25% / 75%。
- 每个来源内部按最终杯 left/middle/right 均衡采样。
- 30% batch 固定重点采样 frame241，70% batch 采样所有 `frame_index >= 241` 的 suffix。
- direction loss 为 0，只使用 flow loss；方向指标仅作为训练守护和 checkpoint 筛选条件。
- 训练 20000 step，global batch size 72，8 张 GPU，seed 42。
- 每 5000 step 保存并验证一次，对应目录为 4999、9999、14999、19999。
- 只有通过方向守护的 checkpoint 才参加固定分层 gradual suffix 验证子集的 XYZ RMSE
  比较，并单独保留最优 checkpoint。

`--steps-per-inference 16` 是真机 eval 的执行参数，不是训练参数。模型在 H4 和 H16
下都会输出 16 步；区别只是 eval 执行前 4 步还是完整 16 步。

## 2. 需要准备的数据和初始化 checkpoint

在工程根目录执行：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

test -x .venv/bin/python
test -d data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10
test -d data/shellgame_real_cup0903_state_epfirst_action_currentrel_eef10
test -f /data2/hzl_workspace_for_pi_mem/labels_merged_306_degap.jsonl
test -f /data2/hzl_workspace_for_pi_mem/cup_0903/labels.jsonl

test -d checkpoints/pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800/params
test -f checkpoints/pi0_mem_shellgame_real_fresh_memory_mild_all/freshmem_train383_val44_mildaug_officialbase_b32_seed42_split20260904_v1/training_manifest.json

test -f evaluation/shellgame_real/freshmem1800_prompt_only_vs_memory_seed42_v2/memory_old306.json
test -f evaluation/shellgame_real/freshmem1800_prompt_only_vs_memory_seed42_v2/memory_cup0903.json

nvidia-smi
df -h /data2/hzl_workspace_for_pi_mem
```

初始化模型必须是下面这个 prompt-only frame241 checkpoint，而不是最终的 step14999：

```text
checkpoints/pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/
  freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800
```

## 3. 推荐命令：训练、定期验证并自动选择最佳 checkpoint

下面的 wrapper 会依次完成：

1. 对初始化 checkpoint 做方向基线测试；
2. 用全部 8 张 GPU 训练到 5000、10000、15000、20000 step；
3. 每个阶段运行 old306 和 cup0903 方向验证；
4. 方向没有明显退化时运行 gradual suffix 验证；
5. 按 old306/cup0903 各类别 2 个 episode 的固定分层验证子集加权 XYZ RMSE 保存
   最佳 checkpoint；每个 episode 测试 5 个 suffix frame；
6. 连续两次方向守护失败时提前停止；
7. 最佳 checkpoint 选出后运行完整方向测试和最终 gradual suffix 测试。

请把 `--run-name` 改成同事自己的唯一实验名。不要复用已经存在的 v4 名称。

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/mem/run_shellgame_real_m6_prompt_only_full_suffix_guarded.py \
  --run-name colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1 \
  --checkpoint checkpoints/pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800 \
  --steps 20000 \
  --interval 5000 \
  --batch-size 72 \
  --anchor-fraction 0.30 \
  --direction-drop-tolerance 0.10 \
  --failure-patience 2 \
  --port 18047
```

wrapper 内部固定训练使用 GPU 0–7、验证使用 GPU 0，并设置
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`。因此不需要在外层再次配置
`CUDA_VISIBLE_DEVICES`。

需要断开终端运行时：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

setsid -f .venv/bin/python scripts/mem/run_shellgame_real_m6_prompt_only_full_suffix_guarded.py \
  --run-name colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1 \
  --checkpoint checkpoints/pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800 \
  --steps 20000 \
  --interval 5000 \
  --batch-size 72 \
  --anchor-fraction 0.30 \
  --direction-drop-tolerance 0.10 \
  --failure-patience 2 \
  --port 18047 \
  >> colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1.launcher.log 2>&1
```

同一台机器同时运行多个实验时，每个实验必须使用不同的 `--run-name` 和 `--port`，
并自行分配不冲突的 GPU。当前 wrapper 固定占用全部 8 张卡，不适合与另一个 8 卡训练并行。

## 4. 仅训练、不运行 checkpoint 守护

不推荐这种方式，因为它不会运行方向守护，也不会按固定验证子集选择最优
checkpoint。只在调试 recipe 时使用：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONUNBUFFERED=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
.venv/bin/python scripts/mem/train_shellgame_real_m6_prompt_only_full_suffix.py \
  --exp-name colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1_fullsuffix \
  --checkpoint checkpoints/pi0_mem_shellgame_real_m6_prompt_action_ablation_mixed/freshmem1800_prompt_only_vs_memory_seed42_v2_prompt_only/800 \
  --steps 20000 \
  --warmup-steps 150 \
  --peak-lr 1e-5 \
  --batch-size 72 \
  --eval-batch-size 72 \
  --num-workers 16 \
  --fsdp-devices 8 \
  --eval-interval 250 \
  --eval-batches 2 \
  --anchor-fraction 0.30 \
  --save-interval 5000
```

首次启动不要加 `--resume` 或 `--overwrite`。只有确认需要清空同名实验时才能使用
`--overwrite`；从已有 checkpoint 接着训练则增加 `--resume`。

## 5. 输出位置和进度检查

推荐 wrapper 的输出：

```text
evaluation/shellgame_real/<run-name>/pipeline_state.json
evaluation/shellgame_real/<run-name>/train.log
evaluation/shellgame_real/<run-name>/direction_eval/
evaluation/shellgame_real/<run-name>/gradual_eval/

checkpoints/pi0_mem_shellgame_real_m6_prompt_only_full_suffix_mixed/<run-name>_fullsuffix/
checkpoints/pi0_mem_shellgame_real_m6_prompt_only_full_suffix_mixed/<run-name>_fullsuffix_best_guarded/
```

查看状态：

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi

python3 -m json.tool evaluation/shellgame_real/colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1/pipeline_state.json

grep -aE 'Step [0-9]+|\[eval\]|direction|gradual|ABORTED|Traceback' \
  evaluation/shellgame_real/colleague_prompt_only_fullsuffix_anchor30_b72_s20k_seed42_v1/train.log | tail -n 80
```

当前 v4 实验根据上述固定分层验证子集选择了 step14999，而不是最后的 step19999。
训练结束后还会对选中 checkpoint 运行完整方向验证，但不会用这项最终指标重新选择
checkpoint。复现实验必须读取 `pipeline_state.json` 中的 `best_checkpoint`，不能默认
使用数值最大的 checkpoint。

## 6. 与真机 H16 的关系

训练输出始终是 16-step action chunk。真机验证时需要在 `umi-arx-kian` 中设置：

```bash
--steps-per-inference 16
```

这会完整执行 index 0–15。它不会改变 checkpoint，也不需要重新训练；如果使用
`--steps-per-inference 4`，模型仍输出 16 步，但 eval 只提交 index 0–3。
