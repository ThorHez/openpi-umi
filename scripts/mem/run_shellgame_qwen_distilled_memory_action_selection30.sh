#!/usr/bin/env bash
set -euo pipefail

cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/data2/hzl_workspace_for_pi_mem/.cache/huggingface
export HF_DATASETS_CACHE=/data2/hzl_workspace_for_pi_mem/.cache/huggingface/datasets

exec .venv/bin/python \
  scripts/mem/train_shellgame_qwen_distilled_memory_action_selection30.py \
  --exp-name direct_visual_mem_step999_v10_selection30_mix500_6gpu_260826 \
  --steps 500 \
  --batch-size 12 \
  --fsdp-devices 6 \
  --num-workers 8 \
  --overwrite
