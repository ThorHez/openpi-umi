#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
EXP_NAME=real306_m5_memory_seed42_v1
TRAIN_LOG=${OPENPI_ROOT}/train_shellgame_real_m5_memory_seed42_v1.log
EVAL_LOG=${OPENPI_ROOT}/eval_shellgame_real_m5_memory_seed42_v1.log

cd "${OPENPI_ROOT}"

env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/mem/train_shellgame_real_m5_action_probe.py \
  --semantic-source memory \
  --exp-name "${EXP_NAME}" \
  --steps 1000 \
  --batch-size 4 \
  --fsdp-devices 4 \
  --num-workers 8 \
  --eval-interval 50 \
  --eval-batches 8 \
  --save-interval 100 \
  2>&1 | tee "${TRAIN_LOG}"

# The shell reaches this command only after training and the asynchronous
# step-999 checkpoint write both complete successfully.
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/mem/eval_shellgame_real_m5_memory_action_probe.py \
  2>&1 | tee "${EVAL_LOG}"
