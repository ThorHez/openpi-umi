#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CONFIG=pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m6_direction_prompt
EXP_NAME=real306_m6_direction_prompt_seed42_v1
FINAL_STEP=20999
M5_PARAMS=${OPENPI_ROOT}/checkpoints/pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_m5/real306_m5_memory_seed42_v1/999/params
TRAIN_LOG=${OPENPI_ROOT}/train_shellgame_real_m6_direction_prompt_seed42_v1.log
EVAL_LOG=${OPENPI_ROOT}/eval_shellgame_real_m6_direction_prompt_seed42_v1.log
CHECKPOINT=${OPENPI_ROOT}/checkpoints/${CONFIG}/${EXP_NAME}/${FINAL_STEP}
OUTPUT=${OPENPI_ROOT}/evaluation/shellgame_real/${EXP_NAME}_step${FINAL_STEP}/m6_direction_prompt_validation.json

cd "${OPENPI_ROOT}"

[[ -f "${M5_PARAMS}/_METADATA" ]] || {
  echo "Missing verified M5 checkpoint: ${M5_PARAMS}" >&2
  exit 1
}
export HF_HOME=${HF_HOME:-${OPENPI_ROOT}/.cache/shellgame_real_stage2/huggingface}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${HF_DATASETS_CACHE}"

env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  .venv/bin/python scripts/mem/train_shellgame_real_m6_direction_prompt.py \
  --exp-name "${EXP_NAME}" \
  --checkpoint "${M5_PARAMS}" \
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
  2>&1 | tee -a "${TRAIN_LOG}"

# This starts only after step 20999 and its asynchronous checkpoint save have
# completed successfully. Evaluation uses the exact held-out 31 episodes.
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  .venv/bin/python scripts/mem/eval_shellgame_real_m6_direction_prompt.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --samples-per-prompt 2 \
  2>&1 | tee "${EVAL_LOG}"
