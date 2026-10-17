#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${ROOT}/.venv/bin/python
EVAL=${ROOT}/scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py
TAG=260901
LOG_DIR=${ROOT}/checkpoints/robomme_unified_framework_logs_${TAG}
mkdir -p "${LOG_DIR}"

run_seed() {
  local seed=$1
  local gpu=$2
  local training_dir=${ROOT}/checkpoints/robomme_unified_framework_no_carry_seed${seed}_${TAG}
  local log=${LOG_DIR}/interventions_no_carry_seed${seed}.log
  echo "[seed=${seed} gpu=${gpu}] causal interventions"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "${PYTHON}" "${EVAL}" \
      --training-dir "${training_dir}" \
      --split test \
      --modes normal zero_video reverse_chunks shuffle_episode_video \
      >"${log}" 2>&1
  echo "[seed=${seed} gpu=${gpu}] causal interventions completed"
}

run_seed 260951 5 &
job_5=$!
run_seed 260952 6 &
job_6=$!
run_seed 260953 7 &
job_7=$!

status=0
wait "${job_5}" || status=1
wait "${job_6}" || status=1
wait "${job_7}" || status=1
exit "${status}"
