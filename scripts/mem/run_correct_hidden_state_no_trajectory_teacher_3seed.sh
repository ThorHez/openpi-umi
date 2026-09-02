#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${ROOT}/.venv/bin/python
TRAIN=${ROOT}/scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py
TAG=${TAG:-260901_correct_hidden_state}
STEPS=${STEPS:-2000}
SMOKE=${SMOKE:-0}
LOG_DIR=${ROOT}/checkpoints/robomme_correct_hidden_state_logs_${TAG}
mkdir -p "${LOG_DIR}"

run_seed() {
  local seed=$1
  local gpu=$2
  local output=${ROOT}/checkpoints/robomme_unified_framework_no_trajectory_teacher_seed${seed}_${TAG}
  local log=${LOG_DIR}/no_trajectory_teacher_seed${seed}.log
  echo "[seed=${seed} gpu=${gpu}] no_trajectory_teacher -> ${output}"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "${PYTHON}" "${TRAIN}" \
      --output-dir "${output}" \
      --steps "${STEPS}" \
      --eval-every 100 \
      --save-every 500 \
      --seed "${seed}" \
      --write-gate \
      --causal-evidence-state \
      --no-recurrent-carry \
      --supervision-mode terminal_answer_only \
      --memory-loss-weight 0 \
      --online-hold-readout-loss-weight 0 \
      --online-transition-readout-loss-weight 0 \
      --online-hold-keep-loss-weight 0 \
      --terminal-answer-selection \
      >"${log}" 2>&1
  echo "[seed=${seed} gpu=${gpu}] no_trajectory_teacher completed"
}

if [[ "${SMOKE}" == "1" ]]; then
  run_seed 260951 1
  exit 0
fi

run_seed 260951 1 & job_1=$!
run_seed 260952 2 & job_2=$!
run_seed 260953 3 & job_3=$!

status=0
wait "${job_1}" || status=1
wait "${job_2}" || status=1
wait "${job_3}" || status=1
exit "${status}"
