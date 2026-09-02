#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PYTHON=/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python
if [[ -x "${REQUESTED_PYTHON}" ]]; then
  DEFAULT_PYTHON=${REQUESTED_PYTHON}
else
  DEFAULT_PYTHON=${ROOT}/.venv/bin/python
fi
PYTHON=${PYTHON:-${DEFAULT_PYTHON}}
EVAL=${ROOT}/scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py
STEP=${STEP:-2000}
TAG=${TAG:-260901_core_memory_tokens}
LOG_DIR=${ROOT}/checkpoints/robomme_core_memory_token_logs_${TAG}
mkdir -p "${LOG_DIR}"

training_dir() {
  local variant=$1
  local seed=$2
  case "${variant}" in
    token_carry_soft)
      echo "${ROOT}/checkpoints/robomme_core_memory_token_soft_seed${seed}_${TAG}" ;;
    reset_token_soft)
      echo "${ROOT}/checkpoints/robomme_core_memory_token_reset_soft_seed${seed}_${TAG}" ;;
    token_carry_unconditional)
      echo "${ROOT}/checkpoints/robomme_core_memory_token_unconditional_seed${seed}_${TAG}" ;;
    token_carry_soft_no_trajectory_teacher)
      echo "${ROOT}/checkpoints/robomme_core_memory_token_no_trajectory_teacher_seed${seed}_${TAG}" ;;
    *) echo "unknown variant: ${variant}" >&2; return 2 ;;
  esac
}

run_eval() {
  local variant=$1
  local seed=$2
  local gpu=$3
  local run
  run=$(training_dir "${variant}" "${seed}")
  local checkpoint=${run}/${STEP}/params
  local output=${run}/test_visual_dependence_step${STEP}.json
  local log=${LOG_DIR}/eval_${variant}_seed${seed}_step${STEP}.log
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    return 1
  fi
  echo "[seed=${seed} gpu=${gpu}] evaluate ${variant}"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "${PYTHON}" "${EVAL}" \
      --training-dir "${run}" \
      --checkpoint "${checkpoint}" \
      --output "${output}" \
      --split test \
      --modes normal zero_video reverse_chunks shuffle_episode_video \
      >"${log}" 2>&1
}

run_seed() {
  local seed=$1
  local gpu=$2
  for variant in \
    token_carry_soft \
    reset_token_soft \
    token_carry_unconditional \
    token_carry_soft_no_trajectory_teacher
  do
    run_eval "${variant}" "${seed}" "${gpu}"
  done
}

run_seed 260951 0 & job_0=$!
run_seed 260952 1 & job_1=$!
run_seed 260953 2 & job_2=$!

status=0
wait "${job_0}" || status=1
wait "${job_1}" || status=1
wait "${job_2}" || status=1
exit "${status}"
