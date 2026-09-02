#!/usr/bin/env bash
set -euo pipefail

# Matched Fig. 2 ablation.  The only temporal carry allowed in this script is
# the full [B,128,64] memory-token tensor consumed by MemoryUpdateBlock.
# In particular, --causal-evidence-state must never be added here.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REQUESTED_PYTHON=/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python
if [[ -x "${REQUESTED_PYTHON}" ]]; then
  DEFAULT_PYTHON=${REQUESTED_PYTHON}
else
  DEFAULT_PYTHON=${ROOT}/.venv/bin/python
fi
PYTHON=${PYTHON:-${DEFAULT_PYTHON}}
TRAIN=${ROOT}/scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py
STEPS=${STEPS:-2000}
TAG=${TAG:-260901_core_memory_tokens}
LOG_DIR=${ROOT}/checkpoints/robomme_core_memory_token_logs_${TAG}
mkdir -p "${LOG_DIR}"

# Historical A/B runs have matching configs, but setting RETRAIN_AB=1 creates
# publication-grade fresh copies with the exact same current training entry
# point as C/D.
A_PATTERN=${ROOT}/checkpoints/robomme_unified_framework_no_causal_seed%s_260901
B_PATTERN=${ROOT}/checkpoints/robomme_unified_framework_no_causal_no_carry_seed%s_260901
RETRAIN_AB=${RETRAIN_AB:-1}

run_variant() {
  local seed=$1
  local gpu=$2
  local variant=$3
  local output=$4
  shift 4
  local log=${LOG_DIR}/${variant}_seed${seed}.log
  if [[ -s "${output}/${STEPS}/params" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[seed=${seed} gpu=${gpu}] reuse ${variant}: ${output}/${STEPS}/params"
    return
  fi
  echo "[seed=${seed} gpu=${gpu}] train ${variant}: ${output}"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "${PYTHON}" "${TRAIN}" \
      --output-dir "${output}" \
      --steps "${STEPS}" \
      --eval-every 100 \
      --save-every 500 \
      --seed "${seed}" \
      "$@" \
      >"${log}" 2>&1
  echo "[seed=${seed} gpu=${gpu}] completed ${variant}"
}

run_seed() {
  local seed=$1
  local gpu=$2
  local a_dir
  local b_dir
  printf -v a_dir "${A_PATTERN}" "${seed}"
  printf -v b_dir "${B_PATTERN}" "${seed}"
  if [[ "${RETRAIN_AB}" == "1" ]]; then
    a_dir=${ROOT}/checkpoints/robomme_core_memory_token_soft_seed${seed}_${TAG}
    b_dir=${ROOT}/checkpoints/robomme_core_memory_token_reset_soft_seed${seed}_${TAG}
    run_variant "${seed}" "${gpu}" token_carry_soft "${a_dir}" \
      --supervision-mode full --write-gate
    run_variant "${seed}" "${gpu}" reset_token_soft "${b_dir}" \
      --supervision-mode full --write-gate --no-recurrent-carry
  fi

  # C: remove only alpha; candidate memory is written unconditionally.
  run_variant "${seed}" "${gpu}" token_carry_unconditional \
    "${ROOT}/checkpoints/robomme_core_memory_token_unconditional_seed${seed}_${TAG}" \
    --supervision-mode full

  # D: retain the exact A architecture, but remove dense teacher-memory and
  # intermediate state-trajectory losses.  Intermediate labels are eval-only.
  run_variant "${seed}" "${gpu}" token_carry_soft_no_trajectory_teacher \
    "${ROOT}/checkpoints/robomme_core_memory_token_no_trajectory_teacher_seed${seed}_${TAG}" \
    --write-gate \
    --supervision-mode terminal_answer_only \
    --memory-loss-weight 0 \
    --online-hold-readout-loss-weight 0 \
    --online-transition-readout-loss-weight 0 \
    --online-hold-keep-loss-weight 0 \
    --terminal-answer-selection
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS=${SMOKE_STEPS:-2}
  TAG=${TAG}_smoke
  run_seed 260951 0
  exit 0
fi

run_seed 260951 0 & job_0=$!
run_seed 260952 1 & job_1=$!
run_seed 260953 2 & job_2=$!

status=0
wait "${job_0}" || status=1
wait "${job_1}" || status=1
wait "${job_2}" || status=1
exit "${status}"
