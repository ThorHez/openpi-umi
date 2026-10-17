#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${ROOT}/.venv/bin/python
EVAL=${ROOT}/scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py
STEP=${STEP:-2000}
GROUP=${GROUP:-all}

run_eval() {
  local training_dir=$1
  local gpu=$2
  local log=${training_dir}/test_visual_dependence_step${STEP}.log
  local output=${training_dir}/test_visual_dependence_step${STEP}.json
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    "${PYTHON}" "${EVAL}" \
      --training-dir "${training_dir}" \
      --checkpoint "${training_dir}/${STEP}/params" \
      --output "${output}" \
      --split test \
      --modes normal zero_video reverse_chunks shuffle_episode_video \
      >"${log}" 2>&1
  echo "fixed-step intervention complete: ${output}"
}

run_baselines() {
  local jobs=()
  for seed in 260951 260952 260953; do
    local gpu=$((4 + seed - 260951))
    (
      run_eval "${ROOT}/checkpoints/robomme_unified_framework_no_carry_seed${seed}_260901" "${gpu}"
      run_eval "${ROOT}/checkpoints/robomme_unified_framework_no_causal_no_carry_seed${seed}_260901" "${gpu}"
      run_eval "${ROOT}/checkpoints/robomme_unified_framework_unconditional_no_carry_seed${seed}_260901" "${gpu}"
    ) & jobs+=("$!")
  done
  local status=0
  for job in "${jobs[@]}"; do
    wait "${job}" || status=1
  done
  return "${status}"
}

run_no_teacher() {
  local jobs=()
  for seed in 260951 260952 260953; do
    local gpu=$((1 + seed - 260951))
    run_eval "${ROOT}/checkpoints/robomme_unified_framework_no_trajectory_teacher_seed${seed}_260901_correct_hidden_state" "${gpu}" & jobs+=("$!")
  done
  local status=0
  for job in "${jobs[@]}"; do
    wait "${job}" || status=1
  done
  return "${status}"
}

case "${GROUP}" in
  baselines) run_baselines ;;
  no_teacher) run_no_teacher ;;
  all)
    run_baselines
    run_no_teacher
    ;;
  *)
    echo "GROUP must be baselines, no_teacher, or all" >&2
    exit 2
    ;;
esac
