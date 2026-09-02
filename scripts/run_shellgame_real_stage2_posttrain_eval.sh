#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
TRAIN_PID=214946
CONFIG=pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2
EXP=real306_currentrel_full80_interface_pi05_seed42_v1
CHECKPOINT_ROOT=${OPENPI_ROOT}/checkpoints/${CONFIG}/${EXP}
CHECKPOINT=${CHECKPOINT_ROOT}/20999
PORT=8017
RESULT_ROOT=${OPENPI_ROOT}/evaluation/shellgame_real/${EXP}_step20999
SERVER_LOG=${RESULT_ROOT}/server.log
STATUS_LOG=${RESULT_ROOT}/posttrain_status.log
NORMAL_JSON=${RESULT_ROOT}/normal_history.json
ZERO_JSON=${RESULT_ROOT}/zero_history.json
WRONG_JSON=${RESULT_ROOT}/wrong_episode_history.json
COMPARISON_JSON=${RESULT_ROOT}/memory_ablation_comparison.json
TRAINING_SUMMARY_JSON=${RESULT_ROOT}/training_summary.json
TRAINING_LOG=${OPENPI_ROOT}/train_${CONFIG}_${EXP}.log
CACHE_ROOT=/data2/hzl_workspace_for_pi_mem/.cache/shellgame_real_stage2/posttrain_jax

mkdir -p "${RESULT_ROOT}" "${CACHE_ROOT}"
exec > >(tee -a "${STATUS_LOG}") 2>&1

echo "$(date --iso-8601=seconds) waiting for training pid=${TRAIN_PID}"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 30
done
echo "$(date --iso-8601=seconds) training process exited"

for _ in $(seq 1 20); do
  if [[ -f "${CHECKPOINT}/_CHECKPOINT_METADATA" && -d "${CHECKPOINT}/params" ]]; then
    break
  fi
  sleep 15
done
if [[ ! -f "${CHECKPOINT}/_CHECKPOINT_METADATA" || ! -d "${CHECKPOINT}/params" ]]; then
  echo "final checkpoint is missing or incomplete: ${CHECKPOINT}" >&2
  exit 1
fi

.venv/bin/python scripts/mem/summarize_shellgame_real_stage2_training.py \
  --log "${TRAINING_LOG}" \
  --output "${TRAINING_SUMMARY_JSON}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap 'rc=$?; cleanup; trap - EXIT; exit "${rc}"' EXIT

cd "${OPENPI_ROOT}"
: >"${SERVER_LOG}"
echo "$(date --iso-8601=seconds) starting checkpoint server on GPUs 4,5,6,7"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}" \
  .venv/bin/python scripts/mem/serve_shellgame_real_stage2_cached.py \
    --port "${PORT}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 180); do
  if grep -q "server listening on" "${SERVER_LOG}"; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "policy server exited before becoming ready" >&2
    tail -n 120 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "timed out waiting for policy server" >&2
  tail -n 120 "${SERVER_LOG}" >&2
  exit 1
fi

echo "$(date --iso-8601=seconds) server ready; starting held-out exact-contract tests"
for history_mode in normal zero wrong_episode; do
  case "${history_mode}" in
    normal) output=${NORMAL_JSON} ;;
    zero) output=${ZERO_JSON} ;;
    wrong_episode) output=${WRONG_JSON} ;;
  esac
  eval_log=${RESULT_ROOT}/${history_mode}_history.log
  echo "$(date --iso-8601=seconds) evaluating history_mode=${history_mode}"
  OPENPI_WEBSOCKET_DISABLE_PING=1 \
  PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python scripts/mem/eval_shellgame_real_stage2_checkpoint.py \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --history-mode "${history_mode}" \
      --episodes-per-class 3 \
      --samples-per-frame 2 \
      --output "${output}" >"${eval_log}" 2>&1
  tail -n 80 "${eval_log}"
done

.venv/bin/python scripts/mem/compare_shellgame_real_stage2_memory_ablation.py \
  --normal "${NORMAL_JSON}" \
  --zero "${ZERO_JSON}" \
  --wrong-episode "${WRONG_JSON}" \
  --output "${COMPARISON_JSON}"
echo "$(date --iso-8601=seconds) all offline tests completed: ${COMPARISON_JSON}"
