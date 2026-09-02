#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
ROBOSUITE_ROOT=${WORKSPACE}/robosuite
RAW_ROOT=${ROBOSUITE_ROOT}/outputs/shellgame_onpolicy_eef_continuous_descent_v4_600ep_260815
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v4_260814/absolute_eef7_mixed_correction_v4_holdz25pct_b12_2k_6gpu_260814/1999
LOG=${OPENPI_ROOT}/generate_eef_continuous_descent_v4_parallel_600ep_260815.log
SERVER_LOG=${OPENPI_ROOT}/serve_eef_continuous_descent_v4_parallel_260815.log
UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_continuous_descent_v4
export UV_CACHE_DIR JAX_COMPILATION_CACHE_DIR

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((30 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free space: need at least 30 GiB, have $((available_kb / 1024 / 1024)) GiB" >&2
  exit 1
fi

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting 6-worker V4 generation"
echo "raw=${RAW_ROOT}"
echo "checkpoint=${CHECKPOINT}"
echo "free_gib=$((available_kb / 1024 / 1024))"

server_pid=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

: >"${SERVER_LOG}"
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
    --checkpoint-dir "${CHECKPOINT}" \
    --port 8000 \
    --num-sampling-steps 4 >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 90); do
  if grep -q 'server listening on' "${SERVER_LOG}"; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited before becoming ready"
    tail -n 100 "${SERVER_LOG}"
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "Timed out waiting for policy server"
  tail -n 100 "${SERVER_LOG}"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python examples/shellgame/generate_onpolicy_eef_continuous_descent_dataset_v4_parallel.py \
    --host 127.0.0.1 \
    --port 8000 \
    --robosuite-root ../robosuite \
    --output "${RAW_ROOT}" \
    --num-episodes 600 \
    --max-attempts 4800 \
    --workers 6 \
    --dataset-seed 260818 \
    --policy-checkpoint-label v4_1999 \
    --prefix-steps 18,24,30,36,39,42 \
    --width 224 \
    --height 224 \
    --fps 10 \
    --min-safe-height-mm 60 \
    --max-safe-height-mm 240 \
    --perturb-steps 6 \
    --offset-bin-tolerance-mm 3 \
    --pre-descent-steps 3 \
    --descend-steps 50 \
    --grasp-steps 10 \
    --descent-jitter-mm 2.5 \
    --max-preclose-xy-mm 5
