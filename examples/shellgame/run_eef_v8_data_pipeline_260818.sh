#!/usr/bin/env bash
set -euo pipefail

NUM_EPISODES=${1:-1200}
WORKERS=${2:-6}
TAG=${3:-balanced${NUM_EPISODES}_260818}

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
ROBOSUITE_ROOT=${WORKSPACE}/robosuite
RAW_ROOT=${ROBOSUITE_ROOT}/outputs/shellgame_onpolicy_eef_sustained_recovery_v8_${TAG}
LEROBOT_ROOT=${ROBOSUITE_ROOT}/outputs/shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_${TAG}
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999
LOG=${OPENPI_ROOT}/generate_eef_sustained_recovery_v8_${TAG}.log
SERVER_LOG=${OPENPI_ROOT}/serve_eef_sustained_recovery_v8_${TAG}.log

cd "${OPENPI_ROOT}"
[[ "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_EPISODES must be positive" >&2; exit 2; }
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be positive" >&2; exit 2; }
(( NUM_EPISODES % 3 == 0 )) || { echo "NUM_EPISODES must be divisible by 3" >&2; exit 2; }
[[ -d "${CHECKPOINT}" ]] || { echo "Missing V6 checkpoint: ${CHECKPOINT}" >&2; exit 1; }

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((50 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free space: need 50 GiB, have $((available_kb / 1024 / 1024)) GiB" >&2
  exit 1
fi

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v8_data
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V8 data pipeline"
echo "episodes=${NUM_EPISODES} workers=${WORKERS} raw=${RAW_ROOT} lerobot=${LEROBOT_ROOT}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

: >"${SERVER_LOG}"
CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
  --checkpoint-dir "${CHECKPOINT}" \
  --port 8000 \
  --num-sampling-steps 4 >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 300); do
  if grep -q "server listening on" "${SERVER_LOG}"; then ready=1; break; fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "V8 policy server exited before ready" >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
(( ready == 1 )) || { echo "Timed out waiting for V8 policy server" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run python examples/shellgame/generate_onpolicy_eef_sustained_recovery_dataset_v8_parallel.py \
  --host 127.0.0.1 \
  --port 8000 \
  --robosuite-root ../robosuite \
  --output "${RAW_ROOT}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-attempts "$((NUM_EPISODES * 36))" \
  --workers "${WORKERS}" \
  --dataset-seed 260821 \
  --policy-checkpoint-label v6_5999 \
  --prefix-steps 30,36,42 \
  --width 224 \
  --height 224 \
  --fps 10 \
  --anchor-steps 10 \
  --anchor-height-tolerance-mm 10 \
  --perturb-steps 6 \
  --offset-bin-tolerance-mm 2 \
  --max-open-steps 60 \
  --grasp-steps 10 \
  --min-lift-steps 20 \
  --descent-jitter-mm 2 \
  --hold-z-above-xy-mm 10 \
  --aligned-xy-mm 6 \
  --close-xy-mm 5 \
  --close-z-mm 3 \
  --close-hold-steps 3 \
  --slow-descent-mm 2 \
  --normal-descent-mm 8

uv run python examples/shellgame/audit_onpolicy_eef_sustained_recovery_dataset_v8.py \
  "${RAW_ROOT}" --expected-episodes "${NUM_EPISODES}"

convert_mode=--overwrite
if [[ -d "${LEROBOT_ROOT}" ]]; then convert_mode=--resume; fi
uv run python examples/shellgame/convert_shellgame_onpolicy_sustained_recovery_v8_to_lerobot_raw_action.py \
  --input "${RAW_ROOT}" \
  --output "${LEROBOT_ROOT}" \
  --action-horizon 16 \
  --image-size 224 \
  --fps 10 \
  --observation-position-frame absolute \
  --rot6d-convention openpi \
  --phase-instructions \
  --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --grasp-phase-ids 8,9,10,11 \
  "${convert_mode}"

echo "[$(date --iso-8601=seconds)] V8 data pipeline complete"
echo "raw=${RAW_ROOT}"
echo "lerobot=${LEROBOT_ROOT}"
