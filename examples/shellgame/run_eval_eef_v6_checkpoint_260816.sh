#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "usage: $0 LABEL CHECKPOINT GPU PORT OUTPUT LOG [guarded|unguarded] [NUM_TRIALS]" >&2
  exit 2
fi

LABEL=$1
CHECKPOINT=$2
GPU=$3
PORT=$4
OUTPUT=$5
LOG=$6
MODE=${7:-guarded}
NUM_TRIALS=${8:-20}
EVAL_SEED=${OPENPI_EVAL_SEED_OVERRIDE:-260813}

if ! [[ "${NUM_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TRIALS must be a positive integer, got: ${NUM_TRIALS}" >&2
  exit 2
fi

if ! [[ "${EVAL_SEED}" =~ ^[0-9]+$ ]]; then
  echo "OPENPI_EVAL_SEED_OVERRIDE must be a non-negative integer, got: ${EVAL_SEED}" >&2
  exit 2
fi

case "${MODE}" in
  guarded)
    REPLAN_STEPS=3
    CONTROL_ARGS=(
      --xy-before-z-reference nearest_cup
      --xy-before-z-latch
      --xy-before-z-threshold 0.005
      --xy-before-z-descent-epsilon 0.0005
    )
    ;;
  unguarded)
    # Match the latest V5 deployable baseline exactly: no simulator cup-pose
    # guard and a five-step receding-horizon execution interval.
    REPLAN_STEPS=${OPENPI_EVAL_REPLAN_STEPS_OVERRIDE:-5}
    CONTROL_ARGS=(--no-xy-before-z-enabled)
    ;;
  *)
    echo "Unknown mode: ${MODE}; expected guarded or unguarded" >&2
    exit 2
    ;;
esac

if ! [[ "${REPLAN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPLAN_STEPS must be a positive integer, got: ${REPLAN_STEPS}" >&2
  exit 2
fi

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache
JAX_COMPILATION_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_eef_v6_eval_${LABEL}
SERVER_LOG=${OPENPI_ROOT}/serve_eef_v6_${LABEL}_260816.log
export UV_CACHE_DIR JAX_COMPILATION_CACHE_DIR

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT}" >&2
  exit 1
fi

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

: >"${SERVER_LOG}"
: >"${LOG}"
CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  uv run python examples/shellgame/serve_old_tracker_full_absolute_eef.py \
    --checkpoint-dir "${CHECKPOINT}" \
    --port "${PORT}" \
    --num-sampling-steps 4 >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if grep -q "server listening on" "${SERVER_LOG}"; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "${LABEL}: policy server exited before becoming ready" >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "${LABEL}: timed out waiting for policy server" >&2
  tail -n 100 "${SERVER_LOG}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python examples/shellgame/eval_absolute_eef_fixed_history_xy_before_z_isolated.py \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --robosuite-root ../robosuite \
    --num-trials "${NUM_TRIALS}" \
    --seed "${EVAL_SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --max-policy-steps 150 \
    --video-out-path "${OUTPUT}" \
    --fps 10 \
    --initial-ball-cup random \
    --min-swaps 3 \
    --max-swaps 3 \
    --phase-instructions \
    --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
    --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
    --observation-position-frame absolute \
    --rot6d-convention openpi \
    --action-mode raw7 \
    --action-dim 7 \
    --osc-input-type absolute \
    --policy-input-mode history \
    --no-control-during-scripted-observation \
    --observe-eef-frames 0 \
    --num-frames 61 \
    --frame-stride 1 \
    --physics-debug \
    --physics-debug-window 30 \
    --websocket-reconnect-interval 4 \
    "${CONTROL_ARGS[@]}" 2>&1 | tee "${LOG}"

echo "${LABEL}: evaluation complete: ${OUTPUT}/result.json"
