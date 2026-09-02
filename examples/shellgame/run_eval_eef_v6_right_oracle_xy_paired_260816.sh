#!/usr/bin/env bash
set -euo pipefail

# Paired diagnostic on the 31 episodes whose target cup finished in the
# spatial-right slot in the existing seed=260813 100-episode V6 evaluation.
# Both modes use the same environment seeds and deterministic diffusion-noise
# seeds.  The oracle mode changes only world-frame XY after the policy has
# expressed the correct cup selection and started descending.

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999
OUTPUT=${OPENPI_ROOT}/evaluation/shellgame/eef7_v6_step5999_right31_oracle_xy_paired_seed260813_salt260816
LOG=${OPENPI_ROOT}/eval_eef7_v6_step5999_right31_oracle_xy_paired_260816.log
SERVER_LOG=${OPENPI_ROOT}/serve_eef7_v6_step5999_right31_oracle_xy_paired_260816.log
GPU=${GPU:-5}
PORT=${PORT:-8015}
RIGHT_EPISODES=0,1,7,8,12,13,14,16,17,18,23,24,25,26,27,28,35,38,39,41,42,50,55,57,58,61,67,81,89,96,98

export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_eef_v6_right_oracle_xy_260816

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
    echo "Policy server exited before becoming ready" >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
if (( ready == 0 )); then
  echo "Timed out waiting for policy server" >&2
  tail -n 100 "${SERVER_LOG}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python examples/shellgame/eval_absolute_eef_xy_residual_stage_ablation.py \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --robosuite-root ../robosuite \
    --num-trials 100 \
    --episode-indices "${RIGHT_EPISODES}" \
    --xy-residual-modes none,through_close \
    --seed 260813 \
    --deterministic-sample-salt 260816 \
    --replan-steps 5 \
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
    --physics-debug-window 150 \
    --websocket-reconnect-interval 4 \
    --xy-residual-close-hold-steps 10 2>&1 | tee "${LOG}"

echo "Evaluation complete: ${OUTPUT}/result.json"
