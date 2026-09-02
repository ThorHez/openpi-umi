#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
V10_CHECKPOINT=${V10_CHECKPOINT:-${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/499}
FT499_CHECKPOINT=${FT499_CHECKPOINT:-${OPENPI_ROOT}/checkpoints/pi0_shellgame_v10_real_onpolicy_oracle_correction_260820/absolute_eef7_v10_real_onpolicy_nom60_v6preserve25_oracle15_b12_500steps_6gpu_260820/499}
OUTPUT=${OUTPUT:-${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_ft499_oracle_step80_paired10_seed260813_260820}
LOG=${LOG:-${OPENPI_ROOT}/eval_eef7_v10_ft499_oracle_step80_paired10_260820.log}
V10_SERVER_LOG=${V10_SERVER_LOG:-${OPENPI_ROOT}/serve_eef7_v10_deterministic_handoff_260820.log}
FT_SERVER_LOG=${FT_SERVER_LOG:-${OPENPI_ROOT}/serve_eef7_ft499_deterministic_handoff_260820.log}

V10_GPU=${V10_GPU:-2}
FT_GPU=${FT_GPU:-3}
# Match the previously validated isolated evaluator: EGL and the primary
# policy server see the same single physical GPU.
EVAL_GPU=${EVAL_GPU:-${V10_GPU}}
V10_PORT=${V10_PORT:-8140}
FT_PORT=${FT_PORT:-8141}
EPISODE_INDICES=${EPISODE_INDICES:-0,1,2,3,4,7,9,12,16,17}
CONDITIONS=${CONDITIONS:-v10_full,ft499_full,v10_to_ft499,v10_to_oracle}
MAX_POLICY_STEPS=${MAX_POLICY_STEPS:-155}

need_v10=0
need_ft=0
case ",${CONDITIONS}," in
  *,v10_full,*|*,v10_to_ft499,*|*,v10_to_oracle,*|*,recorded_v10_to_ft499_to_v10,*) need_v10=1 ;;
esac
case ",${CONDITIONS}," in
  *,ft499_full,*|*,v10_to_ft499,*|*,recorded_v10_to_ft499,*|*,recorded_v10_to_ft499_to_v10,*) need_ft=1 ;;
esac

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache
export OPENPI_WEBSOCKET_DISABLE_PING=1

cd "${OPENPI_ROOT}"
if (( need_v10 )); then
  [[ -d "${V10_CHECKPOINT}" ]] || { echo "Missing V10 checkpoint: ${V10_CHECKPOINT}" >&2; exit 1; }
fi
if (( need_ft )); then
  [[ -d "${FT499_CHECKPOINT}" ]] || { echo "Missing FT499 checkpoint: ${FT499_CHECKPOINT}" >&2; exit 1; }
fi
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

v10_pid=""
ft_pid=""
cleanup() {
  for pid in "${v10_pid}" "${ft_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

: >"${LOG}"

if (( need_v10 )); then
  : >"${V10_SERVER_LOG}"
  CUDA_VISIBLE_DEVICES="${V10_GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  JAX_COMPILATION_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_v10_handoff_det \
    uv run python examples/shellgame/serve_old_tracker_full_absolute_eef_deterministic.py \
      --checkpoint-dir "${V10_CHECKPOINT}" --port "${V10_PORT}" --num-sampling-steps 4 \
      >"${V10_SERVER_LOG}" 2>&1 &
  v10_pid=$!
fi

if (( need_ft )); then
  : >"${FT_SERVER_LOG}"
  CUDA_VISIBLE_DEVICES="${FT_GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  JAX_COMPILATION_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_ft499_handoff_det \
    uv run python examples/shellgame/serve_old_tracker_full_absolute_eef_deterministic.py \
      --checkpoint-dir "${FT499_CHECKPOINT}" --port "${FT_PORT}" --num-sampling-steps 4 \
      >"${FT_SERVER_LOG}" 2>&1 &
  ft_pid=$!
fi

for _ in $(seq 1 180); do
  v10_ready=$((1 - need_v10))
  ft_ready=$((1 - need_ft))
  if (( need_v10 )); then
    grep -q "server listening on" "${V10_SERVER_LOG}" && v10_ready=1
  fi
  if (( need_ft )); then
    grep -q "server listening on" "${FT_SERVER_LOG}" && ft_ready=1
  fi
  if (( v10_ready && ft_ready )); then
    break
  fi
  if (( need_v10 )) && ! kill -0 "${v10_pid}" 2>/dev/null; then
    echo "V10 policy server exited before ready" >&2
    tail -n 100 "${V10_SERVER_LOG}" >&2
    exit 1
  fi
  if (( need_ft )) && ! kill -0 "${ft_pid}" 2>/dev/null; then
    echo "FT499 policy server exited before ready" >&2
    tail -n 100 "${FT_SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

if (( need_v10 )); then
  grep -q "server listening on" "${V10_SERVER_LOG}" || { echo "V10 server readiness timeout" >&2; exit 1; }
fi
if (( need_ft )); then
  grep -q "server listening on" "${FT_SERVER_LOG}" || { echo "FT499 server readiness timeout" >&2; exit 1; }
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python examples/shellgame/eval_v10_ft_oracle_handoff_paired.py \
    --host 127.0.0.1 \
    --port "${V10_PORT}" \
    --secondary-port "${FT_PORT}" \
    --robosuite-root ../robosuite \
    --num-trials 20 \
    --episode-indices "${EPISODE_INDICES}" \
    --conditions "${CONDITIONS}" \
    --seed 260813 \
    --deterministic-sample-salt 260820 \
    --replan-steps 8 \
    --switch-step 80 \
    --max-policy-steps "${MAX_POLICY_STEPS}" \
    --original-budget-steps 150 \
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
    --physics-debug-window "${MAX_POLICY_STEPS}" \
    --websocket-reconnect-interval 4 2>&1 | tee "${LOG}"

echo "Evaluation complete: ${OUTPUT}/result.json"
