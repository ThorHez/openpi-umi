#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
ROBOSUITE_ROOT=${WORKSPACE}/robosuite
RAW_ROOT=${ROBOSUITE_ROOT}/outputs/shellgame_onpolicy_eef_correction_multiheight_holdz_v3_600ep_260814
LEROBOT_ROOT=${ROBOSUITE_ROOT}/outputs/shellgame_lerobot_onpolicy_eef_correction_multiheight_holdz_v3_600ep_260814
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v3_260814/absolute_eef7_mixed_correction_v3_switch5pct_b12_2k_6gpu_260814/1999
EXP_NAME=absolute_eef7_mixed_correction_v4_holdz25pct_b12_2k_6gpu_260814
FINAL_CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v4_260814/${EXP_NAME}/1999
LOG=${OPENPI_ROOT}/run_eef_holdz_v4_pipeline_260814.log
SERVER_LOG=${OPENPI_ROOT}/serve_eef_holdz_v3_data_generation_260814.log
UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_holdz_v4
HF_HOME=${WORKSPACE}/.codex_tmp/hf_home_eef_holdz_v4
HF_DATASETS_CACHE=${WORKSPACE}/.codex_tmp/hf_eef_holdz_v4
export UV_CACHE_DIR JAX_COMPILATION_CACHE_DIR HF_HOME HF_DATASETS_CACHE

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((45 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free space: need at least 45 GiB, have $((available_kb / 1024 / 1024)) GiB" >&2
  exit 1
fi

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting hold-Z V4 pipeline"
echo "raw=${RAW_ROOT}"
echo "lerobot=${LEROBOT_ROOT}"
echo "init=${CHECKPOINT}"

server_pid=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap cleanup_server EXIT

if [[ ! -f "${RAW_ROOT}/generation_summary.json" ]] || \
   ! grep -q '"accepted_episodes": 600' "${RAW_ROOT}/generation_summary.json"; then
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

  echo "[$(date --iso-8601=seconds)] Generating 600 balanced multi-height episodes"
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    uv run python examples/shellgame/generate_onpolicy_eef_correction_dataset_v3.py \
      --host 127.0.0.1 \
      --port 8000 \
      --robosuite-root ../robosuite \
      --output "${RAW_ROOT}" \
      --num-episodes 600 \
      --max-attempts 2400 \
      --dataset-seed 260816 \
      --policy-checkpoint-label v3_1999 \
      --prefix-steps 18,24,30,36,39,42 \
      --width 224 \
      --height 224 \
      --fps 10 \
      --min-offset-mm 6 \
      --max-offset-mm 30 \
      --min-safe-height-mm 60 \
      --max-safe-height-mm 240 \
      --xy-threshold-mm 5 \
      --min-recenter-steps 3 \
      --max-recenter-steps 12 \
      --descend-steps 30 \
      --grasp-steps 15
  cleanup_server
else
  echo "[$(date --iso-8601=seconds)] Raw dataset already complete; skipping generation"
fi

if [[ ! -f "${LEROBOT_ROOT}/holdz_oracle_supervision_audit.json" ]]; then
  echo "[$(date --iso-8601=seconds)] Converting and auditing LeRobot dataset"
  uv run python ../robosuite/robosuite/scripts/convert_shellgame_onpolicy_correction_v3_to_lerobot_raw_action.py \
    --input "${RAW_ROOT}" \
    --output "${LEROBOT_ROOT}" \
    --fps 10 \
    --image-size 224 \
    --action-horizon 16 \
    --observation-position-frame absolute \
    --rot6d-convention openpi \
    --phase-instructions \
    --overwrite
else
  echo "[$(date --iso-8601=seconds)] Converted dataset already audited; skipping conversion"
fi

if [[ ! -d "${FINAL_CHECKPOINT}" ]]; then
  echo "[$(date --iso-8601=seconds)] Starting V4 training on GPUs 0-5"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v4.py \
      --exp-name "${EXP_NAME}" \
      --init-checkpoint "${CHECKPOINT}/params" \
      --steps 2000 \
      --warmup-steps 300 \
      --peak-lr 3e-5 \
      --batch-size 12 \
      --num-workers 8 \
      --fsdp-devices 6 \
      --eval-interval 250 \
      --eval-batches 20 \
      --save-interval 1000 \
      --keep-period 2000 \
      --gripper-loss-weight 4.0
else
  echo "[$(date --iso-8601=seconds)] Final V4 checkpoint already exists; skipping training"
fi

echo "[$(date --iso-8601=seconds)] Hold-Z V4 pipeline complete"
