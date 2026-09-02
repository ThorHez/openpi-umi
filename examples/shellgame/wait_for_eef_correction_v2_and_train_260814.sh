#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
LEROBOT_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_correction_raw7_replan_v2_500ep_260813
AUDIT_FILE=${LEROBOT_ROOT}/v2_data_audit.json
PIPELINE_SESSION=eef_corr_v2_pipeline_260813
SERVER_SESSION=eef_corr_v2_server_260813
OLD_CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_260812/absolute_eef7_old_tracker_phase_balanced_b12_continue6000_phased_260812/5999/params
EXP_NAME=absolute_eef7_mixed_correction_v2_85_15_b12_3k_260814
WATCH_LOG=${OPENPI_ROOT}/wait_train_eef7_mixed_correction_v2_260814.log
TRAIN_LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v2_85_15_3k_260814.log

cd "${OPENPI_ROOT}"
printf '%s Waiting for converted V2 data audit: %s\n' "$(date '+%F %T')" "${AUDIT_FILE}" | tee -a "${WATCH_LOG}"

while [[ ! -f "${AUDIT_FILE}" ]]; do
  if ! tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null; then
    printf '%s ERROR: data pipeline exited before producing an audit report.\n' "$(date '+%F %T')" | tee -a "${WATCH_LOG}"
    exit 1
  fi
  sleep 30
done

if ! grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' "${AUDIT_FILE}"; then
  printf '%s ERROR: V2 data audit did not pass; training will not start.\n' "$(date '+%F %T')" | tee -a "${WATCH_LOG}"
  sed -n '1,240p' "${AUDIT_FILE}" | tee -a "${WATCH_LOG}"
  exit 1
fi

PARQUET_COUNT=$(find "${LEROBOT_ROOT}/data" -type f -name '*.parquet' | wc -l)
if [[ "${PARQUET_COUNT}" -ne 500 ]]; then
  printf '%s ERROR: expected 500 converted parquets, found %s.\n' "$(date '+%F %T')" "${PARQUET_COUNT}" | tee -a "${WATCH_LOG}"
  exit 1
fi

printf '%s Audit passed for 500 episodes. Releasing policy server GPU and starting training.\n' "$(date '+%F %T')" | tee -a "${WATCH_LOG}"
tmux kill-session -t "${SERVER_SESSION}" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_correction_v2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v2.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${OLD_CHECKPOINT}" \
  --steps 3000 \
  --warmup-steps 300 \
  --peak-lr 3e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 1000 \
  --overwrite 2>&1 | tee "${TRAIN_LOG}"
