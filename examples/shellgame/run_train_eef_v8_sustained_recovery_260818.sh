#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
DATA=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_balanced1200_260818
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
EXP_NAME=absolute_eef7_mixed_correction_v8_sustained_60_30_5_5_unique_b12_2k_6gpu_260818
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v8_260818/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v8_sustained_60_30_5_5_2k_260818.log

cd "${OPENPI_ROOT}"
[[ -d "${DATA}" ]] || { echo "Missing converted V8 dataset: ${DATA}" >&2; exit 1; }
[[ -f "${DATA}/sustained_recovery_v8_oracle_supervision_audit.json" ]] || {
  echo "Missing V8 conversion audit" >&2; exit 1;
}
[[ -d "${INIT}" ]] || { echo "Missing V6 initialization: ${INIT}" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v8_sustained
export HF_HOME=${WORKSPACE}/.codex_tmp/hf_home_eef_v8_sustained
export HF_DATASETS_CACHE=${WORKSPACE}/.codex_tmp/hf_eef_v8_sustained
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V8 sustained-recovery 60/30/5/5 unique-row training"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v8.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps 2000 \
  --warmup-steps 300 \
  --peak-lr 1e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 1000 \
  --gripper-loss-weight 4.0
