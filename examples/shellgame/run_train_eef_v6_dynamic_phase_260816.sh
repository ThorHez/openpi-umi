#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v5_260816/absolute_eef7_mixed_correction_v5_balanced1200_60_30_5_5_b12_3k_6gpu_260816/2999/params
EXP_NAME=absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_3k_260816.log
UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v6_dynamic_phase
HF_HOME=${WORKSPACE}/.codex_tmp/hf_home_eef_v6_dynamic_phase
HF_DATASETS_CACHE=${WORKSPACE}/.codex_tmp/hf_eef_v6_dynamic_phase
export UV_CACHE_DIR JAX_COMPILATION_CACHE_DIR HF_HOME HF_DATASETS_CACHE

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -d "${INIT}" ]]; then
  echo "Missing initialization checkpoint: ${INIT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite existing experiment: ${OUTPUT}" >&2
  exit 1
fi

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((60 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free space: need at least 60 GiB, have $((available_kb / 1024 / 1024)) GiB" >&2
  exit 1
fi

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V6 dynamic phase-aware 60/30/5/3/2 action training"
echo "exp=${EXP_NAME}"
echo "init=${INIT}"
echo "gpus=0,1,2,3,4,5"
echo "free_gib=$((available_kb / 1024 / 1024))"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v6.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
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
  --gripper-loss-weight 4.0
