#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
EXP_NAME=absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v6_continue_3k_to6k_lr3e6_260816.log
UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v6_dynamic_phase
HF_HOME=${WORKSPACE}/.codex_tmp/hf_home_eef_v6_dynamic_phase
HF_DATASETS_CACHE=${WORKSPACE}/.codex_tmp/hf_eef_v6_dynamic_phase
export UV_CACHE_DIR JAX_COMPILATION_CACHE_DIR HF_HOME HF_DATASETS_CACHE

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -d "${OUTPUT}/2999/train_state" ]]; then
  echo "Missing resumable V6 step-2999 train state: ${OUTPUT}/2999/train_state" >&2
  exit 1
fi

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((45 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free space: need at least 45 GiB, have $((available_kb / 1024 / 1024)) GiB" >&2
  exit 1
fi

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Resuming V6 from global step 3000 to 6000"
echo "exp=${EXP_NAME}"
echo "checkpoint_dir=${OUTPUT}"
echo "gpus=0,1,2,3,4,5"
echo "lr_after_step3000=3e-6"
echo "free_gib=$((available_kb / 1024 / 1024))"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v6_continue.py \
  --exp-name "${EXP_NAME}" \
  --steps 6000 \
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
