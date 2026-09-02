#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
EXP_NAME=absolute_eef7_mixed_correction_v8_optimized_dynamic_60_30_5_5_unique_b12_2k_6gpu_260818
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v8_optimized_260818/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v8_optimized_continue_2k_to3k_lr1e6_260819.log

cd "${OPENPI_ROOT}"
[[ -d "${OUTPUT}/1999/train_state" ]] || {
  echo "Missing resumable optimized-V8 step-1999 train state" >&2
  exit 1
}

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((45 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=45"
if (( available_kb < required_kb )); then
  echo "Refusing to resume with less than 45 GiB free" >&2
  exit 1
fi

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v8_optimized_sustained
export HF_HOME=${WORKSPACE}/.codex_tmp/hf_home_eef_v8_optimized_sustained
export HF_DATASETS_CACHE=${WORKSPACE}/.codex_tmp/hf_eef_v8_optimized_sustained
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Resuming optimized-V8 from step 2000 to 3000"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 terminal_lr=1e-6"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v8_optimized_continue.py \
  --exp-name "${EXP_NAME}" \
  --steps 3000 \
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
