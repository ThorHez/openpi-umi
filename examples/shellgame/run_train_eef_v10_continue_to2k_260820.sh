#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5}
STEPS=${STEPS:-2000}
EXP_NAME=absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_v10_continue_500_to_${STEPS}_lr3e7_260820.log

cd "${OPENPI_ROOT}"
[[ -d "${OUTPUT}/499/train_state" ]] || {
  echo "Missing resumable V10 step-499 train state: ${OUTPUT}/499/train_state" >&2
  exit 1
}

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((80 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=80"
if (( available_kb < required_kb )); then
  echo "Refusing to continue V10 with less than 80 GiB free" >&2
  exit 1
fi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v9_safe_erroraware
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Resuming general V10 from global step 500 to ${STEPS}"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6"
echo "source_mass=nominal:0.60,v6_preservation:0.30,v9_timing:0.10"
echo "lr_after_step500=3e-7 resume_full_train_state=true"
echo "checkpoint_dir=${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_continue.py \
  --exp-name "${EXP_NAME}" \
  --steps "${STEPS}" \
  --warmup-steps 50 \
  --peak-lr 3e-6 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 1 \
  --gripper-loss-weight 4.0

