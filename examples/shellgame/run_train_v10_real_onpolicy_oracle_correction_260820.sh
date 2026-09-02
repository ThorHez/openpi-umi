#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
STEPS=${STEPS:-500}
RESUME=${RESUME:-0}
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/499/params
EXP_NAME=${EXP_NAME:-absolute_eef7_v10_real_onpolicy_nom60_v6preserve25_oracle15_b12_${STEPS}steps_6gpu_260820}
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_v10_real_onpolicy_oracle_correction_260820/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_v10_real_onpolicy_${STEPS}steps_260820.log

cd "${OPENPI_ROOT}"
[[ -d "${INIT}" ]] || { echo "Missing init checkpoint: ${INIT}" >&2; exit 1; }
if [[ "${RESUME}" == "1" ]]; then
  [[ -d "${OUTPUT}" ]] || { echo "Cannot resume missing output: ${OUTPUT}" >&2; exit 1; }
else
  [[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }
fi

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((35 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=35"
if (( available_kb < required_kb )); then
  echo "Refusing to train with less than 35 GiB free" >&2
  exit 1
fi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v9_safe_erroraware
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V10 real-on-policy Oracle correction fine-tune"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 steps=${STEPS}"
echo "source_mass=nominal:0.60,v6_preservation:0.25,v10_real_onpolicy_oracle:0.15"
echo "init=${INIT} output=${OUTPUT}"
echo "resume=${RESUME}"

OPENPI_RESUME_TRAINING=${RESUME} CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_v10_real_onpolicy_oracle_correction.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps "${STEPS}" \
  --warmup-steps 50 \
  --peak-lr 2e-6 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 125 \
  --eval-batches 20 \
  --save-interval 250 \
  --keep-period 250 \
  --gripper-loss-weight 4.0
