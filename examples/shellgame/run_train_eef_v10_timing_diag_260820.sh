#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
STEPS=${STEPS:-500}
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
EXP_NAME=${EXP_NAME:-absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_${STEPS}steps_6gpu_260820}
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_v10_timing_diag_${STEPS}steps_260820.log

cd "${OPENPI_ROOT}"
[[ -d "${INIT}" ]] || { echo "Missing init checkpoint: ${INIT}" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((50 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=50"
if (( available_kb < required_kb )); then
  echo "Refusing to train V10 with less than 50 GiB free" >&2
  exit 1
fi

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
# V10 has the same compiled model/train shapes as V9; reuse that cache.
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v9_safe_erroraware
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V10 timing diagnostic"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 steps=${STEPS}"
echo "source_mass=nominal:0.60,v6_preservation:0.30,v9_timing:0.10"
echo "v9_groups=hard:.02,low:.02,aligned:.01,front3_descent:.02,close_within3:.02,lift:.01"
echo "init=${INIT} output=${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps "${STEPS}" \
  --warmup-steps 50 \
  --peak-lr 3e-6 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 125 \
  --eval-batches 20 \
  --save-interval 250 \
  --keep-period 250 \
  --gripper-loss-weight 4.0
