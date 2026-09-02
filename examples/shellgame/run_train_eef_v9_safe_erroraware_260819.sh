#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
STEPS=${STEPS:-4000}
V9_DATA=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819
V6_DATA=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816
NOMINAL_DATA=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7
METRICS=${V9_DATA}/xy_sampling_metrics_v9.npz
AUDIT=${V9_DATA}/safe_balanced_recovery_v9_oracle_supervision_audit.json
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params
EXP_NAME=${EXP_NAME:-absolute_eef7_mixed_correction_v9_nom60_v6replay15_v9error25_b12_${STEPS}steps_6gpu_260819}
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v9_260819/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_mixed_correction_v9_nom60_v6replay15_v9error25_${STEPS}steps_260819.log

cd "${OPENPI_ROOT}"
for path in "${V9_DATA}" "${V6_DATA}" "${NOMINAL_DATA}" "${INIT}"; do
  [[ -d "${path}" ]] || { echo "Missing required directory: ${path}" >&2; exit 1; }
done
for path in "${METRICS}" "${AUDIT}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

# Three retained checkpoints plus one asynchronous write need roughly 40 GiB.
# Keep a much larger safety margin because /data2 is shared with other jobs.
available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((100 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=100"
if (( available_kb < required_kb )); then
  echo "Refusing to train V9 with less than 100 GiB free" >&2
  exit 1
fi

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v9_safe_erroraware
# Reuse the existing 59 GiB nominal Arrow cache.  A fresh experiment-local
# cache would duplicate the 5,000-episode nominal dataset before step zero and
# leave too little room for checkpoints.  V6/V9 fingerprints are distinct and
# will be added to this shared datasets cache once.
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting V9 safe error-aware mixed replay training"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 steps=${STEPS}"
echo "source_mass=nominal:0.60,v6_replay:0.15,v9_erroraware:0.25"
echo "v9_groups=hard_initial:0.10,low_1_4mm_le40mm:0.07,aligned:0.04,grasp:0.02,lift:0.02"
echo "init=${INIT} metrics=${METRICS} output=${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v9.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps "${STEPS}" \
  --warmup-steps 300 \
  --peak-lr 1e-5 \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6 \
  --eval-interval 250 \
  --eval-batches 20 \
  --save-interval 1000 \
  --keep-period 2000 \
  --gripper-loss-weight 4.0
