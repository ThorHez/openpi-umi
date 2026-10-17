#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
RAW_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_onpolicy_eef_correction_raw7_replan_v2_500ep_260813
LEROBOT_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_correction_raw7_replan_v2_500ep_260813
LOG=${OPENPI_ROOT}/generate_convert_eef_correction_replan_v2_500ep_260813.log
UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export UV_CACHE_DIR

cd "${OPENPI_ROOT}"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run python examples/shellgame/generate_onpolicy_eef_correction_dataset_v2.py \
    --host 127.0.0.1 \
    --port 8000 \
    --robosuite-root ../robosuite \
    --output "${RAW_ROOT}" \
    --num-episodes 500 \
    --max-attempts 900 \
    --dataset-seed 260815 \
    --width 224 \
    --height 224 \
    --fps 10 2>&1 | tee "${LOG}"

uv run python ../robosuite/robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py \
  --input "${RAW_ROOT}" \
  --output "${LEROBOT_ROOT}" \
  --fps 10 \
  --image-size 224 \
  --action-horizon 16 \
  --observation-position-frame absolute \
  --rot6d-convention openpi \
  --phase-instructions \
  --overwrite 2>&1 | tee -a "${LOG}"

uv run python examples/shellgame/audit_onpolicy_eef_correction_dataset_v2.py \
  --dataset "${LEROBOT_ROOT}" \
  --expected-episodes 500 2>&1 | tee -a "${LOG}"
