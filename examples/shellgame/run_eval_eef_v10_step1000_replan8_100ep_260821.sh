#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/1000
LABEL=v10_continue_step1000_replan8_formal100
OUTPUT=${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_continue_step1000_replan8_isolated100_seed260813_260821
LOG=${OPENPI_ROOT}/eval_eef7_v10_continue_step1000_replan8_isolated100_seed260813_260821.log
LAUNCHER_LOG=${OPENPI_ROOT}/launcher_${LABEL}_260821.log

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"
[[ -d "${CHECKPOINT}/params" ]] || { echo "Missing checkpoint ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

OPENPI_EVAL_REPLAN_STEPS_OVERRIDE=8 \
  bash examples/shellgame/run_eval_eef_v6_checkpoint_260816.sh \
    "${LABEL}" "${CHECKPOINT}" 0 8010 "${OUTPUT}" "${LOG}" unguarded 100 \
    >"${LAUNCHER_LOG}" 2>&1

echo "${LABEL} completed: ${OUTPUT}/result.json"
