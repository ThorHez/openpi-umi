#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CKPT_ROOT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"

for spec in "1000:0:8010" "1500:0:8011" "1999:0:8012"; do
  IFS=: read -r step gpu port <<<"${spec}"
  label=v10_continue_step${step}_replan8
  checkpoint=${CKPT_ROOT}/${step}
  output=${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_continue_step${step}_replan8_isolated20_seed260813_260821
  log=${OPENPI_ROOT}/eval_eef7_v10_continue_step${step}_replan8_isolated20_seed260813_260821.log
  [[ -d "${checkpoint}/params" ]] || { echo "Missing checkpoint ${checkpoint}" >&2; exit 1; }
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 1; }

  OPENPI_EVAL_REPLAN_STEPS_OVERRIDE=8 \
    bash examples/shellgame/run_eval_eef_v6_checkpoint_260816.sh \
      "${label}" "${checkpoint}" "${gpu}" "${port}" "${output}" "${log}" unguarded 20 \
      >"${OPENPI_ROOT}/launcher_${label}_260821.log" 2>&1
  echo "${label} completed"
done
