#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CKPT_ROOT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_v10_failure_suffix_overfit_consolidation_260820/absolute_eef7_v10_failure_suffix_consolidation_30_30_20_20_xy4_lr2e6_b6_150steps_6gpu_260820
STEPS=${STEPS:-25 50 75 100 125 149}

cd "${OPENPI_ROOT}"
for step in ${STEPS}; do
  checkpoint=${CKPT_ROOT}/${step}
  output=${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_failure_suffix_consolidation_step${step}_exact3_seed260813_180steps_260820
  log=${OPENPI_ROOT}/eval_eef7_v10_failure_suffix_consolidation_step${step}_exact3_260820.log
  server_log=${OPENPI_ROOT}/serve_eef7_v10_failure_suffix_consolidation_step${step}_exact3_260820.log
  [[ -d "${checkpoint}/params" ]] || { echo "Missing checkpoint ${checkpoint}" >&2; exit 1; }
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] Evaluating consolidation checkpoint ${step}"
  V10_CHECKPOINT="${checkpoint}" \
  FT499_CHECKPOINT="${checkpoint}" \
  OUTPUT="${output}" \
  LOG="${log}" \
  FT_SERVER_LOG="${server_log}" \
  FT_GPU=3 EVAL_GPU=3 FT_PORT=8141 \
  EPISODE_INDICES=0,1,17 \
  CONDITIONS=recorded_v10_to_ft499 \
  MAX_POLICY_STEPS=180 \
    bash examples/shellgame/run_v10_ft_oracle_handoff_paired_260820.sh
done

