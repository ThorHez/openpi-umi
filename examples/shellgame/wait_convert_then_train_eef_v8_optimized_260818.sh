#!/usr/bin/env bash
set -euo pipefail

CONVERTER_PID=${1:?Usage: $0 CONVERTER_PID}
WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
DATA=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_optimized_balanced1200_260818
AUDIT=${DATA}/sustained_recovery_v8_optimized_oracle_supervision_audit.json
EXPECTED_EPISODES=1006
GPU_IDS=2,3,4,5,6,7
LOG=${OPENPI_ROOT}/wait_convert_then_train_eef_v8_optimized_260818.log

cd "${OPENPI_ROOT}"
exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Waiting for converter pid=${CONVERTER_PID}"

while kill -0 "${CONVERTER_PID}" 2>/dev/null; do
  converted=$(find "${DATA}" -type f -name 'episode_*.parquet' 2>/dev/null | wc -l)
  echo "[$(date --iso-8601=seconds)] conversion=${converted}/${EXPECTED_EPISODES}"
  sleep 30
done

echo "[$(date --iso-8601=seconds)] Converter exited; validating final dataset"
[[ -f "${AUDIT}" ]] || { echo "Missing final conversion audit: ${AUDIT}" >&2; exit 1; }

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
uv run python -c '
import json
from pathlib import Path
root = Path("/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_onpolicy_eef_sustained_recovery_v8_optimized_balanced1200_260818")
audit = json.loads((root / "sustained_recovery_v8_optimized_oracle_supervision_audit.json").read_text())
expected = 1006
converted = int(audit["converted_audit"]["converted_episodes"])
parquet = len(list(root.glob("data/chunk-*/episode_*.parquet")))
assert audit.get("ok") is True, audit
assert converted == expected, (converted, expected)
assert parquet == expected, (parquet, expected)
print(f"Validated optimized-V8 dataset: audit={converted}, parquet={parquet}")
'

echo "[$(date --iso-8601=seconds)] Conversion validated; starting training on GPUs ${GPU_IDS}"
exec env GPU_IDS=${GPU_IDS} bash examples/shellgame/run_train_eef_v8_optimized_sustained_recovery_260818.sh
