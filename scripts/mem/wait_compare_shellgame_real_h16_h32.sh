#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
H32_SERVICE=${1:-shellgame-real-h32-direction-b32-v2.service}
H16_GUARD="$ROOT/guard_real306_m6_direction_stage1_frame241_dirloss010_b32_seed42_v1.json"
H32_GUARD="$ROOT/guard_real306_m6_direction_stage1_frame241_h32_dirloss010_b32_seed42_v1.json"
OUTPUT="$ROOT/evaluation/shellgame_real/action_horizon_h16_vs_h32_seed42/comparison.json"

while systemctl is-active --quiet "$H32_SERVICE"; do
  sleep 60
done

if [[ ! -f "$H16_GUARD" || ! -f "$H32_GUARD" ]]; then
  echo "Cannot compare: a guarded experiment did not complete" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
exec "$ROOT/.venv/bin/python" scripts/mem/compare_shellgame_real_action_horizons.py \
  --h16-guard "$H16_GUARD" \
  --h32-guard "$H32_GUARD" \
  --output "$OUTPUT"
