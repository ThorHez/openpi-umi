#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
PYTHON="$ROOT/.venv/bin/python"
DATASET="$ROOT/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10_h32"
EXP=real306_m6_direction_stage1_frame241_h32_dirloss010_b32_seed42_v1
H16_SERVICE=shellgame-real-m6-stage1-dirloss010-b32.service

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONUNBUFFERED=1

if [[ ! -f "$DATASET/conversion_audit.json" || ! -f "$DATASET/meta/info.json" ]]; then
  if [[ -e "$DATASET" ]]; then
    echo "Incomplete H32 dataset already exists; refusing to overwrite: $DATASET" >&2
    exit 1
  fi
  "$PYTHON" scripts/mem/convert_real_shellgame_stage2_epfirst.py \
    --action-horizon 32 \
    --repo-id local/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10_h32 \
    --output "$DATASET" \
    --audit-output artifacts/shellgame_real_306_stage2_h32_conversion_audit.json \
    --image-workers 8 \
    --min-free-gib 35
fi

"$PYTHON" scripts/mem/validate_shellgame_real_h32_dataset.py

while systemctl is-active --quiet "$H16_SERVICE"; do
  echo "$(date --iso-8601=seconds) waiting for H16 baseline service $H16_SERVICE"
  sleep 60
done

exec "$PYTHON" scripts/mem/run_shellgame_real_m6_direction_stage1_guarded.py \
  --action-horizon 32 \
  --exp-name "$EXP" \
  --max-steps 2000 \
  --interval 100 \
  --batch-size 32 \
  --eval-batch-size 16 \
  --episodes-per-class 3 \
  --min-early-stop-step 500 \
  --early-stop-patience 3 \
  --direction-loss-weight 0.1 \
  --direction-temperature 0.0005 \
  --overwrite
