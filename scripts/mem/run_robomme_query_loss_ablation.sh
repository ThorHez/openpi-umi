#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir%/scripts/mem}"
python_bin="${ROBOMME_PYTHON:-${repo_root}/.venv/bin/python}"
log_dir="${repo_root}/artifacts/robomme_query_loss_ablation_logs_260829"
mkdir -p "${log_dir}"

# Baseline pooled_soft_causal seeds 260908..260910 already exist.  These nine
# jobs form the remaining three cells of the 2x2 query-loss factorial.
specs=(
  "ordinal_only:2.0:0.0:260908"
  "ordinal_only:2.0:0.0:260909"
  "ordinal_only:2.0:0.0:260910"
  "completeness_only:0.0:0.5:260908"
  "completeness_only:0.0:0.5:260909"
  "completeness_only:0.0:0.5:260910"
  "ordinal_completeness:2.0:0.5:260908"
  "ordinal_completeness:2.0:0.5:260909"
  "ordinal_completeness:2.0:0.5:260910"
)

run_one() {
  local gpu="$1"
  local spec="$2"
  local name ordinal_weight completeness_weight seed
  IFS=: read -r name ordinal_weight completeness_weight seed <<<"${spec}"
  local output_dir="${repo_root}/checkpoints/robomme_explicit_event_${name}_seed${seed}_260829"
  local log_file="${log_dir}/${name}_seed${seed}.log"
  echo "[query-loss] gpu=${gpu} variant=${name} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${python_bin}" "${repo_root}/scripts/mem/train_robomme_explicit_event_bottleneck_ablation.py" \
    --variant pooled_soft_causal \
    --output-dir "${output_dir}" \
    --steps 1600 \
    --operation-pretrain-steps 800 \
    --batch-size 12 \
    --eval-batch-size 3 \
    --ordinal-binding-weight "${ordinal_weight}" \
    --completeness-weight "${completeness_weight}" \
    --seed "${seed}" \
    >"${log_file}" 2>&1
  echo "[query-loss] complete variant=${name} seed=${seed}"
}

pids=()
for gpu in 0 1 2 3; do
  (
    for index in "${!specs[@]}"; do
      if (( index % 4 == gpu )); then
        run_one "${gpu}" "${specs[index]}"
      fi
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
