#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-${root}/.venv/bin/python}"
train="${root}/scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py"
tag="${TAG:-260901_single_latent_confirm}"
steps="${STEPS:-2000}"
eval_every="${EVAL_EVERY:-100}"
save_every="${SAVE_EVERY:-500}"
log_dir="${root}/checkpoints/robomme_single_latent_ablation_logs_${tag}"
seeds=(260971 260972 260973)

mkdir -p "${log_dir}"

variant_args() {
    case "$1" in
        latent_soft)
            printf '%s\n' --write-gate
            ;;
        reset_soft)
            printf '%s\n' --write-gate --no-recurrent-carry
            ;;
        latent_unconditional)
            ;;
        latent_soft_no_trajectory_teacher)
            printf '%s\n' \
                --write-gate \
                --supervision-mode terminal_answer_only \
                --memory-loss-weight 0 \
                --online-hold-readout-loss-weight 0 \
                --online-transition-readout-loss-weight 0 \
                --online-hold-keep-loss-weight 0
            ;;
        *)
            echo "unknown variant: $1" >&2
            return 2
            ;;
    esac
}

run_variant() {
    local seed="$1"
    local gpu="$2"
    local variant="$3"
    local output="${root}/checkpoints/robomme_single_latent_${variant}_seed${seed}_${tag}"
    local log="${log_dir}/${variant}_seed${seed}.log"
    local args=()
    mapfile -t args < <(variant_args "${variant}")

    if [[ -s "${output}/result.json" ]]; then
        echo "[skip] variant=${variant} seed=${seed}: ${output}/result.json exists"
        return 0
    fi
    if [[ -e "${output}" ]]; then
        echo "refusing non-empty/incomplete output without explicit cleanup: ${output}" >&2
        return 1
    fi

    echo "[start] gpu=${gpu} seed=${seed} variant=${variant} output=${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.35}" \
    PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" "${train}" \
            --output-dir "${output}" \
            --steps "${steps}" \
            --eval-every "${eval_every}" \
            --save-every "${save_every}" \
            --batch-size 4 \
            --seed "${seed}" \
            --terminal-answer-selection \
            "${args[@]}" \
            >"${log}" 2>&1
    echo "[done] gpu=${gpu} seed=${seed} variant=${variant}"
}

if [[ "${SMOKE_ONLY:-false}" == "true" ]]; then
    smoke_seed="${SMOKE_SEED:-260970}"
    run_variant "${smoke_seed}" 1 latent_soft & smoke_1=$!
    run_variant "${smoke_seed}" 2 reset_soft & smoke_2=$!
    run_variant "${smoke_seed}" 3 latent_unconditional & smoke_3=$!
    run_variant "${smoke_seed}" 4 latent_soft_no_trajectory_teacher & smoke_4=$!
    smoke_status=0
    for pid in "${smoke_1}" "${smoke_2}" "${smoke_3}" "${smoke_4}"; do
        if ! wait "${pid}"; then
            smoke_status=1
        fi
    done
    exit "${smoke_status}"
fi

# Rotate variants across GPUs so a variant is not confounded with one device.
lane_1() {
    run_variant "${seeds[0]}" 1 latent_soft
    run_variant "${seeds[1]}" 1 latent_soft_no_trajectory_teacher
    run_variant "${seeds[2]}" 1 latent_unconditional
}
lane_2() {
    run_variant "${seeds[0]}" 2 reset_soft
    run_variant "${seeds[1]}" 2 latent_soft
    run_variant "${seeds[2]}" 2 latent_soft_no_trajectory_teacher
}
lane_3() {
    run_variant "${seeds[0]}" 3 latent_unconditional
    run_variant "${seeds[1]}" 3 reset_soft
    run_variant "${seeds[2]}" 3 latent_soft
}
lane_4() {
    run_variant "${seeds[0]}" 4 latent_soft_no_trajectory_teacher
    run_variant "${seeds[1]}" 4 latent_unconditional
    run_variant "${seeds[2]}" 4 reset_soft
}

lane_1 & pid_1=$!
lane_2 & pid_2=$!
lane_3 & pid_3=$!
lane_4 & pid_4=$!

status=0
for pid in "${pid_1}" "${pid_2}" "${pid_3}" "${pid_4}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
