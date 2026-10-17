# Real-cup GT-supervised Qwen3-VL fine-tuning results (2026-08-26)

## Objective

Test whether the existing ShellGame Qwen3-VL LoRA can acquire real-world cup-swap perception when supervised by `data/cup_replay_buffer/labels.jsonl`, including the three exchange pairs and the final ball position.

## Ground truth and alignment

- Replay buffer: 100 episodes, 49,326 raw RGB frames at approximately 20 Hz.
- Sidecar labels: 100 rows, each with `initial_cup`, three `moves`, `final_cup`, `frames_per_move=80`, `n_observe_frames=241`, and a per-episode resampled `n_frames`.
- Label and RGB episode IDs are one-to-one.
- Label time is mapped back to raw time by
  `raw_index = round(label_index * (raw_length - 1) / (label_n_frames - 1))`.
- The three labeled moves deterministically roll out from `initial_cup` to `final_cup` for 100/100 episodes.
- A simple initial red-ball RGB audit covers 100/100 episodes and agrees with the sidecar label on 82/100. This heuristic is only an audit and never replaces the supplied GT.

## Samples and split

Episode-disjoint split with seed 260826:

- Train: 80 episodes.
- Validation: 20 episodes: `1, 10, 16, 24, 26, 27, 31, 32, 34, 38, 40, 45, 50, 56, 58, 60, 73, 82, 84, 93`.
- Local event sample: 12 frames uniformly spanning one GT exchange; target is the exchanged screen-position pair.
- Full observation sample: 12 frames spanning all three exchanges; target contains initial position, the three exchange pairs, and final position.
- Original training manifest: 240 local events + 80 full sequences.
- Balanced continuation manifest: 240 local events + 240 repeated full sequences.

The model always uses camera-relative positions: `screen_left_cup`, `screen_middle_cup`, and `screen_right_cup`.

## Training

Initialization:

`checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375`

Stage 1:

- 120 optimizer steps, six A100 GPUs, global batch 12.
- Learning rate `1e-5`, cosine decay, 10 warm-up steps.
- Output: `checkpoints/qwen3vl_real_cup_gt_sequence_lora_v1_260826/`.

Stage 2, sequence-balanced continuation:

- Initialized from stage-1 step 120.
- 90 optimizer steps, global batch 12.
- Learning rate `5e-6`, cosine decay, 5 warm-up steps.
- Output: `checkpoints/qwen3vl_real_cup_gt_sequence_balanced_lora_v2_260826/`.

## Held-out generation results

All numbers below use the same 20 unseen episodes, 60 local swap clips, and 20 full-observation clips.

| Model | JSON valid | Local pair | All 3 local moves exact | Local-event recurrent final | Full-clip initial | Full-clip final | Full JSON exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original checkpoint-375 | 77.5% | 35.0% | 5.0% | 65.0% | 10.0% | 5.0% | 0.0% |
| Stage 1, step 90 | 100% | 71.7% | 35.0% | 45.0% | 55.0% | 50.0% | 0.0% |
| Stage 1, step 120 | 98.8% | 71.7% | 35.0% | 45.0% | 60.0% | **65.0%** | 0.0% |
| Balanced, step 30 | 100% | 78.3% | **55.0%** | 60.0% | 80.0% | 50.0% | 0.0% |
| Balanced, step 60 | 100% | **80.0%** | **55.0%** | 60.0% | **85.0%** | 30.0% | 0.0% |
| Balanced, step 90 | 100% | **80.0%** | **55.0%** | 60.0% | 80.0% | 45.0% | 0.0% |

Counts for the principal results:

- Best local exchange-pair accuracy: 48/60.
- Best exact three-local-event sequence: 11/20 episodes.
- Best direct final-position accuracy from the full clip: 13/20 episodes.
- Exact one-shot full JSON containing correct initial position, all three moves, and final position: 0/20.

The `local-event recurrent final` metric is not a reliable replacement for event accuracy. The original checkpoint accidentally obtains 13/20 final positions despite only 1/20 exact three-move sequences, because multiple wrong swaps can cancel to the same endpoint. This directly demonstrates why final-position supervision alone cannot verify the intermediate history.

## Conclusion

The real labels make Qwen learn useful real-domain swap perception:

- Local pair accuracy rises from chance-level 35.0% to 80.0%.
- Exact three-event histories rise from 5.0% to 55.0%.
- Direct final-position prediction rises from 5.0% to 65.0% at the best stage-1 checkpoint.

This validates the basic adaptation path, but does not yet justify fully automatic pseudo-labeling. Forty-five percent of held-out episodes still contain at least one wrong local swap, and a single 12-frame whole-observation input is insufficient for an exact three-event JSON.

For the recurrent MEM pipeline, the preferred architecture is therefore:

1. Use the balanced step-60 checkpoint as a local event reader.
2. Feed one 12-frame GT- or detector-centered window at a time.
3. Apply the predicted pair recurrently in the symbolic teacher/updater.
4. Use final position only as an episode-level consistency filter, not as proof that the event path is correct.

Before labeling new unannotated demonstrations, the next required evaluation is sliding-window detection with real `no_event` and `incomplete_event` negatives. The present experiment uses GT-centered local windows and measures event recognition, not automatic boundary discovery.

## Artifacts

- Manifest builder: `scripts/mem/build_real_cup_qwen3vl_sft_manifest.py`
- Real-cup contract: `src/openpi/tasks/shellgame/real_cup_qwen3vl_sft_contract.py`
- Trainer: `scripts/mem/train_shellgame_qwen3vl_lora.py`
- Generator/evaluator: `scripts/mem/eval_real_cup_qwen3vl_lora.py`
- Data summary: `artifacts/real_cup_qwen3vl_gt_sft_v1_260826/summary.json`
- Original training manifest: `artifacts/real_cup_qwen3vl_gt_sft_v1_260826/train.jsonl`
- Balanced training manifest: `artifacts/real_cup_qwen3vl_gt_sft_v1_260826/train_balanced.jsonl`
- Validation manifest: `artifacts/real_cup_qwen3vl_gt_sft_v1_260826/val.jsonl`
- Best local-event summary: `evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/balanced_step60.summary.json`
- Best direct-final summary: `evaluation/shellgame/real_cup_qwen3vl_gt_lora_v1/step120.summary.json`

