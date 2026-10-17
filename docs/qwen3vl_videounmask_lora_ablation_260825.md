# Qwen3-VL VideoUnmask LoRA A/B experiment

Date: 2026-08-25

## Question

Can VideoUnmask demonstration video supervision first give Qwen3-VL the
goal-conditioned occlusion-memory ability, and does initializing from the
validated ShellGame adapter plus rehearsal help?

## Leakage-free data contract

- Fixed episode-disjoint split: 80 train / 20 validation, seed 260823.
- Qwen input uses only the 66-frame `is_video_demo=True` prefix.
- Frames 0--31 show the three colored cubes; frames 32--65 show opaque
  containers covering the same positions.
- The execution phase and robot actions are never Qwen inputs.
- `choice_action` is used only to audit the visual pseudo-label. The detected
  target center is within 8 pixels of `choice_action` in 100/100 episodes
  (mean 3.84 px, max 6.40 px).
- Each episode supplies all three target-color queries. Each query has six
  paired visible-to-covered windows, three visible-only windows, and three
  masked-only rejection windows.
- Output is compact JSON with an 8x8 camera-relative cell, or
  `{"event":"insufficient_evidence"}`.

Manifest totals:

| Split | VideoUnmask | ShellGame replay | Total |
|---|---:|---:|---:|
| A train | 2,880 | 0 | 2,880 |
| B train | 2,880 | 960 (25%) | 3,840 |
| Validation | 720 | 0 | 720 |

## Experiments

- A: base `Qwen3-VL-4B-Instruct` -> VideoUnmask LoRA.
- B: ShellGame LoRA step 375 -> VideoUnmask LoRA with 25% ShellGame replay.
- Both use rank 16, alpha 32, LR 2e-5, batch 4, 400 optimizer steps on one
  A100, and the same seed. B step 300 is selected by validation loss; A step
  400 is selected.

Teacher-forcing validation:

| Model/checkpoint | Validation loss | Token accuracy |
|---|---:|---:|
| A step 400 | 0.02163 | 99.37% |
| B step 300 | 0.02248 | 99.01% |
| B step 400 | 0.02286 | 99.10% |

## Held-out greedy generation

All 720 validation windows are generated greedily. All outputs are valid JSON,
and event classification is 100% for both models.

| Metric | A | B |
|---|---:|---:|
| Paired memory exact 8x8 cell | 278/360 (77.2%) | 281/360 (78.1%) |
| Paired memory nearest container | 360/360 (100.0%) | 354/360 (98.3%) |
| Visible grounding exact 8x8 cell | 132/180 (73.3%) | 147/180 (81.7%) |
| Visible grounding nearest container | 180/180 (100.0%) | 180/180 (100.0%) |
| Masked-only correct rejection | 180/180 (100.0%) | 180/180 (100.0%) |

The stricter exact-cell score penalizes adjacent-cell predictions that still
select the correct physical container. B's six nearest-container errors are
the six paired-window variants of one held-out episode (episode 84), not six
independent scenes.

For the benchmark's actual first target prompt, restricted to the 15 held-out
single-target episodes:

| Metric | A | B |
|---|---:|---:|
| Paired exact cell (six windows/episode) | 59/90 (65.6%) | 67/90 (74.4%) |
| Paired nearest container | 90/90 (100.0%) | 84/90 (93.3%) |
| Visible exact cell | 33/45 (73.3%) | 39/45 (86.7%) |
| Visible nearest container | 45/45 (100.0%) | 45/45 (100.0%) |

A train-set per-color location prior reaches only 8/60 (13.3%) exact cell and
20/60 (33.3%) nearest container on the validation layouts. The learned models'
98--100% paired nearest-container result therefore cannot be explained by a
prompt-only color/location prior.

## ShellGame retention

The same 20 held-out ShellGame episodes used previously are replayed with the
original ten-frame evaluation contract.

| Metric | A | B |
|---|---:|---:|
| All ShellGame samples | 53/120 (44.2%) | 120/120 (100.0%) |
| Swap events | 0/60 | 60/60 |
| Complete reveal + 3 swaps | 0/20 | 20/20 |
| Symbolic final slot | 0/20 | 20/20 |

The 25% rehearsal fully preserves the validated ShellGame behavior in this
probe. A learns VideoUnmask but has no completed-exchange ability.

## Conclusion

Both experiments successfully give Qwen3-VL a useful VideoUnmask semantic
ability before action learning. A is marginally better on the operational
nearest-container score in this small validation set; B is better on exact
visible grounding and retains ShellGame perfectly. The evidence does not show
that ShellGame initialization significantly improves VideoUnmask paired-memory
accuracy, but B is the better continual/multi-task checkpoint.

The validation set has only 20 independent episodes, and the six paired
windows per episode are correlated. Hard two-target execution is not solved by
this single-query contract. The next action-facing experiment should map the
predicted cell to the nearest detected container and run the 15 held-out
single-target demonstrations first. Before a paper claim, add wrong-video
pairing and a larger official split/online evaluation.

## Artifacts

```text
artifacts/videounmask_qwen3vl_sft_seed260823/
checkpoints/qwen3vl_videounmask_from_base_lora_A_260825/checkpoint-000400/
checkpoints/qwen3vl_videounmask_from_shellgame_replay25_B_260825/checkpoint-000300/
evaluation/robomme/qwen3vl_videounmask_A_step400_val20_all_variants.summary.json
evaluation/robomme/qwen3vl_videounmask_B_step300_val20_all_variants.summary.json
evaluation/shellgame/qwen3vl_videounmask_A_step400_shellgame_val20.summary.json
evaluation/shellgame/qwen3vl_videounmask_B_step300_shellgame_val20.summary.json
```
