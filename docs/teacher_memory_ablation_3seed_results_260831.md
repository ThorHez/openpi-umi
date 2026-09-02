# Teacher-memory ablation: three-seed result

## Controlled protocol

- Training seeds: 42, 43, 44.
- Fixed episode-disjoint split: `split_seed=42`, 4,500 train and 500 validation episodes.
- Final evaluation: the same 240 held-out samples for every seed and variant.
- Training: 1,000 steps, batch size 12, six A100 GPUs, 12 frames per event.
- Within each seed, the two runs share initialization, data order, architecture,
  frozen updater/readout, optimizer, and budget. The only changed factor is the
  teacher latent-memory loss weight: 0 for state-only and 1 for teacher memory.

## Per-seed exact complete-state accuracy

| Seed | Variant | Update 1 | Update 2 | Final | Mean |
|---:|---|---:|---:|---:|---:|
| 42 | State only | 65.00 | 36.25 | 33.75 | 45.00 |
| 42 | + teacher memory | 81.67 | 74.17 | 50.00 | 68.61 |
| 43 | State only | 75.42 | 47.92 | 32.08 | 51.81 |
| 43 | + teacher memory | 83.75 | 45.42 | 32.92 | 54.03 |
| 44 | State only | 31.67 | 35.42 | 32.50 | 33.19 |
| 44 | + teacher memory | 89.58 | 71.67 | 49.58 | 70.28 |

All values are percentages.

## Three-seed aggregate

| Variant | Update 1 | Update 2 | Final | Mean |
|---|---:|---:|---:|---:|
| State only | 57.4 ± 22.9 | 39.9 ± 7.0 | 32.8 ± 0.9 | 43.3 ± 9.4 |
| + teacher memory | **85.0 ± 4.1** | **63.8 ± 15.9** | **44.2 ± 9.7** | **64.3 ± 8.9** |
| Paired gain | +27.64 ± 26.55 | +23.89 ± 22.87 | +11.39 ± 9.15 | +20.97 ± 17.58 |

Values are mean ± sample standard deviation across training seeds. The teacher
variant improves mean-stage and final accuracy in all three paired seeds, but
the effect size varies substantially: the mean-stage gains are 23.61, 2.22, and
37.08 percentage points. With only three seeds, the result supports a positive
representation/optimization effect under this recipe but does not precisely
estimate its magnitude or establish that this particular teacher is necessary.

## Reproduction assets

- Runner: `scripts/run_teacher_memory_3seed.sh`
- Aggregator: `scripts/summarize_teacher_memory_3seed.py`
- Machine-readable aggregate:
  `evaluation/shellgame/teacher_memory_necessity_12f_3seed_260831/result.json`
- Per-seed CSV:
  `evaluation/shellgame/teacher_memory_necessity_12f_3seed_260831/per_seed.csv`
- Raw logs:
  `evaluation/shellgame/teacher_memory_necessity_12f_3seed_260831/logs/`
- Checkpoints:
  `checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/teacher_necessity_12f_*_seed{43,44}_260831/999`

