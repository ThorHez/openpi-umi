# RoboMME unified shared-framework ablation (2026-09-01)

## Question

Can one shared four-task checkpoint support the fixed-chunk memory analysis
without task-specific modules or per-task tuning, and which common components
are supported by matched three-seed evidence?

## Locked protocol

- Tasks: PickXTimes, VideoUnmask, VideoUnmaskSwap, VideoPlaceOrder.
- One checkpoint per training seed serves all four tasks.
- Shared components: 19-field schema, visual encoder, goal initializer, causal
  evidence scan, memory update, and state readout.
- Task variation: goal values and field-validity mask only.
- Input: every non-overlapping 12-frame chunk; no event-timing trigger.
- Train/dev/test: 280/60/60 episode-disjoint episodes, task balanced.
- Training: 2,000 steps, batch 4, identical optimizer and dev selection.
- Seeds: 260951, 260952, 260953.
- Test labels are used only after checkpoint selection for scoring.

The primary model uses a shared causal evidence scan and learned soft write. It
does not add a second recurrent carry through the output latent memory. The
three table comparisons each change one factor relative to this primary model.

## Three-seed test results

Mean ± sample standard deviation, percentage points:

| Variant | Field | State | Change | Hold | Final | Answer |
|---|---:|---:|---:|---:|---:|---:|
| Causal evidence + soft write | **91.6 ± 0.3** | **54.7 ± 2.8** | **31.8 ± 1.4** | **58.3 ± 3.3** | **29.4 ± 2.5** | **36.1 ± 4.2** |
| w/o causal evidence | 82.8 ± 0.1 | 25.8 ± 1.8 | 15.8 ± 4.8 | 27.3 ± 1.3 | 21.1 ± 5.4 | 28.3 ± 8.3 |
| unconditional write | 90.2 ± 1.2 | 48.3 ± 5.6 | 24.9 ± 4.7 | 51.8 ± 5.8 | 17.8 ± 5.9 | 25.0 ± 6.0 |
| + latent-memory carry | 90.5 ± 0.3 | 48.7 ± 1.4 | 22.5 ± 3.8 | 52.7 ± 2.2 | 17.8 ± 3.5 | 25.6 ± 1.0 |

`Answer` is a common evaluation rule for the final action query: the Pick
dynamic control tuple, every instruction-queried color-to-region binding, or the
queried ordinal-to-region binding. It is computed from fresh checkpoint
inference, not inferred from Field or Final.

Primary-model Answer by task:

| Task | Answer |
|---|---:|
| PickXTimes | 64.4 ± 3.8 |
| VideoUnmask | 20.0 ± 0.0 |
| VideoUnmaskSwap | 17.8 ± 7.7 |
| VideoPlaceOrder | 42.2 ± 13.9 |

The primary-minus-comparison State differences are positive in all three paired
seeds: +29.0 ± 3.9 points versus no causal evidence, +6.5 ± 3.1 versus
unconditional writing, and +6.1 ± 1.8 versus adding latent-memory carry.

## Causal interventions on the same primary checkpoints

All inputs, including Normal, are recomputed by the intervention evaluator;
the Normal row below therefore reports that re-run rather than copying the
checkpoint-selection evaluation above.

| Input | State | Change | Final | Answer | Gate |
|---|---:|---:|---:|---:|---:|
| Normal | 54.8 ± 2.8 | 32.0 ± 1.1 | 29.4 ± 2.5 | 36.1 ± 4.2 | 0.415 ± 0.019 |
| Zero video | 17.5 ± 7.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.113 ± 0.006 |
| Reversed chunks | 51.3 ± 3.2 | 28.4 ± 2.1 | 15.0 ± 4.4 | 21.7 ± 3.3 | 0.415 ± 0.019 |
| Within-task episode shuffle (length-normalized) | 52.6 ± 1.7 | 29.0 ± 2.4 | 27.2 ± 6.3 | 30.0 ± 7.3 | 0.415 ± 0.019 |

The decisive Answer is substantially more sensitive to order than dense State.
After controlling task and valid-history length, episode identity has a smaller
effect: the within-task shuffle lowers Answer by 6.1 points and State by 2.1.
The learned gate is not an event detector: its normal-video means are 0.414 on
change chunks and 0.415 on hold chunks.

## Sequence metric and action decision

Full-sequence exact accuracy is 0% for the primary model, no-causal model, and
unconditional-write model; adding latent carry gives 1.7 ± 2.9%. One transient
multi-field error invalidates an episode, so this metric is retained as a
limitation but removed from the paper table in favor of Answer.

The shared model was not promoted to closed-loop action evaluation. Its overall
Answer is only 36.1 ± 4.2%, with 20.0% on VideoUnmask and 17.8 ± 7.7% on
VideoUnmaskSwap. This fails the predeclared offline gate for spending closed-loop
rollouts and cannot support the current main-table action scores. The paper
therefore labels this experiment as an offline structural diagnostic and keeps
the deployment-matched action systems separate.

## Artifacts

- Machine-readable summary:
  `checkpoints/robomme_unified_framework_ablation_3seed_260901.json`
- Per-run training configs and raw results:
  `checkpoints/robomme_unified_framework_<variant>_seed<seed>_260901/`
- Per-checkpoint intervention outputs: `test_visual_dependence.json` in the
  primary run directories.
- Training runners:
  `scripts/mem/run_robomme_unified_framework_ablation_3seed.sh` and
  `scripts/mem/run_robomme_unified_framework_matched_extension_3seed.sh`.
- Evaluation runners:
  `scripts/mem/run_robomme_unified_framework_answer_eval_3seed.sh` and
  `scripts/mem/run_robomme_unified_framework_interventions_3seed.sh`.
