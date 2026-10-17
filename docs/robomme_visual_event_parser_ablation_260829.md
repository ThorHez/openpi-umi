# RoboMME visual event parser ablation (2026-08-29)

## Question

The transition ablation showed that the persistent semantic table and hard/soft
update algebra are not the dominant problem.  Which visual-parser limitation
causes the learned operations to remain far below the 100% ceiling: local
temporal coverage, lack of causal cross-chunk state, spatial resolution, or
event routing?

## Controlled variants

All learned variants use the same unified operation heads, GT previous semantic
table, task/goal representation, anchor coordinates, train/dev/test split,
800-step budget, and seed 260902.

| Variant | Visual input | Event selection | Cross-chunk state | Purpose |
|---|---|---|---|---|
| Fixed-local4 | Frozen SigLIP 4x4, current non-overlapping 12-frame chunk | None | No | Deployable local baseline |
| Fixed-causal4 | Same as Fixed-local4 | None | Gated causal event state | Isolate recurrent visual history |
| Event-SigLIP4 | Frozen SigLIP 4x4, 12 frames sampled across the complete GT event | Oracle | No | Complete-event temporal upper bound |
| Event-RGB4 | Raw RGB cell mean/std/gradient on a 4x4 grid, same complete event | Oracle | No | Low-resolution RGB control |
| Event-RGB8 | Same raw RGB statistics on an 8x8 grid | Oracle | No | Isolate spatial resolution |

The event-window variants contain no hold examples.  Their event-type accuracy
only distinguishes write from swap and must not be interpreted as a deployable
event detector.  Their relevant metric is payload accuracy.

The requested 8x8 SigLIP cache was attempted twice, but restoring the 6.1 GB
full action checkpoint stalled before creating any artifact.  The paired raw
RGB 4x4/8x8 grid was used instead so that spatial resolution remains the only
changed variable in that comparison.

## Locked-test results

### Online fixed-chunk routing

| Metric | Fixed-local4 | Fixed-causal4 | Causal delta |
|---|---:|---:|---:|
| Event-type accuracy | 88.2% | 89.4% | +1.2 pp |
| Hold false-positive rate | 8.6% | 8.9% | +0.3 pp |
| Update exact-type recall | 51.0% | 69.1% | **+18.1 pp** |
| Update exact-type precision | 31.0% | 39.6% | **+8.6 pp** |
| Payload accuracy on GT updates | 53.7% | 54.4% | +0.7 pp |
| Full update recall | 23.5% | 29.5% | **+6.0 pp** |

The causal state substantially improves event recall/precision, but does not
reduce false positives and barely changes payload quality.  Its largest
task-level effect is on VideoUnmaskSwap: full update recall rises from 32.9%
to 45.6%.  VideoPlaceOrder remains at 2.5% full update recall in both rows.

### Complete-event payload and spatial resolution

| Variant | Write joint payload | Swap pair | All-update payload |
|---|---:|---:|---:|
| Event-SigLIP4 | 35.5% | 64.3% | 40.9% |
| Event-RGB4 | 38.8% | 67.9% | 44.3% |
| Event-RGB8 | **57.0%** | **89.3%** | **63.1%** |

The controlled RGB resolution change gives:

- all-update payload: +18.8 percentage points;
- joint write entity/region: +18.2 points;
- swap pair: +21.4 points.

Event-RGB8 also generalizes cleanly: train/dev/test payload is
64.6%/65.3%/63.1%, so the gain is not a train-only fit artifact.

Per-task Event-RGB4 to Event-RGB8 payload changes are:

| Task | RGB4 | RGB8 | Delta |
|---|---:|---:|---:|
| VideoUnmask | 53.3% | 76.7% | +23.4 pp |
| VideoUnmaskSwap | 44.3% | 68.4% | +24.1 pp |
| VideoPlaceOrder | 37.5% | 42.5% | +5.0 pp |

Spatial aliasing is therefore a confirmed major cause for Unmask and Swap,
but it explains only a small part of Place.

## What the temporal experiments show

Merely supplying a complete event does not automatically solve the parser.
Event-SigLIP4 reaches only 40.9% payload and its swap-pair result (64.3%) is
identical to Fixed-local4.  The current early/late mean representation does not
retain the ordered motion trajectory contained in the sampled event frames.

The task behavior is also different:

- complete event coverage helps VideoUnmask payload (30.0% to 56.7%);
- it does not improve Swap payload (43.0% to 38.0%);
- Fixed-local4's 92.5% Place payload collapses to 35.0% on complete event
  windows, while its actual Place full-update recall was already only 2.5%.

The high fixed-local Place payload is therefore not evidence of a working
visual event parser.  The model can predict much of the ordered write payload
from prompt/previous-table/chunk-completion correlations, but it cannot route a
Place completion online.  Complete-event averaging then dilutes the final
placement evidence rather than tracking pick/move/drop explicitly.

## Revised attribution

1. **Spatial resolution is a confirmed major bottleneck.**  A controlled 4x4
   to 8x8 change recovers 18.8 points of payload and 21.4 points of swap-pair
   accuracy.
2. **Cross-chunk causal visual state is useful but insufficient.**  It recovers
   18.1 points of update routing recall and 6.0 points of full update recall,
   primarily on Swap.
3. **The temporal aggregation itself is structurally weak.**  Seeing complete
   event frames through early/late averages does not exploit the trajectory.
4. **Completion routing remains unusable for long rollout.**  The best causal
   row still has 8.9% hold false positives.  Across dozens of hold decisions,
   persistent semantic memory will almost certainly be corrupted.
5. **Place requires explicit phase/multimodal evidence.**  Neither causal state
   nor 8x8 RGB solves its completion/payload jointly; gripper state and EEF-Z,
   or an explicit pick/move/drop phase state, remain necessary candidates.

The original checkpoint's 39.1% local hold false-positive rate also fell to
8.6% after using a less aggressive square-root class weighting in the
controlled local parser.  Thus inverse-frequency event weighting aggravated
the failure, but it was not the root cause: full update recall is still only
23.5% locally and 29.5% with causal state.

## Structural implication

The next MEM should not feed a larger vector into the existing two-micro-event
head.  It should use:

1. 8x8 or higher-resolution object/region ROI tokens;
2. a transient causal event state that preserves ordered before/moving/after
   evidence rather than temporal means;
3. a completion router trained for extremely low hold false-positive rate;
4. a payload decoder invoked only after completion commit;
5. RGB plus gripper/EEF phase evidence for Place;
6. the existing deterministic semantic table only after the visual operation
   is accepted.

This remains one unified architecture and does not require task-specific
output heads.

## Artifacts

- Parser model: `src/openpi/tasks/robomme/causal_visual_operation_parser.py`
- Training/evaluation: `scripts/mem/train_robomme_visual_operation_parser_ablation.py`
- RGB grid cache: `scripts/mem/cache_robomme_event_rgb_grid_features.py`
- Fixed local: `checkpoints/robomme_visual_parser_fixed_local4_seed260902_260829/result.json`
- Fixed causal: `checkpoints/robomme_visual_parser_fixed_causal4_seed260902_260829/result.json`
- Complete-event SigLIP4: `checkpoints/robomme_visual_parser_event4_seed260902_260829/result.json`
- Complete-event RGB4: `checkpoints/robomme_visual_parser_event_raw4_seed260902_260829/result.json`
- Complete-event RGB8: `checkpoints/robomme_visual_parser_event_raw8_seed260902_260829/result.json`
