# VideoUnmask causal MEM structural readout ablation (2026-08-31)

## Question

Determine whether the 72% closed-loop ceiling of the current causal recurrent
MEM is mainly caused by (1) semantic entity-to-region decoding, (2) conversion
from semantic regions to runtime image coordinates, or (3) the MME action
controller.

## Fixed protocol

- Task: RoboMME `VideoUnmask`, official test split.
- Episodes: the same 50 episodes, ids 0--49.
- Maximum rollout length: 1300 environment steps.
- Action checkpoint:
  `mme_vla_suite/symbolic-grounded-subgoal/79999`.
- MEM checkpoint:
  `robomme_explicit_event_pooled_soft_causal_seed260908_260829`.
- Baseline and all ablations use the same demonstration RGB, action protocol,
  target prompts, and oracle simple action phase. No simulator target region or
  grounded coordinate is used by either proposed correction.

## Independent modifications

### A. Joint semantic assignment

The old readout independently applied argmax to each entity field. This can
produce an unknown region for a required entity or assign two colored cubes to
the same physical container. The new readout retains the MEM's soft
entity-region probabilities and finds the maximum-joint-probability one-to-one
assignment over the episode-local regions.

This changes only the semantic readout of MEM. It does not change RGB
segmentation, image coordinates, action inference, or the recurrent updater.

### B. Unique visual anchor grounding

The old grounding snapped every historical region anchor independently to its
nearest currently detected container. When the segmenter detected only two of
three containers, two semantic regions could collapse to the same image point.
The new grounding jointly assigns each visible candidate at most once. If a
container is missing, it retains that region's demonstration anchor rather
than duplicating another candidate.

This changes the semantic-region-to-pixel bridge, not MEM itself.

## Closed-loop ablation

| Joint semantic assignment | Unique anchor grounding | Success | Change vs. baseline |
|---:|---:|---:|---:|
| No | No | 36/50 = 72% | -- |
| Yes | No | 39/50 = 78% | +3 episodes / +6 pp |
| No | Yes | 38/50 = 76% | +2 episodes / +4 pp |
| Yes | Yes | **41/50 = 82%** | **+5 episodes / +10 pp** |

The combined configuration was repeated using action seeds 7, 17, and 27:

| Action seed | Success |
|---:|---:|
| 7 | 41/50 = 82% |
| 17 | 41/50 = 82% |
| 27 | 41/50 = 82% |
| Mean +/- population SD | **82.0 +/- 0.0%** |

The success/failure episode sets are identical across the three action seeds,
so they must not be presented as 150 independent evaluation episodes. The 95%
Wilson interval for one unique set of 41/50 episodes is 69.2%--90.2%.

## Episode-level causal evidence

- Semantic assignment alone rescues episodes 3, 39, and 43. These are
  multi-target cases in which independent readout emitted an unknown target or
  placed two target colors in the same region.
- Unique anchor grounding alone rescues episodes 13 and 44. In each case only
  two containers were detected. Retaining the unmatched historical anchor
  produces a coordinate within about one pixel of the simulator audit point.
- The combined configuration rescues all five episodes and does not turn any
  baseline success into a failure.

This disjoint rescue set is strong evidence that the two corrections address
separate failure mechanisms rather than obtaining a noisy aggregate gain.

## Remaining bottleneck

The combined model fails episodes 11, 14, 15, 16, 19, 32, 37, 38, and 42. In
all nine, the selected semantic region is wrong. Across all 50 episodes:

- every required grounded container coordinate is within 20 pixels of the
  simulator audit coordinate in 41/50 episodes;
- the action controller succeeds in all 41 of those episodes;
- it fails in all nine episodes with an incorrect semantic region.

Thus the measured closed-loop success is exactly aligned with correct semantic
grounding on this run. The action checkpoint is not the active bottleneck for
these 50 VideoUnmask episodes.

## Conclusion: which MEM component is key?

Within MEM, the key missing component is **stable entity-to-region binding**,
not a larger recurrent updater or a more complex action interface. Joint
semantic assignment provides the largest isolated MEM-side gain (+6 pp), but
the remaining failures show that a test-time constraint cannot recover the
correct identity when the underlying probability table confidently binds the
wrong color to the wrong anchor.

The next training-level ablation should therefore keep the recurrent updater
and action policy fixed and add only:

1. an early entity-anchor binding loss, supervised from simulator-privileged
   color/region correspondences during training only;
2. a one-to-one structured matching loss over all visible entities, not only
   the target colors named in the prompt;
3. a temporal retention loss that preserves the initial binding until a
   visually supported transition occurs.

The corresponding minimal ablation is `current combined`, `+binding loss`,
`+retention loss`, and `+both`. Reaching 43/50 would exceed an 85% point target,
but the present 41/50 result does not yet support that claim.

## Artifacts

- Implementation:
  `robomme_policy_learning/examples/robomme/region_grounding.py`
- Runtime bridge:
  `robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- Unit tests:
  `robomme_policy_learning/examples/robomme/region_grounding_test.py`
- Seed-7 ablation launcher:
  `robomme_policy_learning/scripts/run_causal_mem_videounmask_ablation_seed7_50ep.sh`
- Winner seed-17/27 launcher:
  `robomme_policy_learning/scripts/run_causal_mem_videounmask_winner_seed17_27_50ep.sh`
- Evaluation roots:
  `robomme_policy_learning/runs/evaluation/causal-event-mem-videounmask-*-test50-seed*-260831`
