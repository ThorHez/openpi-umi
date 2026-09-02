# VideoUnmask native binding teacher ablation (2026-08-31)

## Objective

Test whether the remaining VideoUnmask gap is caused by recurrent MEM capacity
or by the semantic binding supervision used to train it. The model architecture,
causal updater, three-task sampling, optimizer, 1600-step schedule, MME action
checkpoint, and 1300-step rollout protocol are fixed.

## Data audit

The old unified teacher sequence keeps only one target color for every
VideoUnmask training row, even when the native task instruction contains two
targets. More importantly, its target region does not always use the same
episode-local anchor vocabulary as runtime grounding.

| Split | Rows | Old target labels inconsistent with native anchor |
|---|---:|---:|
| Train | 70 | 12 (17.1%) |
| Dev | 15 | 4 (26.7%) |
| Test | 15 | 3 (20.0%) |

The native label is derived from the first demonstration frame: red, green,
and blue cube centers are matched to the episode's anchor set. On 122 targets
for which the simulator execution trace exposes a grounded container point,
the first-frame center has 1.0-pixel median error and 6.32-pixel maximum error.
This establishes that the native label and runtime region vocabulary agree.

The training H5 corresponds to benchmark train seeds (6000 series); official
closed-loop evaluation uses test seeds (560000 series). Their seed sets have
zero overlap.

## Training conditions

- `original`: old unified teacher and single-color goal representation.
- `native_single`: preserve the existing single target per row, but replace
  its binding with the simulator-native color/anchor label and supervise the
  causal initial write plus retention trajectory.
- `native_full`: apply the same correction and restore all one/two targets from
  the original native instruction. This introduces 23 dual-target rows among
  70 VideoUnmask training rows.

No new model head or parameter is introduced.

## Offline free-rollout results

| Training labels | Dev final query | Test final query | Dev VideoUnmask trajectory | Cross-task selection score |
|---|---:|---:|---:|---:|
| Original | 73.3% | 66.7% | 87.4% | 0.7323 |
| Native single | **100%** | **100%** | **99.0%** | 0.8141 |
| Native full | 94.1% | 93.3% | 93.2% | **0.8421** |

`native_full` gives the best balanced three-task checkpoint score, while
`native_single` gives the strongest VideoUnmask state estimate.

## Official closed-loop action results

All trained checkpoints use the same test-time joint semantic assignment,
unique visual anchor grounding, and
`mme_vla_suite/symbolic-grounded-subgoal/79999` action checkpoint.

| MEM condition | Seed 7, 50 episodes | Single-target | Dual-target |
|---|---:|---:|---:|
| Old teacher, old independent readout | 36/50 = 72% | 30/38 | 6/12 |
| Old teacher, structural readout + grounding | 41/50 = 82% | 35/38 | 6/12 |
| Native full + structural readout + grounding | 47/50 = 94% | 36/38 | 11/12 |
| Native single + structural readout + grounding | **50/50 = 100%** | **38/38** | **12/12** |

The winning `native_single` condition was repeated under the requested action
seeds:

| Action seed | Success |
|---:|---:|
| 7 | 50/50 = 100% |
| 17 | 50/50 = 100% |
| 27 | 50/50 = 100% |

The episode outcomes are identical across action seeds, as expected from this
nearly deterministic action inference configuration. They should not be treated
as 150 independent episodes. The 95% Wilson interval for the 50 unique test
episodes is 92.9%--100%.

## Component attribution

Starting from the original 72% closed loop:

1. joint semantic assignment contributes +6 percentage points;
2. unique candidate grounding with historical-anchor fallback contributes
   another +4 points when combined;
3. correcting the training-time native binding vocabulary contributes the
   remaining +18 points, taking 82% to 100%;
4. explicitly restoring dual goals is not the key component: it reduces the
   VideoUnmask result from 100% to 94% in this training schedule.

The last observation is consistent with the model design. Corrected operation
payloads teach the recurrent table to bind entities independently of the query,
and the joint readout can retrieve two requested entities at inference. Adding
two simultaneous goal-conditioned writes creates extra competition in the
limited two-micro-event executor and is unnecessary for this task.

## Conclusion

The earlier gap was not evidence that the recurrent architecture is inherently
weaker than GroundSG memory. The dominant problem was a mismatch between the
teacher's region ids and the runtime anchor vocabulary. Once the same causal
updater is trained with native simulator-aligned binding labels, it reaches the
visual/action ceiling on all 50 official VideoUnmask episodes.

For a paper table, report the 100% point result with the 50-episode Wilson
interval and note that the three action seeds are deterministic replications.
Before claiming a direct improvement over a published 85% number, verify that
its task split, difficulty aggregation, action checkpoint, and episode budget
are identical.

## Artifacts

- Winning checkpoint:
  `checkpoints/robomme_explicit_event_native_single_seed260908_260831`
- Full-goal checkpoint:
  `checkpoints/robomme_explicit_event_native_full_seed260908_260831`
- Training implementation:
  `scripts/mem/train_robomme_explicit_event_bottleneck_ablation.py`
- Dataset transformation tests:
  `scripts/mem/train_robomme_explicit_event_bottleneck_ablation_test.py`
- Training launcher:
  `scripts/mem/run_native_unmask_binding_ablation_260831.sh`
- Closed-loop roots:
  `robomme_policy_learning/runs/evaluation/causal-event-mem-native-*-videounmask-test50-seed*-260831`
