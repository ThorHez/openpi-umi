# PickXTimes object-success event ablation (2026-08-31)

## Question

The deployed memory updater must distinguish a gripper close/open command from a
successful manipulation of the instructed-color cube.  This ablation tests
whether adding wrist vision and hand-designed target motion features is enough
to close that gap.

All action smoke tests use the same action policy checkpoint and protocol:

- action checkpoint: `symbolic-grounded-subgoal/79999` through the aligned
  Pick latent-codebook wrapper;
- split: RoboMME PickXTimes test;
- seed: 7;
- 10 episodes, at most 1300 simulator steps per episode;
- no oracle phase correction.

The 10-episode values below are diagnostic smoke tests, **not main-table
results**.  A publishable value still requires the pre-registered 3 seeds x 50
episodes protocol.

## Available observations and labels

The released PickXTimes H5 contains front/wrist RGB-D, camera calibration, EEF
pose, gripper state/command, and subgoal boundaries.  It does not contain target
object pose, contact, or `is_grasping` state.

The simulator implements the actual success predicates in
`robomme/src/robomme/robomme_env/utils/subgoal_evaluate_func.py`:

- successful pick: target object `z > 0.05` and
  `agent.is_grasping(target_object)`;
- successful place: target-object/target XY distance `<= 0.05`, object is no
  longer grasped, object is sufficiently low, and EEF has moved away/up.

Thus finger closure/opening is only an action cue.  It is not the ground-truth
event label.

## Implemented ablations

| Variant | Deployment inputs | Test state exact | Test sequence exact | Test event precision | Test event recall | Action success |
|---|---|---:|---:|---:|---:|---:|
| A: front-only event head | front RGB + gripper/EEF | 99.70% | 86.67% | 97.00% | 100.00% | 1/10 |
| B: front+wrist event head | front+wrist RGB + gripper/EEF | **100.00%** | **100.00%** | 98.98% | 100.00% | 1/10 |
| C: front+wrist+target motion | B + target-color centroid/area + EEF XYZ | 97.17% | 93.33% | 98.96% | 97.94% | 1/10 |
| D: B + observable gate | B + finger/command confirmation + minimum physical interval | same checkpoint as B | same checkpoint as B | n/a | n/a | 1/10 |

Checkpoint directories:

- B: `checkpoints/pickxtimes_event_front_wrist_seed260833_260831`
- C: `checkpoints/pickxtimes_event_front_wrist_motion_seed260834_260831`

Action result directories:

- B: `robomme_policy_learning/runs/evaluation/pick-front-wrist-smoke10-seed7-260831`
- C: `robomme_policy_learning/runs/evaluation/pick-front-wrist-motion-smoke10-seed7-260831`
- D: `robomme_policy_learning/runs/evaluation/pick-front-wrist-observable-gate-smoke10-seed7-260831`

## Diagnostic result

Wrist RGB is useful on successful expert trajectories: it raises full-sequence
exact accuracy from 86.67% to 100%.  However, this gain does not transfer to
closed-loop policy rollouts.  The handcrafted target-motion branch is not a
useful substitute for an object-success label and is mildly harmful even
offline.

The observable gate is active rather than a no-op.  On the 10 test rollouts it:

- rejects 4 candidate events without grasp confirmation;
- rejects 4 physically too-fast transitions;
- increases the minimum committed-event interval from 16 to 96 steps;
- reduces committed pick/place events from 12/8 to 5/3.

Despite suppressing false transitions, it remains at 1/10.  Therefore false
positives are only one part of the problem; the conservative gate also causes
event misses, and finger aperture cannot establish object identity or target
placement.

The apparent contradiction between near-perfect offline event accuracy and
poor action success is explained by two effects:

1. the offline set contains successful expert demonstrations and almost no
   failed-grasp/failed-place hard negatives;
2. per-window accuracy is dominated by no-change windows, while one event error
   is enough to send the recurrent counter and action policy into a different
   phase.

## Recommended next experiment

Collect privileged labels during **training rollouts only** and distill them
into deployable visual success predicates.  Privileged simulator state must not
be used at evaluation time.

1. Record target object pose, target pose, EEF pose, target contact/grasp state,
   and current official subgoal at every simulator step.
2. Construct two debounced labels over several consecutive frames:
   `pick_success = target_lifted AND target_grasped`, and
   `place_success = target_in_region AND released AND stable`.
3. Train separate `pick_success` and `place_success` heads from front+wrist RGB,
   gripper state/command, and EEF motion.  Keep the deterministic ordinal updater
   unchanged.
4. Mix successful demonstrations with failed on-policy attempts and mine
   hard negatives around every gripper close/open transition.
5. Commit a recurrent transition only when the phase-appropriate success
   probability stays above threshold for K consecutive observations.

The necessary ablation should isolate:

- event classification only;
- + explicit success-predicate supervision;
- + failed-rollout hard negatives;
- + temporal debounce;
- privileged-label oracle upper bound.

Only the final non-privileged model should be evaluated with 3 seeds x 50
episodes and reported in the main table.

## Privileged-success distillation follow-up

This follow-up implements the recommended experiment rather than using the
simulator predicates at deployment.

### Training data and contract

- 20 Oracle-action rollouts were collected from the RoboMME **train split**;
  19/20 reached task success.
- Every 12-frame window stores front/wrist SigLIP tokens and deployable proprio,
  plus training-only target pose, goal pose, `is_grasping`, pick-success and
  place-success labels.
- Episodes were split 14/3/3 into internal train/dev/test sets.  No RoboMME test
  episode or test privileged state was used for training or checkpoint choice.
- The student is an independent two-predicate head.  At deployment it receives
  only front/wrist RGB tokens, gripper state/command, EEF Z, and instructed
  target color.  The deterministic ordinal updater is unchanged.

Training cache:
`artifacts/pickxtimes_privileged_success_rollouts_train20_seed7_260831`

Student checkpoint:
`checkpoints/pickxtimes_object_success_privileged_distill_seed260835_260831`

### Internal held-out predicate metrics

The dev-balanced checkpoint is step 100.

| Predicate | Accuracy | Balanced accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| Pick success | 98.48% | 97.22% | 100.00% | 94.44% | 100.00% |
| Place success | 90.15% | 94.04% | 63.89% | 100.00% | 88.07% |

These metrics use the three held-out train-split rollout episodes (132 windows),
not the benchmark test split.

### Closed-loop ablation

All variants below use seed 7, the same 10 benchmark test episodes, the aligned
`symbolic-grounded-subgoal/79999` latent action wrapper, and a 1300-step cap.

| Memory transition source | Success |
|---|---:|
| Original front-only event classifier | 1/10 |
| Front+wrist event classifier | 1/10 |
| Front+wrist + hand-designed target motion | 1/10 |
| Front+wrist + gripper/temporal gate | 1/10 |
| **Distilled object-success predicates, one observation** | **5/10** |
| Distilled object-success predicates, two-observation debounce | 4/10 |
| Distilled predicates + on-policy hard negatives, one observation | 5/10 |
| Oracle latent-codebook upper bound (existing matched test) | 6/10 |
| Official exact SimpleSG upper bound (existing matched test) | 7/10 |

Result directories:

- one observation:
  `robomme_policy_learning/runs/evaluation/pick-object-success-distill-debounce1-smoke10-seed7-260831`
- two observations:
  `robomme_policy_learning/runs/evaluation/pick-object-success-distill-debounce2-smoke10-seed7-260831`

The +40 point gain over every learned-event/gripper-gate variant isolates
explicit object-success supervision as the effective component.  Requiring two
positive action queries delays valid phase transitions by another 16 simulator
steps and reduces success, so additional debounce is not retained.

### On-policy hard-negative ablation

A second train-split collection ran the first distilled student for 20 episodes
(16 successes and 4 failures), producing 575 deployment-distribution windows:
170 pick-positive, 170 place-positive, and 235 windows negative for both
predicates.  These rows were added to training only; the original 3+3 internal
dev/test episodes remained unchanged.

Checkpoint:
`checkpoints/pickxtimes_object_success_onpolicy_hardneg_seed260836_260831`

On the internal held-out split, on-policy negatives reduced place false
positives from 13 to 3 and raised place precision from 63.89% to 88.46%.  Pick
recall decreased from 94.44% to 91.67%.  The formal 10-episode action smoke test
remained 5/10 with exactly the same successful episode ids
`{0, 1, 2, 7, 9}` as the no-hard-negative student.  Thus hard-negative replay
improves predicate calibration but has no measured task-success benefit on this
small slice; it is not the component responsible for the 1/10 to 5/10 gain.

Result directory:
`robomme_policy_learning/runs/evaluation/pick-object-success-hardneg-debounce1-smoke10-seed7-260831`

### Failure attribution

In all five one-observation student failures, the simulator's audit-only simple
subgoal remained at the first pick for the entire rollout.  The student emitted
no state transition and correctly held `[completed=0, holding=0]`: the action
policy never achieved a privileged-positive first grasp.  The failures are
therefore not false MEM transitions.  On successful episodes, predicted
pick/place probabilities are typically strongly separated and the recurrent
count exactly follows the simulator phase through as many as five repetitions.

This 5/10 is a successful mechanism smoke test but is still not a main-table
number.  The next reporting step is the unchanged one-observation model under
the formal 3 seeds x 50 episodes protocol.  Since the matched action upper bound
is only 6--7/10 on this slice, action-backbone failures must be reported or
improved separately from MEM accuracy.

## Aborted 3-seed formal run (2026-09-01)

A formal 3 seeds x 50 episodes run was started with the one-observation
success-distill checkpoint and stopped early by design after the partial rate
was judged too low for the intended paper claim.  All three processes were
terminated cleanly and their completed episode artifacts were retained.

| Seed | Completed | Success | Partial rate |
|---:|---:|---:|---:|
| 7 | 24 | 10 | 41.67% |
| 17 | 23 | 9 | 39.13% |
| 27 | 23 | 10 | 43.48% |
| pooled | 70 | 29 | 41.43% |

These are **censored partial results**, not a valid 3x50 estimate and not a
main-table number.  They are retained only as evidence for stopping this action
evaluation route.  The consistency across the three partial seeds suggests the
low value is not a single-seed accident.  Future work should first improve or
replace the PickXTimes action backbone and re-establish a materially higher
oracle-memory action ceiling before spending compute on another 3x50 MEM run.

Partial result roots:

- `robomme_policy_learning/runs/evaluation/pick-object-success-distill-action-test50-seed7-260831`
- `robomme_policy_learning/runs/evaluation/pick-object-success-distill-action-test50-seed17-260831`
- `robomme_policy_learning/runs/evaluation/pick-object-success-distill-action-test50-seed27-260831`
