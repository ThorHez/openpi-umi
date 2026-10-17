# RoboMME Semantic-Region Grounding + Frozen GroundSG Action Pilot

Date: 2026-08-28

## Question

Can a recurrent MEM state represented as target identity/history plus a local
`region_i` be converted into the coordinate-bearing GroundSG interface expected
by the official `symbolic-grounded-subgoal/79999` action checkpoint?

The experiment covers:

- `VideoUnmask`
- `VideoUnmaskSwap`
- `VideoPlaceOrder`

## Implemented interface

The bridge uses one shared contract rather than task-specific prediction heads:

1. Build an episode-local spatial anchor vocabulary from the demonstration.
2. Read one semantic `region_i` from either an oracle semantic upper bound or
   the existing recurrent MEM readout.
3. Select the corresponding anchor.
4. Snap the anchor once, at execution start, to the nearest current RGB object
   center.
5. Format the result as RoboMME GroundSG text using `<y, x>` coordinate order.
6. Feed it to the frozen `symbolic-grounded-subgoal/79999` action policy.

The action phase is supplied by the simulator SimpleSG oracle in both arms. It
does not supply the target region. This isolates the memory/grounding interface
from phase-transition errors.

For the `oracle_region` upper bound, the GroundSG oracle coordinate is used only
to identify the discrete semantic region. The original oracle coordinate is
never passed to the action policy; the policy receives the bridge's RGB-snapped
coordinate.

## Offline grounding audit

Protocol: first 20 H5 episodes per task, correct semantic region supplied to the
bridge, and predicted current-image point compared with GroundSG point.

| Task | Within 8 px | Mean error | P95 error |
|---|---:|---:|---:|
| VideoUnmask | 20/20 (100%) | 3.42 px | 5.10 px |
| VideoUnmaskSwap | 20/20 (100%) | 3.39 px | 4.55 px |
| VideoPlaceOrder | 20/20 (100%) | 0.55 px | 0.95 px |

Artifact:

`robomme_policy_learning/runs/evaluation/region-grounding-h5-oracle-region/audit20.json`

## Closed-loop protocol

- Dataset split: official `test`
- Episodes: first 10 per task
- Action seed: 7
- Maximum steps: 1300
- Frozen action checkpoint: `symbolic-grounded-subgoal/79999`
- Same RGB grounding bridge in both arms
- No action checkpoint fine-tuning

## Results

| Semantic source | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | Total |
|---|---:|---:|---:|---:|
| Oracle semantic region | 10/10 | 10/10 | 10/10 | 30/30 (100%) |
| Existing recurrent MEM | 4/10 | 1/10 | 1/10 | 6/30 (20%) |

For the recurrent arm, semantic coordinate accuracy (`<= 8 px`) was:

| Task | Semantic coordinate correct | Action success | Episode-level agreement |
|---|---:|---:|---:|
| VideoUnmask | 4/10 | 4/10 | 10/10 |
| VideoUnmaskSwap | 1/10 | 1/10 | 10/10 |
| VideoPlaceOrder | 1/10 | 1/10 | 10/10 |
| Total | 6/30 | 6/30 | 30/30 |

Thus every episode with a correct recurrent semantic coordinate succeeded, and
every episode with a wrong semantic coordinate failed in this pilot.

## Interpretation

The bridge and the official action model are not the current bottleneck. When
the semantic region is correct, the complete path reaches 100% on these 30
episodes. The recurrent action success rate is numerically identical to its
semantic coordinate correctness.

The immediate optimization target should therefore be final semantic region
exactness, especially on `VideoUnmaskSwap` and `VideoPlaceOrder`. Further action
fine-tuning or coordinate-regression tuning is unlikely to improve success when
the MEM selects the wrong region.

This experiment also explains why SimpleSG alone previously performed poorly:
SimpleSG omitted the spatial binding required by the GroundSG action policy.
The new bridge restores that binding without requiring a task-specific action
head.

## Limitations

- The action phase still comes from oracle SimpleSG. A deployable version must
  infer phase from action/proprioception or retain the action policy's phase
  controller.
- The local anchor vocabulary is built with deterministic visual heuristics,
  not a learned grounding network.
- The closed-loop result uses one action seed and 10 episodes per task. It is a
  diagnostic pilot, not the final paper-scale evaluation.
- The existing recurrent checkpoints are single-task checkpoints trained on the
  local split; their weak official-test region accuracy is now directly exposed.

## Reproduction

Implementation:

- `robomme_policy_learning/examples/robomme/region_grounding.py`
- `robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- `robomme_policy_learning/scripts/eval_region_grounding_h5.py`
- `robomme_policy_learning/scripts/run_region_grounding_three_task.sh`

Closed-loop result root:

`robomme_policy_learning/runs/evaluation/region-grounding-three-task-test10-seed7-260828`

Validated with:

```text
3 passed in 0.03s
```
