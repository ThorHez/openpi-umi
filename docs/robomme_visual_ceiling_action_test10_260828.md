# RoboMME Visual Ceiling + Frozen MME Action Upper Bound

Date: 2026-08-28

## Question

If the demonstration can be decoded into the correct target region by a strong
visual/privileged ceiling, is the frozen official
`symbolic-grounded-subgoal/79999` action policy sufficient to complete the
three compatible RoboMME tasks?

## Protocol

- Official RoboMME `test` split
- First 10 episodes per task
- Tasks: `VideoUnmask`, `VideoUnmaskSwap`, `VideoPlaceOrder`
- Action seed: 7
- Maximum rollout length: 1300 steps
- Frozen action checkpoint: `symbolic-grounded-subgoal/79999`
- No action-policy fine-tuning
- Simulator SimpleSG oracle supplies only the current action phase
- Visual ceiling supplies target identity/region; execution GroundSG is used
  only for post-hoc audit

This is therefore a memory/grounding-to-action upper-bound experiment, not a
fully autonomous end-to-end score. In particular, phase transitions are still
oracle.

## Visual ceiling contract

- `VideoUnmask`: bind cube color to the episode-local container region from the
  first demonstration RGB frame.
- `VideoUnmaskSwap`: initialize the color binding from RGB and apply all swap
  pairs recovered from full-demonstration RGB motion. Variable demonstration
  lengths are supported.
- `VideoPlaceOrder`: use demonstration-only subgoal boundaries and grounded
  placement anchors to recover the placement order. If a cube occludes the
  target at the start of a drop segment, use the next demonstration pickup
  coordinate for the same resting cube, matching the validated offline ceiling.
  For hard episodes, target relocation is decided from full-context RGB patch
  motion.

The PlaceOrder arm is not a pure-RGB model: its segmentation and placement
anchor coordinates are privileged ceiling information. Its purpose is to test
the downstream action upper bound and establish a distillation target.

## Closed-loop result

| Semantic source | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | Total |
|---|---:|---:|---:|---:|
| Visual ceiling | 10/10 | 10/10 | 10/10 | **30/30 (100%)** |
| Oracle semantic region (previous run) | 10/10 | 10/10 | 10/10 | 30/30 (100%) |
| Existing recurrent MEM (previous run) | 4/10 | 1/10 | 1/10 | 6/30 (20%) |

The first run was interrupted after 24 completed episodes by a transient Vulkan
renderer initialization error. The evaluator was restarted and resumed from
`progress.json`; it skipped the completed 24 episodes and evaluated only the
remaining six. No episode result was duplicated or replaced.

## Trace audit

- Successful episodes: 30/30
- Semantic-action decision rows: 299
- Mean predicted-vs-audit GroundSG point error: 3.41 px
- Maximum point error: 8.49 px
- Episodes whose semantic point stayed within 8 px: 29/30
- All 30 episodes remained on the correct actionable object/target and
  succeeded; the sole 8.49 px case was still inside the correct container.
- The 10 PlaceOrder episodes include 6 easy, 2 medium, and 2 hard episodes; all
  succeeded.

## Conclusion

Under the same 30-episode diagnostic protocol, the visual ceiling fully closes
the gap to the oracle-region upper bound. The frozen MME action checkpoint and
the semantic-region-to-GroundSG bridge are sufficient when target memory is
correct. The current 20% recurrent-MEM closed-loop score is therefore dominated
by memory-state/region errors, not by a weak action backbone.

The immediate research target should be distilling the ceiling's target-state
trajectory into the recurrent MEM, especially swap updates and ordered
placement state. Further action fine-tuning is not justified by this upper-bound
test. A later fully autonomous evaluation must replace the SimpleSG phase oracle
and should use multiple action seeds.

## Artifacts

Implementation:

- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/subgoal_predictor.py`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/examples/robomme/region_grounding.py`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/scripts/run_region_grounding_three_task.sh`

Final result:

- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/visual-ceiling-action-test10-seed7-260828/grounded-action-region-bridge-visual_ceiling/ckpt79999/seed7/region_grounding_visual_ceiling/log.json`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/visual-ceiling-action-test10-seed7-260828/grounded-action-region-bridge-visual_ceiling/ckpt79999/seed7/region_grounding_visual_ceiling/region_grounding_traces/`
- `/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/runs/evaluation/visual-ceiling-action-test10-seed7-260828/grounded-action-region-bridge-visual_ceiling/ckpt79999/seed7/region_grounding_visual_ceiling/videos/`

Validation:

```text
3 passed in 0.03s
```
