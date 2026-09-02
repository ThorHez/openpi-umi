# PickXTimes / VideoUnmask action-readiness optimization — 2026-08-27

## Scope

This round tested four concrete changes after the first 10-episode action smoke test:

1. PickXTimes on-policy no-progress hard negatives for the recurrent MEM.
2. PickXTimes initial-memory control to separate MEM drift from the action policy.
3. VideoUnmask joint goal-color × target-region balancing.
4. VideoUnmask oracle-gripper diagnosis and a true delta-EEF action target.

All action conclusions below use closed-loop RoboMME simulator rollouts. Teacher-forced action MAE is reported only as a training diagnostic.

## 1. PickXTimes: action versus MEM diagnosis

### Initial-memory control

The action adapter was run for 10 train episodes while keeping the MEM fixed at its initialized goal-conditioned state. Each episode ran for 500 simulator steps.

| Control | Episodes | Task success | First pick |
|---|---:|---:|---:|
| Fixed initial MEM + learned action | 10 | 0/10 | 0/10 |

This result rules out recurrent memory drift as the cause of the first-pick failure. The current Pick action adapter cannot complete even the first primitive when given a stable and correct initial subgoal.

### On-policy no-progress negatives

Ten fixed-initial-memory rollouts were cached as frozen SigLIP patch-token sequences. Episodes 12 and 13 were held out; the other eight were used as no-progress negatives. The target state remains the initial state for every chunk.

Two recipes were tested:

- Mixed negative replay: 50% negative batches, weight 4, 100 update steps.
- Targeted gate/keep: 25% negative batches, weight 2, gate penalty 20, memory-keep penalty 10, 50 update steps, three seeds.

| Model | Pick dev final | Held-out all-state | Held-out final | False done | False completed count |
|---|---:|---:|---:|---:|---:|
| Original | — | 14.29% | 0% | 50% | 100% |
| Mixed-negative best | 80.0–86.7% | 26.19% | 0% | 50% | 100% |
| Gate/keep, dev-selected step 10 | 100% | 15.48% | 0% | 50% | 100% |
| Gate/keep, final step 50 | 93.3% | 16.67% | 0% | 50% | 100% |

The negative replay slightly improves early all-state accuracy, but neither recipe prevents the model from hallucinating `completed_count=3` on both unseen no-progress trajectories. A soft gate of roughly 0.01 still accumulates over 41 recurrent chunks, and the learned gate does not separate event from hold chunks reliably.

Decision: do not replace the current Pick MEM checkpoint with either hard-negative checkpoint. More importantly, do not spend further MEM-only compute before repairing the first-pick action controller.

## 2. VideoUnmask: balanced memory training

The train split was sampled uniformly over the joint `(goal color, target region)` combination. Three seeds were trained and selected on dev; seed 260831 was best.

| Model | Test final exact | Test transition exact | Test sequence exact | Test all-state | Test field |
|---|---:|---:|---:|---:|---:|
| Original single-task MEM | 40.0% | 43.33% | 20.0% | 64.36% | 92.90% |
| Joint-balanced MEM | 40.0% | 43.33% | 20.0% | 64.36% | 93.07% |

The only change is +0.17 percentage points in field accuracy; all task-level exact metrics are identical. On the 10 action-smoke validation episodes, precomputation produced only 2/10 exact target regions versus 3/10 for the original model. The action run later hit a native simulator segmentation fault after three episodes; those three had 0/3 region exact, 1/3 reach, 0/3 grasp, and 0/3 success.

Decision: goal-region frequency imbalance is not the main generalization problem. Keep the original single-task MEM checkpoint.

## 3. VideoUnmask: oracle target and gripper diagnosis

The existing Pi action expert was evaluated with the exact target point and simulator oracle gripper gate. Thus neither MEM region selection nor learned gripper timing can cause failure.

| Target | Gripper | Episodes | Reach | Grasp | Lift | Success |
|---|---|---:|---:|---:|---:|---:|
| Oracle point | Oracle gate | 10 | 8/10 | 0/10 | 0/10 | 0/10 |

Mean minimum target XY distance was 2.07 cm, while mean maximum target lift was only 0.269 cm. The policy reaches the correct XY neighborhood but does not produce a graspable Z/orientation trajectory. This establishes the current primary VideoUnmask bottleneck as action control after reach.

## 4. True delta-EEF ablation

The lightweight action adapter was extended without changing the original recipe:

- Position label: `next_position - current_position`.
- RPY label: shortest wrapped adjacent-frame difference.
- Inference: integrate the predicted delta into the current EEF pose.
- Safety: per-axis position clamp plus shortest-angle rotation clamp.

The saved step-1000 checkpoint achieved 0.15 cm one-step validation position MAE, 0.06° rotation MAE, and 97.2% gripper accuracy. Closed-loop results with exact geometric targets were:

| Action target | Episodes | Success | IK errors | Episodes issuing close | Mean final EEF Z |
|---|---:|---:|---:|---:|---:|
| Absolute EEF baseline | 10 | 0/10 | 2/10 | 7/10 | 0.208 m |
| Delta EEF | 10 | 0/10 | 5/10 | 1/10 | 0.518 m |

The delta model is worse despite excellent one-step validation error. It drifts upward under its own state distribution, rarely enters the close phase, and has more IK failures. Therefore the original jumpy videos cannot be fixed by changing only the coordinate representation.

## Conclusion and next action experiment

Neither PickXTimes nor VideoUnmask is ready for a claimed end-to-end action result:

- Pick: the action controller fails before the first memory update matters.
- VideoUnmask: target XY reach is mostly solved under oracle input, but contact, close, and lift are not.

The next useful optimization is action-side on-policy recovery data with an explicit phase contract, not another MEM loss sweep:

1. Represent action as phase-conditioned residual waypoints: pre-grasp, descend, close-hold, and lift.
2. Collect recovery states from current closed-loop failures, especially high-Z drift and near-target misses.
3. Balance the four phases and train close/lift transitions explicitly.
4. First require at least 8/10 oracle-target grasps; only then reconnect the predicted MEM and measure end-to-end success.

## Artifacts

- Pick no-progress evaluator: `scripts/mem/eval_robomme_pick_no_progress.py`
- Pick no-progress token cache: `artifacts/pickxtimes_onpolicy_no_progress_train10_260827/initial_memory`
- Pick targeted runs: `checkpoints/robomme_single_task_pick_no_progress_gate_seed260837_260827` through seed 260839
- Balanced VideoUnmask run: `checkpoints/robomme_single_task_unmask_goal_region_balanced_seed260831_260827`
- Oracle Pi action result: `evaluation/robomme/videounmask_single_fixed_chunk_mem_action_260827/oracle_point_val10_oracle_gripper.json`
- Delta action run: `evaluation/robomme/videounmask_memory_action_adapter_delta_260827/action_target_state_fixed_crop_delta`
- Paired absolute/delta closed-loop results: `evaluation/robomme/videounmask_memory_action_adapter_delta_260827`

## 5. Phase-waypoint and recovery continuation

The proposed four-phase contract was implemented as `pregrasp -> descend -> close_hold -> lift`.
The action target is the residual from the current EEF pose to the demonstrated endpoint of the
current phase.  Oracle phase and oracle gripper are used only for this action-upper-bound diagnosis.

### Lightweight adapter

The best lightweight phase-waypoint checkpoint obtained the first full success in this line of
experiments: 1/10 validation episodes, with 5/10 IK errors.  This is an improvement over the prior
0/10 controllers, but is far below the 8/10 gate required before connecting learned MEM.

### Pi action expert

Pi was retrained with current front/wrist images, phase-balanced rows, and 16 repeated residual
waypoint commands.  Synthetic recovery augmentation doubled the train set from 6,646 to 13,292
rows.  Each augmented row perturbs current EEF XY by up to 4 cm, Z by up to 6 cm, and orientation,
then recomputes the residual to the unchanged phase waypoint.

| Pi conditioning | Closed-loop preflight | Reach | Grasp | Success | Mean min XY |
|---|---:|---:|---:|---:|---:|
| Image target + recovery | 3 | 0/3 | 0/3 | 0/3 | 16.6 cm |
| World XY + recovery | 3 | 0/3 | 0/3 | 0/3 | 16.0 cm |
| Goal-relative world XY + recovery | 3 | 1/3 | 0/3 | 0/3 | 4.11 cm |
| Goal-relative + explicit phase-goal adapter | 3 | 0/3 | 0/3 | 0/3 | 5.43 cm |

World XY labels are not simulator annotations: they are automatically extracted from the first
demonstrated close/contact waypoint.  The goal-relative adapter exposes continuous XY error to the
action tokens and materially improves target approach.  However, traces show phase-0 commands still
descend to the table before XY alignment, causing contact-induced target motion and IK errors.  A
larger phase-goal adapter made offline validation competitive (best loss 0.0833) but made closed-loop
stability worse (3/3 IK errors), another direct example of teacher-forced loss not predicting action
success.

Decision: do not connect these Pi checkpoints to MEM and do not expand to 10--20 episodes.  The
action gate remains 0/3 grasp under oracle target/phase/gripper.  The most informative retained model
is the simpler goal-relative checkpoint because it reduces mean minimum XY error from about 16 cm to
4.11 cm.  The next action experiment should enforce a high-Z pregrasp safety invariant or collect
actual simulator-rendered recovery rollouts; duplicating images while perturbing EEF state is not a
complete on-policy correction dataset, especially for the wrist camera.

Additional artifacts:

- Recovery dataset: `data/robomme_videounmask_lerobot_pi_phase_waypoint_recovery2x_train_260827`
- World-XY recovery dataset: `data/robomme_videounmask_lerobot_pi_worldxy_phase_waypoint_recovery2x_train_260827`
- Goal-relative checkpoint: `checkpoints/pi0_robomme_videounmask_point_action_260823/worldxy_goalrelative_phase_waypoint_recovery2x_b2_200steps_1gpu_260827/199`
- Goal-relative preflight: `evaluation/robomme/videounmask_pi_worldxy_goalrelative_recovery2x_260827/preflight_val_ep0_1_12_step199.json`
- Phase-goal negative result: `evaluation/robomme/videounmask_pi_worldxy_phasegoal_recovery2x_260827/preflight_val_ep0_1_12_step75.json`
