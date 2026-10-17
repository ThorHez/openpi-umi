# PickXTimes online phase-alignment optimization (2026-08-27)

## Goal

Replace the previous final-state-dominated PickXTimes objective with a training and
selection objective that matches the online action-policy requirement:

1. do not switch from pick to place before the simulator confirms the grasp phase;
2. do not increment `completed_count` before a placement completes;
3. preserve memory on no-change chunks;
4. retain final-count performance only after transition and hold accuracy.

The model remains a fixed 12-frame recurrent student with the shared canonical
readout. No task-specific head, sliding window, or explicit event detector is added.

## Implemented changes

### Closed-loop phase cache

`robomme_policy_learning/examples/robomme/subgoal_predictor.py` can now cache
12-frame SigLIP patch-token chunks during PickXTimes rollouts. At the final frame of
each chunk it decodes the simulator's online simple subgoal into the canonical state:

```text
(completed_count, holding, ready_to_press, done)
```

Two collection modes are supported:

- oracle subgoal controls the action policy, providing successful controller states;
- recurrent MEM controls the action policy while the oracle is recorded only as a
  training label, providing hard off-policy negatives.

The oracle label is never used to construct the recurrent MEM policy input.

### Online phase dataset and losses

`openpi-umi/scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py` now
supports `--online-phase-dir` and mixed offline/online batches. Online state labels
are paired with canonical teacher-memory prototypes having the same goal color,
required count, and symbolic state.

The added objective contains:

- weighted readout CE on oracle-confirmed transition chunks;
- weighted readout CE on oracle-confirmed no-change/hold chunks;
- memory consistency on online no-change chunks;
- the existing canonical memory distillation and offline state loss.

With `--online-aligned-selection`, checkpoint selection changes from
`final -> transition -> sequence -> state` to
`transition -> no-change -> state -> final`.

## Verification completed

- Python compilation passed for the modified collector and trainer.
- Unified fixed-chunk student tests: 5/5 passed.
- Synthetic cache construction verified all batch shapes, state masks, and canonical
  memory lookup.
- A real 1-step mixed offline/online training run completed forward, backward,
  optimizer update, dev evaluation, and checkpoint save.
- In that pipeline smoke test, all three new loss terms were present in the metrics.
  The synthetic all-zero feature cache was used only for code-path verification and
  must not be used as experiment data.

## Blocker in this run

The 11 GB official `symbolic-grounded-subgoal/79999` checkpoint was started three
times. Every attempt remained in Orbax OCDBT restore without reaching the websocket
listen stage; the final attempt was allowed 15 minutes. No controller rollout was
therefore generated, and no optimized checkpoint or post-optimization success rate
is reported. Previous oracle 5/5 artifacts remain intact and show that the checkpoint
itself had loaded successfully in an earlier run.

## Continuation commands

Start the official action policy:

```bash
cd /data2/hzl_workspace_for_pi_mem/robomme_policy_learning
PYTHONPATH=/data2/hzl_workspace_for_pi_mem/robomme_policy_learning/src \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv/bin/python \
  scripts/serve_policy.py --seed=7 --port=18011 \
  policy:checkpoint \
  --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 \
  --policy.config=mme_vla_suite
```

After the server is listening, collect five oracle-policy train episodes:

```bash
cd /data2/hzl_workspace_for_pi_mem/robomme_policy_learning
PYTHONPATH=packages/openpi-client/src:third_party/robomme_benchmark/src:examples/robomme \
ROBOMME_SIM_SITE_PACKAGES=/data2/hzl_workspace_for_pi_mem/robomme/.venv/lib/python3.11/site-packages \
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv/bin/python examples/robomme/eval.py \
  --args.host=127.0.0.1 --args.port=18011 --args.model-seed=7 \
  --args.policy-name=groundsg-pick-phase-cache-oracle-train5 \
  --args.model-ckpt-id=79999 --args.only-tasks=PickXtimes \
  --args.dataset-split=train --args.episode-ids=0,1,2,3,4 \
  --args.use-oracle --args.subgoal-type=simple_subgoal \
  --args.pick-phase-cache-dir=/data2/hzl_workspace_for_pi_mem/openpi-umi/artifacts/pickxtimes_groundsg_online_phase_train_260827
```

Collect five current-MEM-policy train episodes with oracle labels for audit/training:

```bash
cd /data2/hzl_workspace_for_pi_mem/robomme_policy_learning
PYTHONPATH=packages/openpi-client/src:third_party/robomme_benchmark/src:examples/robomme \
ROBOMME_SIM_SITE_PACKAGES=/data2/hzl_workspace_for_pi_mem/robomme/.venv/lib/python3.11/site-packages \
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv/bin/python examples/robomme/eval.py \
  --args.host=127.0.0.1 --args.port=18011 --args.model-seed=7 \
  --args.policy-name=groundsg-pick-phase-cache-mem-train5 \
  --args.model-ckpt-id=79999 --args.only-tasks=PickXtimes \
  --args.dataset-split=train --args.episode-ids=5,6,7,8,9 \
  --args.use-recurrent-mem --args.subgoal-type=simple_subgoal \
  --args.pick-phase-cache-dir=/data2/hzl_workspace_for_pi_mem/openpi-umi/artifacts/pickxtimes_groundsg_online_phase_train_260827
```

Train the first aligned pilot from the previous best Pick checkpoint:

```bash
cd /data2/hzl_workspace_for_pi_mem/openpi-umi
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 PYTHONPATH=src:. \
.venv/bin/python scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py \
  --task=pickxtimes_local_event --steps=150 --batch-size=4 \
  --learning-rate=1e-5 --end-learning-rate=3e-6 --warmup-steps=10 \
  --change-state-weight=4 --final-state-weight=1 \
  --write-gate --write-gate-bias=-2 \
  --resume-checkpoint=checkpoints/robomme_single_task_pick_equal_exposure_seed260827_260827/best/params \
  --online-phase-dir=artifacts/pickxtimes_groundsg_online_phase_train_260827 \
  --online-phase-fraction=0.5 --online-phase-weight=2 \
  --online-hold-weight=2 --online-transition-weight=4 \
  --online-hold-readout-loss-weight=1 \
  --online-transition-readout-loss-weight=1 \
  --online-hold-keep-loss-weight=0.1 --online-aligned-selection \
  --eval-every=25 --save-every=50 --skip-test --seed=260840 \
  --output-dir=checkpoints/robomme_pick_online_phase_aligned_seed260840_260827
```

The first evaluation gate is improved dev transition and no-change accuracy. Only a
checkpoint passing that gate should replace the MEM path in the existing strict
PickXTimes val episodes 0--4 closed-loop command.

## Completed validation results

The earlier restore blocker was traced to sandboxed commands seeing CPU only while
GPU0/1 were almost full. The experiment was completed with the action server on GPU4
and the MEM/simulator on GPU3.

### Collected data

- 10 train-split controller rollouts.
- 404 causal 12-frame chunks.
- 24 simulator-confirmed state transitions.
- Oracle-policy subset: 3/5 success; the two failures supplied real no-progress
  negatives.
- Current-MEM-policy subset: 0/5 success and reproduced premature holding/count
  transitions.

### Training ablations

| Recipe | Best step | Dev transition | Dev hold | Dev state | Dev final |
|---|---:|---:|---:|---:|---:|
| Original Pick checkpoint | 75 | 43.4% | -- | 41.6% | 100.0% |
| 50% online, strong phase loss | 125 | 10.1% | 37.6% | 33.6% | 0.0% |
| 25% online, conservative phase loss | 75 | 25.3% | 39.3% | 37.3% | 86.7% |

The strong recipe collapsed toward preserving the initial state. The conservative
recipe retained most final performance but remained worse than the original on
offline dev transition/state metrics.

### Strict closed-loop comparison on the same val episodes 0--4

| MEM | Success | Online subgoal exact | First-mismatch median step |
|---|---:|---:|---:|
| Original | 0/5 (0%) | 38/103 (36.9%) | 96 |
| Conservative phase-aligned | 1/5 (20%) | 62/172 (36.0%) | 160 |

The successful episode was val episode 1, the single-repetition PickXTimes task. The
other four episodes failed. Phase alignment delayed the first erroneous transition,
but often changed the failure mode from premature `holding/place` to insufficient
`holding/place` switching followed by premature count accumulation. Therefore this
is a small closed-loop improvement, not a robust solution, and the checkpoint should
not replace the original model as the general PickXTimes MEM.

The data contain only 24 transition chunks versus 380 hold chunks. The next version
should balance at the transition-window level rather than sampling entire rollout
episodes; otherwise no-change evidence dominates even when transition rows are
weighted.

Artifacts:

- Online cache: `artifacts/pickxtimes_groundsg_online_phase_train_260827/`
- Strong run: `checkpoints/robomme_pick_online_phase_aligned_seed260840_260827/`
- Conservative run: `checkpoints/robomme_pick_online_phase_aligned_conservative_seed260841_260827/`
- Conservative closed loop: `robomme_policy_learning/runs/evaluation/groundsg-pick-online-phase-conservative-val5/ckpt79999/seed7/recurrent_mem/`
