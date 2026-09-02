# PickXTimes: GroundSG → recurrent MEM functional distillation pilot (2026-08-27)

## Question

Can the recurrent MEM acquire action-relevant knowledge from the official RoboMME
GroundSG policy, without replacing the existing canonical state/count supervision?

## Method

The official GroundSG checkpoint is frozen.  At a fixed H5 observation, it is
queried after reset with:

1. the oracle grounded subgoal, producing reference action chunk `a*`;
2. counterfactual simple subgoals for Pick, Place, and Press, producing `a_i`.

The functional teacher target is

`q_i = softmax(-RMSE(a_i, a*) / 0.05)`.

Targets are aligned to recurrent states at the end of non-overlapping 12-frame
chunks.  Samples cover every GT state-change chunk and one chunk before/after it.
The MEM readout's canonical `holding`, `ready_to_press`, and `done` distributions
are projected to Pick/Place/Press probabilities.  Training minimizes cross entropy
to `q` in addition to the original memory-trajectory, state, and count losses.
Thus GroundSG teaches action semantics, while count remains GT supervised.

## Data and teacher audit

- Train: 6 episodes, 114 aligned recurrent states.
- Dev: 6 disjoint episodes, 102 aligned recurrent states.
- Teacher top-1 vs dataset phase: 72.8% train, 71.6% dev.
- For confidence `max(q) >= 0.6`: 23 train states and 21 dev states.
- High-confidence teacher top-1 vs dataset phase: 91.3% train, 100% dev.
- The high-confidence subset is strongly Pick-heavy: train GT phases are 20 Pick / 3
  Place / 0 Press; dev phases are 19 Pick / 1 Place / 1 Press.

The confidence audit is important: GroundSG often gives nearly identical action
chunks to Place and Press while the arm is still far from the relevant object.
Treating those ambiguous states as hard phase supervision is not justified.

## Controlled experiments

All runs start from
`robomme_single_task_pick_equal_exposure_seed260827_260827/best/params` and use the
same seed, 50% sampling from the six teacher-labelled train episodes, 75 updates,
and the original GT losses.

| Run | GroundSG supervision | Dev action agreement | Final state | Transition state | All-state |
|---|---:|---:|---:|---:|---:|
| Step 0 baseline, all targets | none | 36.3% | 100.0% | 43.4% | 41.6% |
| GT-only control, step 75, all targets | weight 0 | 35.3% | 93.3% | 38.4% | 40.2% |
| All targets, step 75 | weight 0.1 | 35.3% | 40.0% | 29.3% | 41.6% |
| Step 0 baseline, confidence ≥ 0.6 | none | 52.4% | 100.0% | 43.4% | 41.6% |
| GT-only control, step 75, confidence ≥ 0.6 | weight 0 | 52.4% | 93.3% | 38.4% | 40.3% |
| High-confidence distillation, step 50 | weight 0.02 | **61.9%** | 93.3% | 33.3% | 40.8% |
| High-confidence distillation, step 75 | weight 0.02 | 57.1% | 93.3% | 33.3% | 40.0% |

The unfiltered weight-0.1 run reduced soft-target CE from 1.78 to 1.48 but did not
improve top-1 and severely damaged final state.  It demonstrates optimization of
an ambiguous target, not useful knowledge transfer.

The high-confidence run produces a narrow positive result: compared with the
matched GT-only control, peak action agreement increases from 52.4% to 61.9% while
final state remains 93.3%.  However, transition-state accuracy is lower, and the
high-confidence set is dominated by Pick.  It therefore shows that GroundSG
contains a distillable action-functional signal, especially for Pick, but does not
yet demonstrate balanced Pick/Place/Press transfer or preservation of the recurrent
state trajectory.

## Decision

This pilot partially supports the claim **“GroundSG can provide auxiliary
action-functional supervision to MEM.”**  It does not yet support either balanced
phase transfer or the stronger claim **“the distilled MEM is action-ready.”**  No
closed-loop action rollout is promoted from this run, because the transition-state
regression exceeds the preservation tolerance.

The next safe optimization is a constrained update: compute the GroundSG gradient
only on high-confidence states and project/remove its component that conflicts with
the canonical transition-state gradient (or update only a small action-facing MEM
adapter).  Acceptance should require:

- high-confidence action agreement > 60%;
- final-state accuracy >= 93%;
- transition-state accuracy no worse than the matched control by more than 2 points.

It should also report per-phase action agreement and include enough confident Place
and Press samples; otherwise an aggregate improvement can be driven almost entirely
by Pick.

## Gradient-decoupling follow-up

Two gradient-projection variants were subsequently trained with the same seed,
confidence threshold, action-loss weight, sampling, and 75-step schedule:

1. **Global PCGrad:** if GroundSG and the complete canonical GT gradient conflict,
   remove the conflicting GroundSG component.
2. **Transition-aware PCGrad:** apply global PCGrad, then additionally remove the
   GroundSG component that conflicts with transition-state readout CE.

| Run / checkpoint | Action agreement | Final state | Transition state | All-state |
|---|---:|---:|---:|---:|
| Matched GT-only control, step 75 | 52.4% | 93.3% | 38.4% | 40.3% |
| Vanilla high-confidence, step 50 | **61.9%** | 93.3% | 33.3% | 40.8% |
| Global PCGrad, step 25 | **61.9%** | 93.3% | 32.3% | 40.6% |
| Global PCGrad, step 50 | 52.4% | 93.3% | 33.3% | **41.3%** |
| Transition-aware PCGrad, step 25 | **61.9%** | 80.0% | 31.3% | 40.5% |
| Transition-aware PCGrad, step 50 | 52.4% | 93.3% | **34.3%** | 40.9% |

On the periodically logged train updates, global projection triggered on 6/8 and
transition projection on 7/8 sampled updates.  Typical action-gradient norms were
30–72 versus canonical-gradient norms of 0.75–4.0; even after multiplying the
action gradient by 0.02, its update scale can be comparable with the canonical
gradient.  The conflict is therefore measurable rather than hypothetical.

Neither projection variant passes the acceptance gate.  Transition-aware PCGrad
recovers only one transition point at step 50, while losing the action improvement.
It can also reduce final-state exactness early in training.  A local first-order
constraint on a sampled train batch does not guarantee preservation of the
held-out recurrent exact-state trajectory, especially with a Pick-heavy action
teacher subset and a discontinuous exact-match metric.

This result changes the next recommendation: do not keep increasing PCGrad
complexity inside the canonical updater.  Freeze the validated recurrent MEM and
train a **task-agnostic action-facing residual adapter** on its memory tokens.  The
adapter can be shared across RoboMME tasks and act only on the copy of MEM consumed
by the action expert; the canonical memory and readout remain unchanged by
construction.  This preserves the no-task-specific-head design while cleanly
separating state memory from action-functional alignment.

## Artifacts

- Label builder: `scripts/mem/build_pickxtimes_groundsg_action_teacher_cache.py`
- Teacher cache: `artifacts/robomme_pickxtimes_groundsg_action_teacher_v1_260827/`
- Trainer: `scripts/mem/train_robomme_four_task_fixed_chunk_distillation.py`
- GT-only control: `checkpoints/pickxtimes_groundsg_action_control_v1_260827/`
- Unfiltered run: `checkpoints/pickxtimes_groundsg_action_distill_v1_260827/`
- High-confidence run: `checkpoints/pickxtimes_groundsg_action_distill_conf06_w002_v1_260827/`
- Global gradient projection: `checkpoints/pickxtimes_groundsg_action_pcgrad_conf06_w002_v1_260827/`
- Transition-aware projection: `checkpoints/pickxtimes_groundsg_action_pcgrad_transition_conf06_w002_v1_260827/`

## Environment note

The requested interpreter
`/data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/python` is currently a broken
symlink to `/opt/conda/bin/python3.11`.  This experiment used the verified project
environment `/data2/hzl_workspace_for_pi_mem/openpi-umi/.venv/bin/python`.
