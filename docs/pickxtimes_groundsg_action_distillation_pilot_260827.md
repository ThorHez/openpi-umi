# PickXTimes GroundSG action-distillation signal pilot (2026-08-27)

## Question

Can the recurrent MEM learn action-relevant knowledge from the official RoboMME
GroundSG checkpoint, even though that checkpoint consumes symbolic GroundSG text
rather than exposing a recurrent memory latent?

## Protocol

- Task: PickXTimes.
- Dataset episode: offline demonstration episode 0, red cube, two repetitions.
- Teacher: official `symbolic-grounded-subgoal/79999` checkpoint.
- Samples: four phase boundaries, with offsets -16, 0, and +16 frames; 12
  observations in total.
- Candidate inputs for every fixed observation:
  - correct online GroundSG;
  - every wrong phase GroundSG observed in the episode;
  - correct simple subgoal without pixel coordinates;
  - a repeated correct GroundSG control.
- Before every inference the teacher policy is reset to seed 7. Thus every candidate
  uses the same diffusion noise and the repeated-correct difference measures the
  deterministic noise floor.
- Metric: RMSE between the complete 20-step teacher action chunks. We also compare
  each candidate with the demonstration joint-action suffix.

## Results

| Metric | Result |
|---|---:|
| Samples | 12 |
| Repeated-correct action RMSE | 0.000 |
| Mean wrong-phase action RMSE to correct | 0.260 |
| Mean nearest-wrong action RMSE to correct | 0.059 |
| Correct simple vs grounded action RMSE | 0.018 |
| Correct GroundSG beats wrong phase against demonstration | 77.8% |

Phase breakdown:

| Correct phase | Samples | Wrong mean RMSE | Wrong minimum RMSE | Simple-vs-grounded RMSE | Demo win |
|---|---:|---:|---:|---:|---:|
| Pick | 4 | 0.333 | 0.079 | 0.005 | 66.7% |
| Place | 6 | 0.233 | 0.057 | 0.031 | 83.3% |
| Press | 2 | 0.194 | 0.026 | 0.007 | 83.3% |

## Interpretation

The official GroundSG action policy provides a clean action-relevance signal. The
repeat control is exactly zero while wrong semantic phases change the action chunk by
roughly 0.26 RMSE on average. Correct GroundSG also produces an action suffix closer
to the demonstration than a wrong phase in 77.8% of pairwise comparisons.

The semantic phase dominates the signal in this episode: removing coordinates while
retaining the correct simple subgoal changes actions by only 0.018 RMSE on average.
Coordinates matter more around place boundaries (0.031 average and 0.153 on one
sample), so grounding should remain a current-frame adapter rather than recurrent
state.

The nearest wrong phase can occasionally be almost action-equivalent. In particular,
different pick ordinals may require the same immediate primitive. Therefore action
distillation cannot replace count/state supervision; it should weight or regularize
the recurrent phase objective.

## Recommended first training objective

Freeze the GroundSG action teacher. For the same observation, noise, and diffusion
timestep, compare the oracle-GroundSG velocity with the velocity conditioned by the
MEM phase representation:

```text
L = L_state
  + lambda_flow * mse(v_mem, stop_gradient(v_groundsg_oracle))
  + lambda_transition * L_transition
```

Keep the existing simulator state/count labels. Start with a small `lambda_flow`
(0.05--0.1), because this pilot shows action equivalence does not uniquely identify
the repetition count.

## Artifacts

- Reproducible evaluator: `scripts/mem/eval_pickxtimes_groundsg_distillation_signal.py`
- Full per-sample result: `evaluation/robomme/pickxtimes_groundsg_distillation_signal_260827.json`
