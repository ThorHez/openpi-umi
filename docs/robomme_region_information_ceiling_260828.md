# RoboMME region-information ceiling (2026-08-28)

## Question

Are the available processed RoboMME data fields sufficient to determine the
semantic region required by `VideoUnmask`, `VideoUnmaskSwap`, and
`VideoPlaceOrder`, or is the recurrent MEM failing because essential region
information is absent?

## Ceiling method

The probe uses a deterministic causal symbolic accumulator. It does **not**
read `state_targets` while making predictions.

Its prediction inputs are only:

- task and goal color/ordinal;
- causal event type and entity;
- event-local `region_a` and `region_b`.

For Unmask tasks it maintains `color -> region`. A visible/covered event writes
the entity's region, and a swap event exchanges every matching region. For
PlaceOrder it appends each demonstrated placement to an ordered region table
and applies later region swaps to all stored entries. The requested color or
ordinal is read only after the complete causal event stream has been consumed.

Final GT state is used only by the scorer. Thus the result tests whether the
event annotation contract contains enough causal information; it does not test
whether a visual network can infer those events from RGB.

## Locked-test results

| Input to accumulator | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | Overall query accuracy |
|---|---:|---:|---:|---:|
| Full causal region events | 100.0% | 100.0% | 100.0% | 100.0% (56/56) |
| Hold state; ignore swap region updates | 100.0% | 50.0% | 93.3% | 75.0% (42/56) |
| Candidate-count random | 33.3% | 29.2% | 41.1% | 33.5% |

Episode-exact accuracy with full events is also 100% on all 45 test episodes.
The complete region-state trajectory, not only the final answer, is 100% exact.
The same full-event ceiling is 100% on train and development splits.

## Interpretation

The processed training information is sufficient to recover the required
region exactly. Missing simulator fields or missing final-region labels are not
the primary explanation for the current MEM result.

The ablation localizes the important information:

- `VideoUnmask` needs a reliable initial entity-to-region write and subsequent
  hold under occlusion. It has no relevant swap update in this contract.
- `VideoUnmaskSwap` critically needs both the initial entity-to-region binding
  and the two regions participating in every swap. Ignoring swaps reduces test
  query accuracy from 100% to 50%.
- Most `VideoPlaceOrder` test queries remain at their demonstrated region, so
  ignoring swaps still gives 93.3%; nevertheless the full transition fields
  resolve the remaining moved-target case and reach 100%.

Compared with the frozen recurrent MEM plus fixed readout (40.0%, 34.6%, and
53.3% test accuracy), this ceiling shows a large learnability/interface gap.
In particular, the Place model is far below even the 93.3% hold-only baseline,
so its main problem is not subtle swap reasoning: it is failing to preserve the
initial ordered placement table.

## What this does and does not prove

This probe proves that the **processed privileged/event supervision** is
sufficient. It does not yet prove that every `region_a/region_b` can be inferred
from the student's 12-frame RGB/proprio input. A separate full-context visual
ceiling is required to test observation sufficiency.

Therefore the next training change should not add more state fields. It should
explicitly supervise two operations:

1. initial entity/ordinal-to-anchor binding;
2. sparse `swap(region_a, region_b)` updates with an identity/hold constraint on
   all unaffected entries.

Only if a strong full-context visual event predictor cannot recover
`entity/region_a/region_b` should the dataset be judged visually insufficient.

## Artifacts

- Probe: `scripts/mem/probe_robomme_region_information_ceiling.py`
- Unit test: `scripts/mem/probe_robomme_region_information_ceiling_test.py`
- Metrics: `checkpoints/robomme_region_information_ceiling_symbolic_260828/result.json`

