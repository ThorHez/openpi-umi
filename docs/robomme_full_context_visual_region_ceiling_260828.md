# RoboMME full-context visual region ceiling (2026-08-28)

## Question

After establishing that canonical privileged event fields can recover the final
region exactly, does the demonstration observation itself contain enough visual
evidence to recover the same region?

## Two attempted ceilings

### 1. Monolithic learned full-context summarizer

A bidirectional visual summarizer was trained on all non-overlapping frames
from frame 0 through the end of each demonstration. Each 12-frame chunk was
represented by frozen 4x4 SigLIP features containing its temporal mean,
last-minus-first change, and mean absolute frame difference. A bidirectional
temporal transformer then predicted the final episode-local region directly.

This model was leakage-free but was **not** an effective ceiling:

| Split | Query accuracy |
|---|---:|
| Train at dev-selected checkpoint | 33.7% |
| Dev | 44.2% |
| Locked test | 28.6% |

The result is below the 33.5% candidate-count random baseline on test. With
only 246 training queries, asking a newly initialized temporal readout to learn
color/ordinal grounding, spatial region assignment, and long-range motion in
one loss is statistically and structurally underdetermined. This failed model
cannot be used as evidence that RGB is insufficient.

### 2. Decomposed full-context visual-oracle summarizer

The second ceiling exposes the correct task decomposition while still
forbidding final-region leakage:

- **VideoUnmask:** detect colored cubes in the first RGB frame and assign the
  same row-major local region vocabulary used by the data contract.
- **VideoUnmaskSwap:** initialize color-to-region from RGB, identify every swap
  pair from full-window RGB patch motion, and update the symbolic table.
- **VideoPlaceOrder:** initialize the ordinal table from demonstration-only
  subgoal boundaries/anchor coordinates, then decide whether and where the
  queried target moved using RGB motion over the full final swap segment.

The Place motion threshold (`19.0` mean RGB intensity units) was selected on
train/dev only and was not retuned on the fixed test split.

Prediction is forbidden from reading:

- canonical event labels;
- canonical `region_a/region_b`;
- state targets;
- execution-stage GroundSG coordinates.

## Results

| Split / task | Correct | Region accuracy |
|---|---:|---:|
| Train VideoUnmask | 70/70 | 100.0% |
| Train VideoUnmaskSwap | 106/106 | 100.0% |
| Train VideoPlaceOrder | 68/70 | 97.1% |
| **Train overall** | **244/246** | **99.2%** |
| Dev VideoUnmask | 15/15 | 100.0% |
| Dev VideoUnmaskSwap | 22/22 | 100.0% |
| Dev VideoPlaceOrder | 14/15 | 93.3% |
| **Dev overall** | **51/52** | **98.1%** |
| Test VideoUnmask | 15/15 | 100.0% |
| Test VideoUnmaskSwap | 26/26 | 100.0% |
| Test VideoPlaceOrder | 15/15 | 100.0% |
| **Fixed test overall** | **56/56** | **100.0%** |

## Conclusion

The available data contains enough spatial and motion information to determine
the required region. The current recurrent MEM's low test performance is not
best explained by missing final-region information.

The contrast between the two ceilings is the central result:

| System | Fixed-test region accuracy |
|---|---:|
| Symbolic causal event ceiling | 100.0% |
| Decomposed full-context visual-oracle ceiling | 100.0% |
| Monolithic learned full-context summarizer | 28.6% |
| Existing recurrent MEM fixed readout | 41.1% |
| Frozen MEM + shared anchor pointer | 39.3% |

Thus the main problem is **learnability and inductive bias**, not information
absence. A monolithic final-region loss does not teach the model the operations
that the successful visual ceiling explicitly performs.

## Implication for MEM training

The next MEM version should distill the decomposed algorithm rather than only
the final answer:

1. supervise initial `entity/ordinal -> anchor` writes;
2. supervise the visual swap pair independently of the stored target identity;
3. apply an explicit differentiable swap update to the semantic table;
4. enforce identity/hold loss on all unaffected table entries;
5. use final region readout as an evaluation loss, not the only learning
   signal.

This preserves a recurrent implementation while giving it the same structural
advantage as the 100% ceiling.

## Limitations

- PlaceOrder uses demonstration-only temporal boundaries and anchor coordinates
  to initialize the table. It proves that the dataset fields plus RGB motion
  are sufficient, but it is not yet a deployable RGB-only Place detector.
- The Place threshold is a small-corpus heuristic; the test split contains only
  15 episodes and one target-relevant moved-target case.
- The visual-oracle ceiling is a mechanism diagnostic, not a method proposed
  for the final paper comparison.

## Artifacts

- Visual-oracle probe:
  `scripts/mem/probe_robomme_full_context_visual_oracle_ceiling.py`
- Visual-oracle metrics:
  `checkpoints/robomme_full_context_visual_oracle_region_ceiling_260828/result.json`
- Learned summarizer:
  `src/openpi/tasks/robomme/full_context_region_summarizer.py`
- Learned training script:
  `scripts/mem/train_robomme_full_context_visual_region_ceiling.py`
- Learned result:
  `checkpoints/robomme_full_context_visual_region_ceiling_seed260828_260828/result.json`
