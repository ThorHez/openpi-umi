# RoboMME shared anchor-pointer minimal probe (2026-08-28)

## Question

Can the low semantic-region accuracy be fixed by keeping the existing recurrent
MEM frozen and replacing its fixed region classifier with one shared,
episode-local anchor pointer?

This is a readout-only diagnostic. It does not change the recurrent updater,
the 12-frame chunking, or the soft memory gate.

## Setup

- Tasks: `VideoUnmask`, `VideoUnmaskSwap`, and `VideoPlaceOrder`.
- Input memory: final recurrent memory from the existing independently trained
  single-task checkpoints.
- Candidate anchors: exact episode-local region centers recovered from the H5
  metadata/detectors used by the existing GT event pipeline.
- Anchor representation: bilinearly sampled token from the first-frame pooled
  4x4 SigLIP grid plus normalized `(y, x)` coordinates.
- Query: recurrent memory plus unified task, goal-color, and ordinal embeddings.
- Output: masked dot-product scores over at most four episode-local anchors.
- No task-specific output heads; one pointer is shared by all three tasks.
- Anchor order is randomly permuted during training to prevent learning a fixed
  region-index prior.
- Training queries: 246; development queries: 52; locked test queries: 56.
- Optimizer: AdamW, 2,000 steps, batch size 32, learning rate `3e-4`.
- Checkpoint selection: maximize `min(task dev accuracy)`, breaking ties by mean
  task development accuracy.

The selected checkpoint is step 350. The final step reached 100% training-batch
accuracy but did not improve development accuracy, which is direct evidence of
overfitting.

## Results

| Split / task | Existing fixed readout | Shared anchor pointer | Delta |
|---|---:|---:|---:|
| Train overall | 45.9% | 74.8% | +28.9 pp |
| Dev overall | 38.5% | 48.1% | +9.6 pp |
| Test overall | 41.1% | 39.3% | -1.8 pp |
| Test VideoUnmask | 40.0% | 33.3% | -6.7 pp |
| Test VideoUnmaskSwap | 34.6% | 42.3% | +7.7 pp |
| Test VideoPlaceOrder | 53.3% | 40.0% | -13.3 pp |

The candidate-count-aware random test baseline is 33.5%. Thus the pointer is
only 5.8 percentage points above random on the locked test set and is worse than
the existing fixed readout overall.

## Conclusion

The readout-only hypothesis is rejected for this implementation. A shared
episode-local pointer can fit the training episodes and gives a small
development gain, but the gain does not transfer to the locked test set. It
therefore does not provide a reliable route to 80% region accuracy.

The most likely reasons are upstream of the final classifier:

1. The three frozen recurrent MEM checkpoints were trained independently, so
   their latent coordinate systems are not aligned. A shared projection is
   being asked to decode three different latent bases after training has ended.
2. The current MEM objective does not explicitly bind an entity/state token to
   an episode-local visual anchor. The pointer is trying to recover this binding
   post hoc.
3. A single token sampled from a 4x4 first-frame grid is a coarse anchor
   representation, especially near object boundaries.
4. There are only 246 training queries for a roughly 1.3 MB pointer checkpoint;
   the train/dev divergence confirms insufficient statistical support.

This result should not be sent into a long closed-loop action evaluation. The
previous bridge experiment already showed that, with the official frozen action
checkpoint, coordinate-correct region predictions and action success matched on
all 30 evaluated episodes. The remaining failure is therefore the semantic
region predictor, not evidence against the action controller.

## Recommended next decisive probe

Before a larger end-to-end run, perform a two-row information-sufficiency
diagnostic with the same anchors and labels:

1. Train a low-capacity pointer separately on the unified privileged/teacher
   state. This tests whether the anchor interface and labels themselves can
   generalize.
2. Train the same-capacity pointer on each frozen student MEM separately. This
   removes the independently trained latent-space mismatch.

If the teacher-state probe is high while the per-task student probes remain
low, the recurrent updater must be retrained with an explicit per-step
entity-to-anchor alignment loss. If even the teacher-state probe is low, improve
the anchor construction (higher-resolution ROI features) before touching the
updater.

## Artifacts

- Model: `src/openpi/tasks/robomme/anchor_pointer_readout.py`
- Unit test: `src/openpi/tasks/robomme/anchor_pointer_readout_test.py`
- Training script: `scripts/mem/train_robomme_anchor_pointer_probe.py`
- Selected parameters: `checkpoints/robomme_anchor_pointer_probe_v1_260828/pointer_params.msgpack`
- Full metrics/history: `checkpoints/robomme_anchor_pointer_probe_v1_260828/result.json`

