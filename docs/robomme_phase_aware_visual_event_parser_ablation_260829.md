# RoboMME phase-aware visual event parser ablation (2026-08-29)

## Question

The previous ablations showed that the deterministic semantic table is not the
dominant bottleneck. The learned visual parser instead suffered from spatial
aliasing, weak causal routing, and frequent false writes. This experiment asks:

1. does online 8x8 visual evidence improve operation payloads;
2. can an independent completion head preserve update recall at a very low hold
   false-positive rate;
3. does explicit `idle/moving/settling/complete` supervision add value beyond
   the independent completion head;
4. do these local gains survive deterministic semantic-table replay?

This experiment covers VideoUnmask, VideoUnmaskSwap, and VideoPlaceOrder. It
does not include PickXTimes because Pick uses a count/interaction transition
contract rather than the region write/swap parser evaluated here.

## Data and architecture

- Input is the original online, causal sequence of non-overlapping 12-frame
  chunks. No complete future event window is supplied at inference.
- Each RGB frame is encoded as an 8x8 grid of cell mean, standard deviation,
  and x/y gradient statistics (`64 x 12` per frame).
- The parser samples anchor-conditioned visual tokens and feeds chunk evidence
  into the existing gated causal transient state.
- Two micro-event slots are retained because some 12-frame chunks contain two
  completed events.
- The decomposed parser predicts:
  - four-way transient phase: `idle`, `moving`, `settling`, `complete`;
  - independent completion probability for each micro-event slot;
  - event kind (`write` or `swap`) only for true updates;
  - write entity/region or swap pair payload only for true updates.
- A semantic-table operation is committed only when completion probability
  exceeds a threshold calibrated on dev for hold FPR <= 0.5%.

Privileged phase labels are used only during training. For the next unfinished
event, a chunk is `idle` before its visual start, `moving` before 70% progress,
`settling` afterward, and `complete` when the fixed causal state index advances.

All new rows train for 1,200 steps. Completion-only and phase-aware use the
same seed (260903), samples, initialization, inputs, losses, and checkpoint
selection; their only controlled difference is phase-loss weight 0 versus 0.5.
The selected checkpoint is step 800 for both.

## Locked-test operation results

There are 45 test episodes, 1,908 valid micro-event slots, and 149 true updates.

| Parser | Commit rule | Hold FPR | Update recall | Update precision | Payload on GT updates | Full-update recall |
|---|---|---:|---:|---:|---:|---:|
| Joint 3-class causal RGB8 | argmax | 4.38% | **81.88%** | 60.70% | **79.87%** | **64.43%** |
| Joint 3-class causal RGB8 | dev-calibrated threshold | 0.63% | 48.32% | 86.75% | **79.87%** | 40.27% |
| Independent completion RGB8, no phase loss | dev-calibrated threshold | 0.80% | **71.81%** | 88.43% | 76.51% | 57.72% |
| Independent completion RGB8 + phase loss | dev-calibrated threshold | **0.51%** | 69.80% | **92.04%** | 77.85% | 57.72% |

The ordinary argmax row has the highest local full-update recall, but its 4.38%
hold FPR produces 77 false commits on the test split and is unsuitable for a
persistent memory. Applying the same low-FPR requirement to that joint head
collapses update recall to 48.32%.

The independent completion head is the main structural improvement. At a
similar low FPR, it raises update recall by 23.49 points and full-update recall
by 17.45 points over the calibrated joint head. Completion confidence no
longer has to compete with write/swap class scores in one softmax.

Phase supervision is useful but secondary. Relative to completion-only it:

- reduces test hold FPR from 0.80% to 0.51%;
- raises update precision from 88.43% to 92.04%;
- leaves full-update recall unchanged at 57.72%;
- decreases swap-pair payload from 92.86% to 89.29%, while improving calibrated
  routing balance.

The phase row's last bullet is deliberately retained as a negative result:
phase supervision does not improve every local payload component. Its benefit
appears in conservative temporal routing and replay, not raw payload accuracy.

## Deterministic operation-replay results

| Parser at low-FPR commit | Mean final query | Min-task final | Transition exact | Hold exact | All-state exact |
|---|---:|---:|---:|---:|---:|
| Calibrated joint head | 41.11% | 0.00% | 33.07% | 43.17% | 41.82% |
| Independent completion, no phase | 45.21% | 40.00% | 42.52% | 56.11% | 54.30% |
| Independent completion + phase | **49.66%** | **42.31%** | **43.31%** | **61.31%** | **58.91%** |

Per-task final-query accuracy is:

| Parser | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder |
|---|---:|---:|---:|
| Calibrated joint head | **73.33%** | **50.00%** | 0.00% |
| Independent completion, no phase | 53.33% | 42.31% | 40.00% |
| Independent completion + phase | 53.33% | 42.31% | **53.33%** |

The joint head's mean hides a complete Place failure. Completion factorization
recovers Place to 40%, and phase supervision raises it to 53.33%. Phase also
adds 5.20 points of hold-state exactness and 4.61 points of all-state exactness,
despite identical full-update recall. This is evidence that explicit phase
regularizes *when* the accepted updates occur.

## Relation to earlier visual ablations

Earlier controlled rows established two other effects:

| Controlled intervention | Main test effect |
|---|---|
| Local SigLIP4 -> causal SigLIP4 | update recall 51.0% -> 69.1%; full-update recall 23.5% -> 29.5% |
| Oracle-event RGB4 -> RGB8 | payload 44.3% -> 63.1%; swap pair 67.9% -> 89.3% |

Together with the present rows, the evidence supports four separate claims:

1. causal transient state helps recognize temporally extended events;
2. higher spatial resolution helps identify operation payloads;
3. completion/type factorization is the largest low-FPR routing improvement;
4. phase supervision improves conservative replay and Place balance, but is
   not the primary source of local full-update recall.

The online SigLIP4 to raw-RGB8 comparison is not a pure spatial ablation because
both feature family and resolution change. Spatial attribution therefore comes
from the paired oracle-event RGB4/RGB8 experiment, not that cross-family row.

## What is still missing from the ceiling

The phase-aware row is a meaningful structural improvement, but it has not
learned the ceiling:

- only 69.8% of true updates are committed with the correct type;
- write-region accuracy is 75.2%, so roughly one in four true writes has the
  wrong region even with the correct routing intervention available;
- all-state exactness is 58.9% and mean final query is 49.7%, still far from
  the 100% oracle operation replay;
- phase accuracy is 80.7%, but complete-phase recall is only 48/127 = 37.8%;
- the within-chunk encoder still reduces the 12 frames to mean/early/late
  summaries. It is causal across chunks but is not yet a full ordered
  within-chunk trajectory encoder.

Most importantly, this is an operation-parser diagnostic, not a deployable
end-to-end MEM result. The parser receives anchor coordinates and GT previous
semantic tables when generating local logits. The reported replay accumulates
the predicted operations in a deterministic table, but it does not re-run the
visual parser with its own predicted table at every future step. Consequently,
49.7% final accuracy is an optimistic bridge result; free semantic feedback can
still introduce exposure error.

## Decision

The experiment rejects the hypothesis that the recurrent structure is
intrinsically unable to absorb the ceiling. A small causal recurrent parser can
make useful low-FPR updates when the visual interface and commit objective are
properly factorized. The largest current structural gain is **independent
completion routing**, not the auxiliary phase classifier.

The next justified experiment is therefore not another loss-weight sweep. It
is a free-feedback integration test that replaces GT previous tables with the
parser's own hard semantic table, while preserving the independent completion
head. Add RGB+gripper/EEF-Z only if the Place miss/phase audit confirms that the
visual stream cannot distinguish settling from completion.

## Artifacts

- Parser: `src/openpi/tasks/robomme/causal_visual_operation_parser.py`
- RGB8 online cache builder: `scripts/mem/cache_robomme_fixed_chunk_rgb_grid_features.py`
- Phase-label builder: `scripts/mem/build_robomme_fixed_chunk_phase_labels.py`
- Phase/completion trainer: `scripts/mem/train_robomme_phase_aware_visual_parser.py`
- Fair joint-head threshold evaluator: `scripts/mem/eval_robomme_joint_parser_conservative_commit.py`
- Joint RGB8 checkpoint: `checkpoints/robomme_visual_parser_fixed_causal8_seed260904_260829/result.json`
- Fair calibrated joint result: `checkpoints/robomme_visual_parser_fixed_causal8_seed260904_260829/conservative_commit_eval.json`
- Completion-only result: `checkpoints/robomme_completion_only_rgb8_seed260903_260829/result.json`
- Phase-aware result: `checkpoints/robomme_phase_aware_rgb8_seed260903_260829/result.json`
