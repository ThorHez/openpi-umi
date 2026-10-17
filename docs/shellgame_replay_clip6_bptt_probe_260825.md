# ShellGame 60-frame replay / 10x6 recurrent BPTT probe

## Question

Can a replay item store one complete 60-frame observation episode while the
current student re-unrolls ten non-overlapping six-frame clips, allowing the
final cup-position loss to backpropagate through all earlier memory updates?

The student updater receives only the current six-frame visual clip and the
previous compact memory. It receives no phase, event, relation, or Qwen output.
Qwen is used only indirectly through the existing warm-start checkpoint.

## Implementation

The isolated experiment is implemented in:

`examples/shellgame/train_replay_unrolled_clip6_memory_probe.py`

The validated data contract is:

1. Load only observation frames 0--59.
2. Sample one offset in `[0, 5]` independently for every replay item.
3. Select frames `offset..59` and zero-pad the missing tail.
4. Split the resulting 60 positions into ten disjoint six-frame clips.
5. Recompute the current student's ten recurrent memory states in one graph.
6. Apply final-slot CE to the last state and lower-weight committed-slot CE to
   the first nine states, then backpropagate through the complete unroll.

The self-test verified that the final logit has a nonzero gradient to the first
clip (`first_clip_grad=0.0597391`).

## Controlled runs

Both runs used 4 GPUs, batch size 8, 1000 steps, identical episode-held-out
split (4500 train / 500 validation), peak LR `1e-4`, final loss weight 1.0, and
intermediate loss weight 0.25.

### Warm initialization (GPU 0--3)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_clean60_warm_bptt_1k_260825 \
  --init-mode=warm --steps=1000 --batch-size=8 --num-workers=4 \
  --fsdp-devices=4 --eval-interval=100 --eval-batches=20 \
  --save-interval=500 --keep-period=1000 --overwrite
```

Final held-out metrics:

- final cup accuracy: **36.88%**
- all-clip committed-slot accuracy: **70.87%**
- partial-swap hold accuracy: **75.68%**
- transition-endpoint accuracy: **52.71%**
- final CE: **1.0528**

Checkpoint:

`checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_clean60_warm_bptt_1k_260825/999`

### Scratch replay tracker (GPU 4--7)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_clean60_scratch_bptt_1k_260825 \
  --init-mode=scratch --steps=1000 --batch-size=8 --num-workers=4 \
  --fsdp-devices=4 --eval-interval=100 --eval-batches=20 \
  --save-interval=500 --keep-period=1000 --overwrite
```

Final held-out metrics:

- final cup accuracy: **36.25%**
- all-clip committed-slot accuracy: **42.75%**
- partial-swap hold accuracy: **29.99%**
- transition-endpoint accuracy: **32.50%**
- final CE: **1.0962**
- memory token variance: **7.86e-6** (representation collapse)

Checkpoint:

`checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_clean60_scratch_bptt_1k_260825/999`

## Leakage caught during the experiment

An initial implementation loaded frames 0--64 so that every offset had 60 real
frames. This is invalid: frame 60 starts `robot_approach`, and the expert arm
motion reveals the target cup. That version reached 88--90% final accuracy but
performed poorly when offset 0 excluded all post-observation frames. Those
checkpoints are retained only as a leakage control and must not be cited as a
memory-tracking result.

## Conclusion

The replay/BPTT mechanism is technically valid: the graph spans all ten clips,
and the warm model learns substantial intermediate visual state. However, this
first loss/transition recipe does **not** learn reliable final tracking. Warm
initialization is useful (70.9% versus 42.8% all-clip accuracy), but the old
three-event updater is not stable under ten partial/no-op updates. The scratch
model collapses almost completely.

This motivated the carry-gated follow-up below.

## Carry-gated recurrent update follow-up

### Change

The updater now optionally predicts a scalar continuous gate from the old
memory summary and the current visual-evidence summary:

```text
candidate = recurrent_transform(old_memory, clip_evidence)
gate = sigmoid(MLP([summary(old_memory), summary(clip_evidence)]))
new_memory = old_memory + gate * (candidate - old_memory)
```

The gate output is initialized with zero kernel and bias `-2`, so every step
starts at `sigmoid(-2) = 0.1192`, close to carrying the previous state. It has
no event, phase, relation, or Qwen input.

The objective was also rebalanced by semantic state group rather than token
frequency:

```text
L = L_final + L_transition_endpoint + L_hold
```

Each group has weight 1.0. The final, completed-swap, and non-transition hold
groups therefore contribute equally even though hold clips are more numerous.
The previous loss remains selectable by leaving transition/hold weights at
zero.

The updated self-test verified:

- ten recurrent states and ten scalar gates have the expected shapes;
- the initial mean gate is `0.119203`;
- final logits still backpropagate to clip 0 (`first_clip_grad=0.2305`);
- warm migration copies all 148 existing tracker leaves and initializes only
  the eight newly introduced gate leaves.

### Controlled gated runs

Both runs again used the same 4500/500 episode split, 4 GPUs, batch size 8,
1000 steps, peak LR `1e-4`, and 20 held-out eval batches. The warm run was
initialized from the clean ungated step-999 checkpoint; the scratch run copied
the same frozen Pi base but randomized all 156 tracker leaves.

Warm command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_gated_clean60_warm_bptt_1k_260825 \
  --init-mode=warm \
  --warm-checkpoint=checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_clean60_warm_bptt_1k_260825/999/params \
  --steps=1000 --warmup-steps=100 --peak-lr=1e-4 --decay-lr=1e-5 \
  --final-slot-weight=1 --intermediate-slot-weight=0 \
  --transition-slot-weight=1 --hold-slot-weight=1 \
  --carry-gate --carry-gate-bias=-2 \
  --batch-size=8 --num-workers=4 --fsdp-devices=4 \
  --eval-interval=100 --eval-batches=20 \
  --save-interval=500 --keep-period=1000 --overwrite
```

Scratch used the identical command on GPUs 4--7 with:

```text
--exp-name=replay_clip6_gated_clean60_scratch_bptt_1k_260825
--init-mode=scratch
```

Final held-out results:

| Metric | Ungated warm | Gated warm | Gated scratch |
|---|---:|---:|---:|
| final cup accuracy | 36.88% | **75.00%** | 45.00% |
| all-clip state accuracy | 70.87% | **88.75%** | 75.25% |
| partial-swap hold accuracy | 75.68% | **91.76%** | 77.86% |
| transition-endpoint accuracy | 52.71% | **85.62%** | 66.25% |
| final CE | 1.0528 | **0.5635** | 1.0091 |
| transition gate mean | n/a | 0.1864 | 0.0973 |
| hold gate mean | n/a | 0.1633 | 0.0937 |

Checkpoints:

```text
checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_gated_clean60_warm_bptt_1k_260825/999
checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_gated_clean60_scratch_bptt_1k_260825/999
```

### Updated conclusion

The experiment validates the proposed replay formulation: one clean 60-frame
episode can be replayed as ten non-overlapping six-frame clips, the current
student can unroll memory online, and a single backward pass can train the
complete recurrent chain. A carry-biased learned update plus balanced state
supervision raises warm final accuracy from 36.9% to 75.0%, so the earlier
failure was not a fundamental limitation of non-overlapping clips or full
unrolled BPTT.

Warm initialization remains essential at this budget: scratch improves to
45.0% final accuracy but is far behind warm. The learned gate is selective in
the useful direction (`transition > hold`), but it is a soft distinction and
does not yet yield the near-100% tracking seen in the strongest
stage/relation-supervised tracker. The remaining gap is therefore best viewed
as visual transition initialization/sample efficiency, not a broken replay
graph or insufficient memory capacity.

## Low-LR continuation: step 999 to 1499

To test whether the remaining gap was primarily insufficient optimization, the
gated warm run was resumed for another 500 steps from the full step-999 train
state. The optimizer state was restored, and the learning rate schedule was
replaced with a conservative `1e-5 -> 2e-6` cosine schedule over 1500 total
steps. At resumed step 1000, the actual LR was about `4e-6`; it decayed to
about `2e-6` by the end. The original step-999 checkpoint was explicitly
preserved.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python \
  examples/shellgame/train_replay_unrolled_clip6_memory_probe.py \
  --exp-name=replay_clip6_gated_clean60_warm_bptt_1k_260825 \
  --init-mode=warm --steps=1500 --warmup-steps=100 \
  --peak-lr=1e-5 --decay-lr=2e-6 \
  --final-slot-weight=1 --intermediate-slot-weight=0 \
  --transition-slot-weight=1 --hold-slot-weight=1 \
  --carry-gate --carry-gate-bias=-2 \
  --batch-size=8 --num-workers=4 --fsdp-devices=4 \
  --eval-interval=100 --eval-batches=20 \
  --save-interval=250 --keep-period=999 --resume
```

Validation trajectory:

| Step | final cup | transition endpoint | partial-swap hold | total loss |
|---:|---:|---:|---:|---:|
| 999 / pre-continuation final | 75.00% | 85.62% | 91.76% | 1.0759 |
| 1100 | 77.50% | 85.42% | 90.57% | 1.1138 |
| 1200 | 76.88% | 88.54% | 88.34% | 1.0945 |
| 1300 | 78.12% | 89.17% | 89.07% | **1.0037** |
| 1400 | 76.25% | 88.54% | 91.38% | 1.0214 |
| 1499 / final | **78.12%** | **88.75%** | 88.25% | 1.0314 |

Final checkpoint:

```text
checkpoints/shellgame_replay_unrolled_clip6_memory_probe/replay_clip6_gated_clean60_warm_bptt_1k_260825/1499
```

The extra 500 steps provide only a marginal gain: about +3.1 points on final
and transition accuracy, while partial-swap hold drops by about 3.5 points.
Because each reported evaluation covers 20 batches rather than all 500 held-out
episodes, changes of this size may include sampling noise. The result rules out
training length as the main explanation for the remaining error; simply adding
more steps is unlikely to approach 100% without improving transition evidence
or recurrent-state preservation.

## Full gate causal ablation at step 1499

The continuation checkpoint was evaluated on all 500 held-out episodes under
all six clip offsets, giving exactly 3000 sequences per condition. This removes
the sampling noise of the earlier 20-batch evaluations.

Four inference-only conditions used identical images and parameters:

1. `normal`: unchanged learned gate;
2. `freeze_tail`: allow the third transition, then freeze all later memory
   updates;
3. `oracle_change_mask`: retain the learned gate only for clips overlapping a
   GT swap interval and set it to zero elsewhere;
4. `oracle_open`: force gate 1.0 on swap-overlap clips and zero elsewhere.

Results:

| Condition | final | stage 1 | stage 2 | stage 3 | partial hold |
|---|---:|---:|---:|---:|---:|
| normal | **77.23%** | **97.93%** | **91.60%** | **76.83%** | **89.41%** |
| freeze tail | 76.83% | 97.93% | 91.60% | 76.83% | 89.41% |
| oracle change mask | 47.43% | 50.43% | 45.53% | 47.43% | 62.50% |
| oracle open (gate=1) | 36.70% | 49.73% | 42.10% | 36.70% | 52.70% |

For the normal condition:

- 2305/3000 sequences were correct at the third transition endpoint;
- 91.11% of those remained correct at final, so tail updates flipped 8.89%
  of initially correct third-stage states;
- among the 695 sequences wrong at the third endpoint, tail updates recovered
  31.22%;
- the destructive and corrective effects nearly cancel, leaving normal only
  0.40 points above `freeze_tail`;
- final accuracy across offsets was 74.6%, 79.0%, 79.6%, 77.4%, 78.2%, and
  74.6%, so no single clip boundary explains the failure.

The final confusion matrix (GT rows, predicted columns) was:

```text
[[662, 165,  79],
 [ 67, 895,  58],
 [191, 123, 760]]
```

The center cup is substantially easier (87.75%) than the two edge cups
(73.07% and 70.76%).

### Causal conclusion

Late settle clips are not the primary bottleneck. The main error accumulates
across visual state transitions: endpoint accuracy falls from 97.9% after swap
1 to 91.6% after swap 2 and 76.8% after swap 3. Moreover, GT timing cannot be
used as a hard gate: removing non-swap updates destroys performance, showing
that the model uses apparently unchanged clips to build scene context and
partial evidence. Forcing gate 1.0 is even worse, confirming that the candidate
transition is calibrated for small residual updates rather than complete state
replacement.

The next training change should therefore improve recurrent transition
robustness and edge-cup visual evidence, not add a hard event trigger or freeze
the tail. The raw result is stored at:

```text
evaluation/shellgame/replay_clip6_gate_causal_ablation_step1499_260825.json
```
