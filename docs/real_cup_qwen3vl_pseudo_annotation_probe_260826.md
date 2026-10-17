# Real-cup Qwen3-VL intermediate-event pseudo-annotation probe (2026-08-26)

## Question

Can the existing ShellGame Qwen3-VL LoRA infer cup-swap events from a real-world cup demonstration when the replay buffer has no intermediate swap labels, so that its JSON predictions can replace manual event annotation for recurrent-memory training?

## Data audit

Source archive:

`data/cup_replay_buffer.zip`

Extracted buffer:

`data/cup_replay_buffer/replay_buffer.zarr`

- 100 episodes, 49,326 frames, about 20 Hz.
- RGB: `camera0_rgb`, float32 `[0, 1]`, shape `[49326, 224, 224, 3]`.
- Other arrays: action, robot state, timestamp, and episode boundaries.
- The archive contains no task instruction, cup identity, swap boundary, swap pair, or explicit final-ball-slot field.
- The red ball is visually exposed near the beginning and end. The evaluation therefore detects only these visible endpoints from RGB; endpoints are never passed to Qwen.

Selected episodes 0--9 were exported as uint8 NPZ, MP4, and contact sheets under:

`artifacts/cup_replay_real_qwen_probe/`

## Model and contract

- Base: `/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct`
- ShellGame LoRA: `checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375`
- Inference environment: `/data1/conda_envs/qwen3vl_shellgame/bin/python`
- Output contract: one compact JSON object containing either a swap pair, `no_event`, or `incomplete_event`.
- Qwen receives only chronological RGB frames and the same generic ShellGame event prompt. It does not receive the initial/final detected ball slot or intermediate ground truth.

## Protocol

The primary probe uses a sliding window over each real demonstration:

- 20 input frames per window.
- Source stride 4 at 20 Hz, giving a 3.8-second temporal span and an effective 5 Hz Qwen clip.
- Window starts every 10 source frames (0.5 seconds).
- 10 held-out real episodes and 469 windows in total.

To test whether a missed temporal scale caused failure, episodes 0--2 were also evaluated with 10-frame windows spanning approximately 0.9, 1.8, and 3.6 seconds, plus a 20-frame/3.8-second setting. Episode 1 additionally received two manually selected candidate complete-action intervals, `[170, 270]` and `[290, 350]`.

Endpoint consistency is only a weak episode-level check. A red-ball detector searches the first/last 80 source frames, allowing for the fact that the demonstrator may reveal the ball before the exact last frame. It does not establish that an intermediate event sequence is correct.

## Results

| Probe | Episodes | Windows | Temporal span | `swap` | `incomplete_event` | `no_event` | Valid JSON |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 frames, stride 2 | 3 | 407 | 0.9 s | 0 | 116 | 291 | 100% |
| 10 frames, stride 4 | 3 | 397 | 1.8 s | 0 | 185 | 212 | 100% |
| 10 frames, stride 8 | 3 | 375 | 3.6 s | 0 | 268 | 107 | 100% |
| 20 frames, stride 4 | 3 | 187 | 3.8 s | 0 | 143 | 44 | 100% |
| Main confirmation | 10 | 469 | 3.8 s | **0** | **363** | **106** | **100%** |

The corrected endpoint detector is valid on all 10 confirmation episodes. Because Qwen emitted no swap event, its recurrent rollout simply preserves the initial slot and matches the final slot on 3/10 episodes. This is exactly the 3/10 no-change endpoint baseline and therefore provides no evidence of event tracking.

### Oracle-boundary diagnostic

For the two manually selected episode-1 intervals:

- Under the normal contract, both were classified as `incomplete_event`.
- When the prompt forcibly disallowed abstention, both intervals returned the identical pair `screen_left_cup <-> screen_right_cup`.
- The forced predictions do not produce an endpoint-consistent rollout.

This shows that increasing the window span or perfectly centering a candidate action is insufficient. The forced pair is a guess rather than a calibrated intermediate label.

## Conclusion

The current experiment **rejects the zero-shot hypothesis**: the simulation-fine-tuned ShellGame LoRA cannot presently generate usable intermediate swap JSON for this real-cup replay buffer.

The model has learned the response schema well (100% valid JSON) and abstains rather than hallucinating, but it has not transferred the visual event semantics. The main failure is a simulation-to-real and event-contract domain gap: fisheye imagery, human hands, moving cups with printed IDs, longer/variable actions, occlusion, and different motion statistics. It is not primarily a JSON-format problem or a too-short-window problem.

Therefore these predictions must not yet be used as ground truth to train recurrent MEM; doing so would teach the student to preserve memory through almost every real swap.

## Smallest defensible next experiment

The useful annotation-reduction claim should be tested as a **few-shot real-domain adaptation** experiment, rather than as zero-shot transfer:

1. Manually annotate swap start/end and pair on a small seed set (for example 8--10 episodes or roughly 20--40 completed swaps). Keep at least 10 separately annotated episodes as a validation set.
2. Continue the current LoRA with a mixed replay sampler, initially 70% real event clips and 30% synthetic ShellGame clips, preserving `no_event` and partial-action negatives.
3. Compare: current zero-shot LoRA, few-shot real-adapted LoRA, and endpoint-only supervision without intermediate pseudo-labels.
4. Report event precision/recall/F1, pair accuracy on matched events, exact event-sequence accuracy, endpoint consistency, and human annotation minutes.
5. Only after event precision and recall are acceptable, generate JSON for the remaining unlabeled episodes and test whether MEM trained with those pseudo-labels approaches the fully annotated upper bound.

This would support the intended claim if a small event-labeled seed lets Qwen label the much larger remainder and produces a downstream MEM gain. It avoids claiming that final position alone verifies an otherwise unobserved event path.

## Reproduction artifacts

- Exporter: `scripts/mem/export_cup_replay_buffer_episodes.py`
- Qwen sliding-window evaluator: `scripts/mem/eval_real_cup_qwen3vl_events.py`
- Main outputs: `evaluation/shellgame/real_cup_qwen3vl_step375_probe/confirm10_frames20_stride4.jsonl`
- Main corrected summary: `evaluation/shellgame/real_cup_qwen3vl_step375_probe/confirm10_frames20_stride4.summary.json`
- Oracle-range normal output: `evaluation/shellgame/real_cup_qwen3vl_step375_probe/ep1_oracle_ranges_normal.jsonl`
- Oracle-range forced output: `evaluation/shellgame/real_cup_qwen3vl_step375_probe/ep1_oracle_ranges_forced.jsonl`

