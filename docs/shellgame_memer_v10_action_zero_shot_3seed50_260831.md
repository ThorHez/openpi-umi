# MemER ShellGame closed-loop evaluation (3 seeds x 50 episodes)

## Result used in the main table

MemER achieves **70/150 ShellGame end-to-end successes (46.7%)**. The mean and
sample standard deviation across evaluation seeds are **46.7 +/- 4.6 percentage
points**; the pooled Wilson 95% interval is **[38.9%, 54.6%]**.

This is a zero-shot ShellGame transfer, not a ShellGame-trained MemER result.
The released MemER Qwen3-VL-4B grounded-subgoal adapter predicts one subtask
after the 60-frame observation prefix. Its subtask is then supplied to the
frozen V10 action policy with both V10's native tracker memory and the external
semantic-memory branch disabled.

## Per-seed results

| Evaluation seed | MemER grounding | ShellGame end-to-end success |
|---:|---:|---:|
| 260829 | 17/50 (34%) | 26/50 (52%) |
| 261829 | 17/50 (34%) | 22/50 (44%) |
| 262829 | 16/50 (32%) | 22/50 (44%) |
| **Pooled** | **50/150 (33.3%)** | **70/150 (46.7%)** |

## Memory/action decomposition

- All 150 MemER responses contained a parseable grounded coordinate.
- The cached high-level predictions are correct on 30/89 unique episodes
  (33.7%). Repeated episodes across evaluation seeds give 50/150 correct
  groundings (33.3%).
- When MemER grounding is correct, end-to-end success is 26/50 (52%).
- When MemER grounding is wrong, end-to-end success is 44/100 (44%).

Zero-shot MemER grounding is near the three-way chance level. A ShellGame-trained
MemER adapter and a matched grounded-subgoal action policy would be needed for a
stronger port.

## Protocol and audit

- Evaluation seeds: 260829, 261829, 262829.
- Episodes: 50 per seed, 150 rollouts total, using the same locked episode
  lists as the FrameSamp and V10 action baselines.
- Controller: 16-step action chunks, execute 8 steps, maximum 150 policy steps.
- Diffusion noise: deterministic; noise salt equals the evaluation seed.
- MemER checkpoint: `robomme_policy_learning/runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300`.
- Action checkpoint: V10 step 1000, served in `v10_action_no_memory` mode.
- Privileged simulator metadata is used only after inference to reconstruct and
  score the episode; it is not used to construct the MemER subtask.
- Completion audit: 150/150 records, three 50-episode seed results, and the
  `_COMPLETE` marker are present.

## Machine-readable result

`evaluation/shellgame/memer_v10_action_zero_shot_3seed50_260831/summary.json`
