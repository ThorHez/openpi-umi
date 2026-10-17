# pi0.5 ShellGame baseline v1

This is the memory-free control for ShellGame experiments. It fine-tunes the
released `pi05_base` flow-matching policy on the latest clean ShellGame
demonstration set and intentionally exposes only the current observation.

## Contract

- Config: `pi05_shellgame_baseline_v1`
- Initialization: `gs://openpi-assets/checkpoints/pi05_base/params`
- Data: `data/shellgame_static_phase_instruction_dataset2`
  (4,470 successful episodes, 692,850 frames)
- Observation: current wrist RGB + current third-person RGB, proprioceptive
  state, and phase instruction; no history or memory tokens
- Action: raw robosuite OSC command, 16-step chunks, 7 real dimensions padded
  to the pretrained 32-dimensional action head
- Loss: flow-matching loss on the first 7 action dimensions only
- Optimization: full fine-tuning, global batch 32 (two or four FSDP GPUs), AdamW, cosine
  `5e-5 -> 1e-5`, 1,000 warmup steps, EMA 0.999, seed 42
- Checkpoints: every 1,000 steps, retain only the latest checkpoint
- Cache: the derived Hugging Face Arrow index is placed under
  `/dev/shm/pi05_shellgame_baseline_v1` to avoid duplicating 64 GiB of image
  data on the nearly-full shared data volume

## Train

The launcher refuses to overwrite an existing run and checks that at least
100 GiB remains on the shared volume before starting:

```bash
scripts/run_pi05_shellgame_baseline_v1.sh
```

For a short smoke run, use a new experiment name:

```bash
STEPS=2 EXP_NAME=smoke_b32_2gpu scripts/run_pi05_shellgame_baseline_v1.sh
```

Resume the same run on physical GPUs 4--7 from its latest complete checkpoint:

```bash
GPU_IDS=4,5,6,7 FSDP_DEVICES=4 RESUME=1 \
  EXP_NAME=full_ft_seed42_b32_2gpu_260827_retry1 \
  scripts/run_pi05_shellgame_baseline_v1.sh
```

`STEPS` remains the final global-step target when resuming; it is not the
number of additional steps. Orbax restores the latest complete checkpoint and
reshards the training state to the requested FSDP mesh.

## Serve and evaluate

After a checkpoint is available, start the standard policy server:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_shellgame_baseline_v1 \
  --policy.dir=checkpoints/pi05_shellgame_baseline_v1/<exp>/<step>
```

Run ShellGame in the baseline's single-frame/raw-action contract:

```bash
.venv/bin/python examples/shellgame/main.py \
  --policy-input-mode single_frame \
  --action-mode raw7 \
  --action-dim 7 \
  --action-horizon 16 \
  --phase-instructions \
  --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
  --max-policy-steps 300 \
  --num-trials 100 \
  --seed 260827 \
  --video-out-path evaluation/shellgame/pi05_baseline_v1_<step>_seed260827
```

Report both cup-selection accuracy and physical lift success. Because the
policy has no temporal state, cup selection after the ball is hidden is the
primary memory-free reference point; it should not be interpreted as a test of
the recurrent memory mechanism.

## Recorded no-MEM reference

The validated 100-episode V10 action-only/current-only control, including the
relaxed correct-selection-plus-contact metric, is documented in
[`shellgame_v10_action_only_no_mem_baseline_100ep_260828.md`](shellgame_v10_action_only_no_mem_baseline_100ep_260828.md).
Its primary relaxed end-to-end result is 35/100 (35%), statistically consistent
with the three-cup chance level. Do not substitute the complete V10 policy
result that retains its native 60-frame tracker.
