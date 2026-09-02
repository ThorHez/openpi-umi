# Real-cup Qwen3-VL full-context-first probe (2026-08-26)

## Question

Can Qwen3-VL first learn the complete reveal-to-decision observation without event-window cutting, before later being adapted to local recurrent windows?

## Protocol

- Initialization: `checkpoints/qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375`.
- Same episode-disjoint split as the local experiment: 80 train, 20 validation.
- Input: 36 uniformly sampled chronological frames from label time 0 through 240.
- The clip includes the initial reveal and all three exchanges, but excludes the post-decision grasp/action suffix.
- No event boundary is supplied to Qwen.
- Each episode produces five independent single-fact questions over the same full clip:
  - initial ball cup;
  - first exchange pair;
  - second exchange pair;
  - third exchange pair;
  - final ball cup.
- Each answer contains only one fact. In particular, the third-exchange target contains no first/second GT event tokens, and the final target contains no GT move prefix.

Training contains 400 rows: 80 initial, 80 first-swap, 80 second-swap, 80 third-swap, and 80 final queries. Validation contains 100 rows with the same balance.

The LoRA was continued for 200 steps on six A100 GPUs, global batch 6, learning rate `1e-5`, with checkpoints every 50 steps.

## Held-out generation results

| Checkpoint | Initial | Swap average | Swap 1 | Swap 2 | Swap 3 | All 3 swaps | Direct final | Recurrent final | All 5 facts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original step-375 | 50% | 35.0% | 40% | 30% | 35% | 5% | **35%** | 40% | 0% |
| Full step 50 | 65% | 40.0% | 30% | 55% | 35% | 5% | 30% | **45%** | 0% |
| Full step 100 | **95%** | 45.0% | 45% | 50% | 40% | 5% | 30% | 40% | 0% |
| Full step 150 | 90% | **51.7%** | **65%** | **50%** | **40%** | **15%** | 25% | **45%** | **5%** |
| Full step 200 | 90% | 50.0% | **65%** | 45% | **40%** | 10% | 30% | 40% | **5%** |

All checkpoints produce schema-valid JSON on 100/100 questions.

## Interpretation

Full-context SFT clearly learns the visually easy early fact: initial-cup accuracy reaches 95%. It also improves the first exchange to 65%. However, performance degrades with temporal distance: the third exchange remains at 40%, direct final position does not improve over the 35% baseline, and only 3/20 validation episodes have all three swaps correct at the best checkpoint.

This is not explained by target imbalance. Each ordinal has 20 validation queries with roughly balanced pair classes. At step 150, the third-exchange predictor emits the left-right pair on 16/20 episodes, showing temporal confusion/collapse rather than insufficient examples of one target class.

Compared with the local-first experiment:

- full-context step 150: 51.7% individual swap, 15% all-three exact;
- local balanced step 60: 80.0% individual swap, 55% all-three exact.

Thus full-context-only training is not yet a strong long-horizon teacher. Thirty-six frames provide enough information to recognize the scene and early events, but attention over the full clip does not reliably bind an ordinal query to the corresponding later exchange or compose the three transitions into the final state.

## Conclusion and next curriculum step

The hypothesis is only partially supported: full context is useful for real-domain grounding and initial-state perception, but by itself does not teach stable three-event memory.

The best checkpoint for a subsequent crop curriculum is step 150:

`checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826/checkpoint-000150`

The controlled next comparison should initialize from this checkpoint and train on progressively cropped 24-frame then 12-frame event clips, retaining 20% full-context replay. It should be compared against the existing local-first initialization from checkpoint-375. The key question is whether full-context pretraining improves local convergence or held-out exact three-event accuracy beyond the current 55%; the present full-context metrics alone do not establish that benefit.

## Artifacts

- Builder: `scripts/mem/build_real_cup_qwen3vl_full_context_manifest.py`
- Contract: `src/openpi/tasks/shellgame/real_cup_qwen3vl_sft_contract.py`
- Evaluator: `scripts/mem/eval_real_cup_qwen3vl_full_context.py`
- Data summary: `artifacts/real_cup_qwen3vl_full_context36_sft_v1_260826/summary.json`
- Best result: `evaluation/shellgame/real_cup_qwen3vl_full_context_v1/step150.summary.json`
- Training directory: `checkpoints/qwen3vl_real_cup_full_context36_lora_v1_260826/`

