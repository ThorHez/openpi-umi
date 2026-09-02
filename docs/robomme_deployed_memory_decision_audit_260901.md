# RoboMME deployment-matched memory-decision audit (2026-09-01)

## Purpose

This audit measures the task-relevant memory decision supplied to the action
policy by the same deployed pipelines used for the paper's current closed-loop
results. It replaces the earlier oracle-window student table, whose model and
protocol were not matched to the final systems.

## Protocol

- Split: official RoboMME test episodes.
- PickXTimes: score all 150 deployed rollouts from action seeds 7, 17, and 27,
  because the recurrent memory state evolves with the closed-loop trajectory.
- VideoUnmask, VideoUnmaskSwap, and VideoPlaceOrder: score the 50 unique test
  demonstrations. Their memory output is computed from the demonstration and is
  deterministic across the three action seeds, so repeating it would not add
  independent memory predictions.
- Static-task ground truth is obtained after inference. The official episode is
  rebuilt and reset with its native demonstration; simulator segmentation then
  locates the post-demonstration target actor, and its centroid is mapped to the
  anchors stored in the deployed trace. These labels are used only for scoring
  and are never provided to memory or policy.
- An episode is memory-correct only when every binding requested by its task
  instruction is correct. PickXTimes uses exact final dynamic control state.

## Results

| Task | Audited decision | Correct / N | Memory accuracy | E2E success |
|---|---|---:|---:|---:|
| PickXTimes | Final control state | 143 / 150 | 95.3% | 82.0% |
| VideoUnmask | All requested color-to-region bindings | 50 / 50 | 100.0% | 90.0% |
| VideoUnmaskSwap | All requested post-swap bindings | 49 / 50 | 98.0% | 92.7% |
| VideoPlaceOrder | Queried ordinal-to-region decision | 44 / 50 | 88.0% | 86.0% |

Additional PickXTimes diagnostics:

- Exact state at individual control queries: 4,933 / 5,282 = 93.4%.
- Exact entire control-query sequence: 29 / 150 = 19.3%.
- Closed-loop successes: 123 / 150 = 82.0%.

The full-sequence criterion is much stricter than the final-decision criterion:
one transient mismatch makes the whole rollout incorrect, even if the state
later recovers before the decisive action. It therefore should not be compared
directly with end-to-end success.

For VideoUnmaskSwap, the target-level result is 71 / 73 = 97.3%; its episode
score is 49 / 50 because a multi-target episode is counted wrong if either
requested binding is wrong.

As a sanity check on the post-hoc region labels, the maximum target-centroid to
assigned-anchor distance is 5.89 px for VideoUnmask, 6.29 px for
VideoUnmaskSwap, and 0.54 px for VideoPlaceOrder. The minimum gap to the
second-nearest anchor is respectively 21.15, 13.85, and 18.68 px, so none of
the scored labels is a near-tie between two runtime anchors.

## Reproduction

From the repository root:

```bash
python3 robomme_policy_learning/scripts/analyze_deployed_memory_decisions.py
```

The machine-readable result is written to:

`robomme_policy_learning/runs/evaluation/deployed-memory-decision-audit-260901/summary.json`

The evaluation-only static-task ground-truth shards are stored in that same
directory. Their extractor is:

`robomme_policy_learning/scripts/extract_official_test_target_regions.py`
