# RoboMME recurrent MEM vs visual ceiling: four-way causal ablation (2026-08-29)

## Question

Why does the anchor-conditioned recurrent MEM fail to absorb the 100% visual
ceiling transition process?  Is the dominant cause event routing, operation
payload, recurrent state update algebra, or soft gating?

## Controlled protocol

All rows use the same selected 1,200-step checkpoint and the same locked test
episodes.  No weights are retrained.  The two oracle rows use oracle data only
as diagnostic interventions.

| Row | Event routing | Payload | State update | State conditioning |
|---|---|---|---|---|
| A | Oracle | Learned | Hard deterministic | GT previous state for local logits |
| B | Learned | Oracle on real GT updates | Hard deterministic | GT previous state for local logits |
| C | Learned | Learned | Hard deterministic | Free rollout |
| D | Learned | Learned | Original soft gate | Free rollout |

For B, a false-positive event on a GT hold slot has no meaningful oracle
payload, so it retains the learned payload.  This deliberately preserves the
damage caused by false routing.  An oracle-event + oracle-payload sanity row
must reconstruct every trajectory exactly.

## Locked-test result

| Variant | Mean final query | Transition exact | Hold exact | All-state exact |
|---|---:|---:|---:|---:|
| A: oracle event, learned payload | 14.4% | 15.7% | 60.6% | 54.6% |
| B: learned event, oracle payload | 31.2% | 29.9% | 33.9% | 33.3% |
| C: full learned, hard updater | 9.6% | 9.4% | 22.9% | 21.1% |
| D: full learned, soft updater | 9.6% | 11.0% | 22.7% | 21.2% |
| Oracle event + oracle payload sanity | 100.0% | 100.0% | 100.0% | 100.0% |

### Payload diagnosis

With oracle event routing, learned test payload quality is:

| Payload component | Accuracy |
|---|---:|
| Write entity | 48.8% |
| Write region | 53.7% |
| Joint write `(entity, region)` | 19.8% |
| Swap pair | 75.0% |
| All-update payload | 30.2% |

Oracle routing therefore does not rescue the MEM: only 30.2% of true updates
receive a completely correct learned payload, and final query accuracy remains
14.4%.  The write path is the largest payload bottleneck.

### Event-routing diagnosis

With GT previous state and oracle payload, the learned routing has:

| Routing metric | Value |
|---|---:|
| Aggregate event-type accuracy | 60.8% |
| True-update exact-type recall | 60.4% |
| True-update exact-type precision | 11.0% |
| GT-hold false-positive rate | 39.1% |
| Episodes with an entirely correct event sequence | 0/45 |

There are 149 real test updates, but the local learned router emits 816
updates.  Its aggregate accuracy hides a severe false-update problem caused by
the large number of hold slots.  Even perfect payload cannot overcome this;
row B reaches only 31.2% final query accuracy.

Under free rollout, routing becomes slightly more conservative but remains far
from deployable: hold false-positive rate is 28.8%, exact-type update recall is
53.7%, and precision is 12.8%.

### Hard versus soft update

Rows C and D are effectively identical:

- mean final query: 9.6% versus 9.6%;
- all-state exact: 21.1% versus 21.2%;
- transition exact: 9.4% versus 11.0%.

Therefore soft interpolation is not the dominant cause of the current broad
failure.  A hard updater remains preferable for a future deployable design,
but replacing soft with hard commit cannot repair incorrect event and payload
predictions.

### Exposure/recurrent-state diagnosis

An additional local-full-hard row uses GT previous state to generate every
step's logits, then replays all learned operations from the episode start.  It
reaches 9.2% test final accuracy, versus 9.6% for true free hard rollout.

This shows that conditioning the operation heads on a perfect previous table
does not rescue the accumulated sequence.  The earlier 63.7% teacher-forced
final result was high because every step directly restarted from the GT
previous table; it measured isolated one-step transitions and masked prior
operation errors.  The dominant drift is accumulated routing/payload error,
not an inability of the deterministic table algebra to carry a correct state.

## Train-to-test generalization

| Variant | Train final | Dev final | Test final |
|---|---:|---:|---:|
| A: oracle event, learned payload | 28.0% | 21.6% | 14.4% |
| B: learned event, oracle payload | 44.8% | 38.9% | 31.2% |
| C: full learned, hard updater | 36.7% | 23.1% | 9.6% |
| D: full learned, soft updater | 35.8% | 23.0% | 9.6% |

The consistent train/dev/test degradation confirms that the 4x4 local visual
operation translator also overfits the small corpus.

## Conclusion

The recurrent symbolic table and its write/swap algebra are not the present
bottleneck: oracle operations recover 100%, while hard and soft learned
updaters are indistinguishable.  The ceiling knowledge is lost before the
state update in two places:

1. **Payload perception is the largest isolated bottleneck**, especially the
   joint write entity/region prediction (19.8% on test).
2. **Event routing is also structurally unusable**, with 39.1% hold false
   positives and only 11.0% update precision under GT-state local inference.

The next structural experiment should therefore target the visual operation
translator, not table losses or gate temperature: higher-resolution region
features plus a causal transient event state that accumulates motion evidence
and emits one conservative completion commit.  Boundary and payload must be
evaluated separately before reconnecting the persistent recurrent table.

## Limitations

- Test contains 45 episodes and 56 final queries across three tasks.
- The hard row reuses a checkpoint trained with the soft event gate.  It proves
  that inference-time soft mixing is not the current dominant failure, but it
  does not rule out benefits from training a new conservative hard-commit
  architecture end to end.
- A/B are privileged causal diagnostics, not deployable configurations.

## Artifacts

- Evaluator: `scripts/mem/eval_robomme_transition_causal_ablation.py`
- Hard-commit switch: `src/openpi/tasks/robomme/anchor_conditioned_transition_memory.py`
- Result: `checkpoints/robomme_transition_causal_ablation_seed260901_260829/result.json`
