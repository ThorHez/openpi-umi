# PickXTimes action vs memory paired diagnosis (2026-09-01)

## Protocol

The same RoboMME test episode ids 0--19 were evaluated with seed 7 and a
1300-step cap under three conditions:

1. **A: official SimpleSG** -- official text SimpleSG and the official
   `symbolic-grounded-subgoal/79999` action policy;
2. **B: Oracle latent** -- simulator phase mapped to the aligned latent
   codebook and the latent-conditioned action wrapper;
3. **C: Student MEM latent** -- deployable distilled object-success MEM mapped
   to the same latent codebook and latent-conditioned action wrapper.

Every simulator step was audited with the issued action, active text/latent
condition, official simulator phase, target/goal/EEF pose, target grasp contact,
and privileged pick/place success predicates.  Audit state was never passed to
the policy or MEM.

## Results

| Condition | First real pick | First real place | Final success |
|---|---:|---:|---:|
| A: official SimpleSG | 12/20 (60%) | 12/20 (60%) | **12/20 (60%)** |
| B: Oracle latent | 9/20 (45%) | 9/20 (45%) | **9/20 (45%)** |
| C: Student MEM latent | 11/20 (55%) | 11/20 (55%) | **10/20 (50%)** |

Difficulty breakdown:

| Condition | Easy | Medium | Hard |
|---|---:|---:|---:|
| A: official SimpleSG | 6/10 | 3/5 | 3/5 |
| B: Oracle latent | 6/10 | 1/5 | 2/5 |
| C: Student MEM latent | 7/10 | 2/5 | 1/5 |

Successful episode ids:

- A: `{0, 1, 2, 7, 9, 12, 14, 15, 16, 17, 18, 19}`
- B: `{0, 1, 9, 12, 15, 16, 17, 18, 19}`
- C: `{0, 1, 5, 9, 10, 12, 15, 16, 17, 18}`

## Attribution

### Low-level/action ceiling

Even the official SimpleSG condition fails to complete a first real grasp in
8/20 episodes.  In A and B, every episode that completes its first real grasp
also completes the first place and the entire task.  Thus their dominant error
is not long-horizon counting: it is failure to establish the first physical
grasp.

### Latent-action bridge

Oracle latent falls from 12/20 to 9/20 relative to official SimpleSG.  Its nine
successful episodes are a strict subset of the SimpleSG successful set; latent
conditioning additionally loses episodes 2, 7, and 14.  Because B uses perfect
simulator phase, this 15-point gap cannot be attributed to learned MEM state
estimation.  It is attributable to the latent-conditioned action path, although
A and B use different official/aligned checkpoint wrappers and therefore do
not isolate a single adapter layer.

### Learned MEM

Student MEM achieves 96.82% exact dynamic-state agreement after allowing the
expected one-action-query (16-step) phase-update delay.  In 9/20 episodes, no
real first grasp occurs and the MEM correctly remains at the initial latent for
the whole rollout.  Those failures are action/latent-execution failures, not
false MEM transitions.

There is one clear MEM-dominated failure: episode 19.  The model commits an
extra pick/place pair while the simulator is still waiting for the second pick,
advances the count early, and reaches `ready_to_press` while the simulator still
requires the fifth pick.  This explains the difference between 11 episodes
with a real pick/place and 10 final successes.

The Student condition happens to score 10/20 versus Oracle latent 9/20, but
this must not be interpreted as learned memory outperforming the oracle.
Diffusion-action sampling is not bitwise paired across independent policy
server runs: even when B and C present the same initial latent, their sampled
action sequences differ.  The robust conclusions are the phase/physical-event
attributions within each rollout and the large SimpleSG-to-Oracle-latent gap.

## Decision

The current bottleneck is a combination of:

1. an official action ceiling of only 60% on this paired slice;
2. a further 15-point loss in the latent-conditioned action path;
3. a smaller but real learned-MEM recurrent drift failure on long repetition.

The next highest-value work is to align/distill the official SimpleSG action
behavior into the latent-conditioned action wrapper while keeping Oracle
memory fixed.  MEM should remain frozen during that experiment.  A new formal
MEM evaluation is justified only after `Oracle latent + action` approaches the
official SimpleSG ceiling and preferably exceeds 75--80% on a validation set.

Result roots:

- `robomme_policy_learning/runs/evaluation/pick-paired-official-simplesg-test20-seed7-260901`
- `robomme_policy_learning/runs/evaluation/pick-paired-oracle-latent-test20-seed7-260901`
- `robomme_policy_learning/runs/evaluation/pick-paired-student-mem-test20-seed7-260901`
