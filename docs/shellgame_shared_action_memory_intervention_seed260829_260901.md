# ShellGame shared-action memory intervention, seed 260829 (2026-09-01)

## Question

Does the deployment-matched frozen external-memory action head causally use the
student memory, without adding a ShellGame-specific waypoint anchor?

## Scope and protocol

- This is a one-seed diagnostic of the deployment-matched ShellGame state-only
  memory and frozen external-memory action head. It is **not** an evaluation of
  the four-task RoboMME checkpoint, which has not been promoted to closed-loop
  control.
- Evaluation seed: 260829.
- Held-out episodes: 50.
- Conditions: normal student memory (`direct_visual`), a different-episode
  memory (`wrong_visual`), and an all-zero memory (`zero`).
- The action checkpoint, episode list, four-step diffusion sampler, 8-step
  replanning interval, 150-step horizon, and noise contract are fixed.
- Diffusion noise is `SeedSequence([260829, episode, policy_query_index])` in
  every condition.
- No memory or action parameter is updated.
- The three conditions were evaluated in parallel on identical A100 GPUs. The
  merged artifact verifies exactly 50 records per condition and one record for
  every episode-condition pair. No EGL corrupt-read retry occurred.

## Results on all 50 episodes

| Memory input | Correct target | Within 30 mm | Target contact | Lift |
|---|---:|---:|---:|---:|
| Normal student memory | 21/50 (42%) | 11/50 (22%) | 26/50 (52%) | 0/50 (0%) |
| Different-episode memory | 21/50 (42%) | 14/50 (28%) | 27/50 (54%) | 1/50 (2%) |
| Zero memory | 21/50 (42%) | 34/50 (68%) | 39/50 (78%) | 2/50 (4%) |

Against normal student memory, paired exact McNemar tests give:

- different-episode memory: `p=1.0` for correct target, `p=0.375` for 30-mm
  approach, and `p=1.0` for target contact;
- zero memory: `p=1.0` for correct target, `p=1.52e-5` for 30-mm approach, and
  `p=0.00443` for target contact.

The different-episode condition is not guaranteed to be semantically wrong on
episodes where the student memory is itself wrong. The conditioned result below
is therefore the cleaner interface diagnostic.

## Conditioned on semantically correct student memory

The student memory is semantically correct on 16/50 episodes. On this fixed
subset, the different-episode donor has a different predicted final state.

| Memory input | Correct target | Within 30 mm | Target contact | Lift |
|---|---:|---:|---:|---:|
| Normal student memory | 13/16 (81.2%) | 6/16 (37.5%) | 12/16 (75.0%) | 0/16 (0%) |
| Different-episode memory | 13/16 (81.2%) | 7/16 (43.8%) | 13/16 (81.2%) | 1/16 (6.2%) |
| Zero memory | 12/16 (75.0%) | 12/16 (75.0%) | 13/16 (81.2%) | 2/16 (12.5%) |

No conditioned normal-versus-intervention difference is significant by the
paired exact McNemar test (`p>=0.0703` for every reported endpoint).

## Interpretation

This run does not support the old Table VI claim that the current action
interface benefits from external student memory. Correct-target selection is
unchanged by a different-episode memory, and zero memory improves the all-50
approach and contact metrics. On the semantically correct subset, normal memory
still has no measured advantage. The result is consistent with an action head
that does not use this student-memory distribution reliably and may be
perturbed by it.

This is a negative one-seed diagnostic, not a three-seed paper result. It should
not replace Table VI as a positive result. The defensible paper choices are to
remove the old waypoint-anchor table, or to repeat this intervention over three
seeds and report the negative interface finding explicitly.

## Artifacts

- Merged result: `evaluation/shellgame/shared_action_memory_intervention_seed260829_260901_v2/result.json`
- Per-condition results and logs: subdirectories of the merged result directory.
- Reproduction runner: `examples/shellgame/run_eval_shared_action_memory_intervention_one_seed_260901.sh`
