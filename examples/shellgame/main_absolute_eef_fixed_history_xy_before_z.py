"""Fixed-history absolute-EEF evaluation with an XY-before-Z descent guard.

The policy, history input, diffusion chunk, rotation, and gripper commands are
unchanged.  For an absolute OSC command that requests downward motion, Z is
held at the measured EEF height until the command's XY target is within the
configured threshold of the measured EEF XY.  Upward motion is never gated.
"""

from __future__ import annotations

import dataclasses
import logging

import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import tyro


@dataclasses.dataclass
class Args(base.Args):
    xy_before_z_threshold: float = 0.005
    xy_before_z_descent_epsilon: float = 0.0005
    # predicted_command is deployable without object pose perception.  The
    # nearest_cup mode is a simulator diagnostic: cup identity is inferred
    # from the policy command, while only cup geometry comes from the env.
    xy_before_z_reference: str = "predicted_command"
    xy_before_z_latch: bool = False


_base_policy_env_action = base._policy_env_action  # noqa: SLF001


def _guarded_policy_env_action(
    shell,
    env,
    history,
    start_eef_pos,
    start_eef_quat,
    action_plan,
    gripper_action,
    *,
    client,
    args: Args,
    prompt=None,
):
    env_action, next_gripper, policy_action = _base_policy_env_action(
        shell,
        env,
        history,
        start_eef_pos,
        start_eef_quat,
        action_plan,
        gripper_action,
        client=client,
        args=args,
        prompt=prompt,
    )
    if args.action_mode != "raw7" or policy_action is None:
        return env_action, next_gripper, policy_action

    measured = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
    guarded = np.asarray(env_action, dtype=np.float32).copy()
    if args.xy_before_z_reference == "predicted_command":
        reference_xy = guarded[:2]
    elif args.xy_before_z_reference == "nearest_cup":
        cup_positions = base._cup_positions(shell, env)  # noqa: SLF001
        state = getattr(env, "_xy_before_z_guard_state", None)
        if state is None:
            state = {"inferred_cup": None, "released": False}
            env._xy_before_z_guard_state = state  # noqa: SLF001
        inferred_cup = state["inferred_cup"]
        if inferred_cup is None or not args.xy_before_z_latch:
            inferred_cup = min(
                cup_positions,
                key=lambda cup: float(np.linalg.norm(guarded[:2] - cup_positions[cup][:2])),
            )
            if state["inferred_cup"] is None:
                logging.info("XY-before-Z inferred target cup=%s", inferred_cup)
            state["inferred_cup"] = inferred_cup
        reference_xy = np.asarray(cup_positions[inferred_cup][:2], dtype=np.float32)
    else:
        raise ValueError(
            "--xy-before-z-reference must be 'predicted_command' or 'nearest_cup'"
        )
    xy_distance = float(np.linalg.norm(reference_xy - measured[:2]))
    if args.xy_before_z_reference == "nearest_cup" and args.xy_before_z_latch:
        state = env._xy_before_z_guard_state  # noqa: SLF001
        if not state["released"] and xy_distance <= args.xy_before_z_threshold:
            state["released"] = True
            logging.info(
                "XY-before-Z released at distance=%.2fmm target=%s",
                xy_distance * 1_000.0,
                state["inferred_cup"],
            )
        guard_active = not state["released"]
    else:
        guard_active = True
    descending = float(guarded[2]) < float(measured[2]) - args.xy_before_z_descent_epsilon
    if guard_active and descending and xy_distance > args.xy_before_z_threshold:
        guarded[2] = measured[2]
    return guarded, next_gripper, policy_action


def main() -> None:
    args = tyro.cli(Args)
    if args.xy_before_z_threshold <= 0:
        raise ValueError("--xy-before-z-threshold must be positive")
    if args.xy_before_z_descent_epsilon < 0:
        raise ValueError("--xy-before-z-descent-epsilon must be non-negative")
    if args.xy_before_z_reference not in {"predicted_command", "nearest_cup"}:
        raise ValueError(
            "--xy-before-z-reference must be 'predicted_command' or 'nearest_cup'"
        )
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        "XY-before-Z guard enabled: threshold=%.1fmm descent_epsilon=%.1fmm reference=%s latch=%s",
        args.xy_before_z_threshold * 1_000.0,
        args.xy_before_z_descent_epsilon * 1_000.0,
        args.xy_before_z_reference,
        args.xy_before_z_latch,
    )
    base._policy_input = fixed_eef._fixed_history_policy_input  # noqa: SLF001
    base._policy_env_action = _guarded_policy_env_action  # noqa: SLF001
    base.eval_shellgame(args)


if __name__ == "__main__":
    main()
