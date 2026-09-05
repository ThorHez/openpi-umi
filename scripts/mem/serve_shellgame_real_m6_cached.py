#!/usr/bin/env python3
# ruff: noqa: E402
"""Serve M6 and derive its direction prompt from cached MEM classification."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _training_config
from openpi.training.mem.recipes import shellgame_real_wrist_m6 as _m6
from openpi.training.mem.recipes import shellgame_real_wrist_m6_direction_stage1_h32 as _m6_h32
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_ablation as _prompt_ablation
from openpi.training.mem.recipes import shellgame_real_wrist_m6_prompt_only_full_suffix as _prompt_full_suffix
from scripts.mem.serve_shellgame_real_stage2_cached import CUP_NAMES
from scripts.mem.serve_shellgame_real_stage2_cached import CURRENT_FRAME
from scripts.mem.serve_shellgame_real_stage2_cached import HISTORY_FRAMES
from scripts.mem.serve_shellgame_real_stage2_cached import MODEL_ACTION_DIM
from scripts.mem.serve_shellgame_real_stage2_cached import VIDEO_CURRENT_STEP_KEY
from scripts.mem.serve_shellgame_real_stage2_cached import VIDEO_FRAME_KEY_PREFIX
from scripts.mem.serve_shellgame_real_stage2_cached import summarize_memory_classification


class M6CachedHistoryPolicy:
    def __init__(self, memory_policy, action_policy, *, action_horizon: int, current_only_action: bool):
        self._memory_policy = memory_policy
        self._action_policy = action_policy
        self._action_horizon = int(action_horizon)
        self._current_only_action = bool(current_only_action)
        self._history: dict[str, object] | None = None
        self._memory: dict[str, object] | None = None

    def infer(self, obs: dict) -> dict:
        mode = obs.get("mode")
        if mode == "reset":
            self._history = None
            self._memory = None
            return {"cache_ready": False}
        if mode == "reset_history":
            expected = {f"{VIDEO_FRAME_KEY_PREFIX}{index}" for index in range(HISTORY_FRAMES)}
            present = {key for key in obs if key.startswith(VIDEO_FRAME_KEY_PREFIX)}
            if present != expected:
                raise ValueError(f"Expected history image keys 0..{HISTORY_FRAMES - 1}, got {len(present)} keys")
            self._history = {key: obs[key] for key in expected}
            classifier_obs = dict(self._history)
            classifier_obs[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = self._history[
                f"{VIDEO_FRAME_KEY_PREFIX}{HISTORY_FRAMES - 1}"
            ]
            classifier_obs.update(
                {
                    "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                    "robot0_eef_rot_axis_angle": np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                    "robot0_gripper_width": np.zeros(1, dtype=np.float32),
                    "prompt": _m6.direction_prompt(1),
                    # Required only by the mixed-dataset transform.  The MEM
                    # classifier itself does not consume these dummy values.
                    "episode_index": np.asarray(0, dtype=np.int32),
                    "frame_index": np.asarray(CURRENT_FRAME, dtype=np.int32),
                    "episode_length": np.asarray(CURRENT_FRAME + 1, dtype=np.int32),
                }
            )
            self._memory = summarize_memory_classification(self._memory_policy.infer_memory(classifier_obs))
            predicted_cup = int(self._memory["predicted_final_cup"])
            self._memory["direction_prompt"] = _m6.direction_prompt(predicted_cup)
            logging.info(
                "M6 MEM final cup=%s probabilities=%s prompt=%r",
                CUP_NAMES[predicted_cup],
                np.round(self._memory["predicted_final_cup_probabilities"], 4).tolist(),
                self._memory["direction_prompt"],
            )
            return {"cache_ready": True, "memory": self._memory}
        if mode == "infer_step":
            if self._history is None or self._memory is None:
                raise RuntimeError("History is not cached; upload reset_history first")
            action_obs = {
                key: value
                for key, value in obs.items()
                if key not in {"mode", VIDEO_CURRENT_STEP_KEY, "noise_seed", "prompt"}
            }
            action_obs.setdefault("episode_index", np.asarray(0, dtype=np.int32))
            action_obs.setdefault("frame_index", np.asarray(CURRENT_FRAME, dtype=np.int32))
            action_obs.setdefault("episode_length", np.asarray(CURRENT_FRAME + 2, dtype=np.int32))
            if self._current_only_action:
                # Prompt-only action sampling never reads history. Present the
                # current frame as a one-frame video instead of restacking and
                # transferring the 241 cached frames on every control step.
                action_obs[f"{VIDEO_FRAME_KEY_PREFIX}0"] = obs[VIDEO_CURRENT_STEP_KEY]
            else:
                action_obs = {**self._history, **action_obs}
                action_obs[f"{VIDEO_FRAME_KEY_PREFIX}{CURRENT_FRAME}"] = obs[VIDEO_CURRENT_STEP_KEY]
            # Never trust a client-supplied direction at deployment. The
            # prompt and raw memory are derived from the same frozen history.
            action_obs["prompt"] = self._memory["direction_prompt"]
            noise_seed = int(obs.get("noise_seed", 0))
            noise = np.random.default_rng(noise_seed).standard_normal(
                (self._action_horizon, MODEL_ACTION_DIM), dtype=np.float32
            )
            result = self._action_policy.infer(action_obs, noise=noise)
            result["memory"] = self._memory
            return result
        return self._action_policy.infer(obs)


def _shared_model_policy_view(shared_policy: _policy.Policy, config, checkpoint: Path) -> _policy.Policy:
    """Create a history-transform view while reusing the same loaded model arrays."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.robot_type is not None:
        data_config = _training_config._set_robot_type(data_config, data_config.robot_type)  # noqa: SLF001
    if data_config.asset_id is None:
        raise ValueError("Shared policy view requires an asset_id")
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    return _policy.Policy(
        shared_policy._model,  # noqa: SLF001 - intentionally one shared model instance
        transforms=[
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_masks=getattr(data_config, "normalize_masks", None),
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_masks=getattr(data_config, "normalize_masks", None),
            ),
            *data_config.data_transforms.outputs,
        ],
        metadata=shared_policy.metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-horizon", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--condition-mode",
        choices=("prompt_memory", "prompt_only"),
        default="prompt_memory",
    )
    parser.add_argument("--port", type=int, default=8017)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if args.condition_mode == "prompt_only":
        if args.action_horizon != 16:
            raise ValueError("prompt_only currently supports action_horizon=16 only")
        action_config = _prompt_full_suffix.make_train_config(
            exp_name="serve_only",
            checkpoint=str(checkpoint),
            steps=1,
            batch_size=1,
            fsdp_devices=1,
            num_workers=0,
            eval_interval=1,
            eval_batches=1,
            save_interval=1,
        )
        action_policy = policy_config.create_trained_policy(action_config, checkpoint)
        history_config = _prompt_ablation.make_train_config(
            condition_mode="prompt_only",
            exp_name="serve_history_only",
            checkpoint=str(checkpoint),
            steps=1,
            batch_size=1,
            fsdp_devices=1,
            num_workers=0,
            eval_interval=1,
            eval_batches=1,
            save_interval=1,
        )
        memory_policy = _shared_model_policy_view(action_policy, history_config, checkpoint)
        current_only_action = True
    else:
        recipe = _m6_h32 if args.action_horizon == 32 else _m6
        config = recipe.make_train_config(
            exp_name="serve_only",
            checkpoint=str(checkpoint),
            steps=1,
            batch_size=1,
            fsdp_devices=1,
            num_workers=0,
            eval_interval=1,
            eval_batches=1,
            save_interval=1,
        )
        action_policy = policy_config.create_trained_policy(config, checkpoint)
        memory_policy = action_policy
        current_only_action = False
    metadata = dict(action_policy.metadata)
    metadata.update(
        {
            "supports_cached_infer": True,
            "history_frames": HISTORY_FRAMES,
            "total_model_frames": HISTORY_FRAMES + 1,
            "action_horizon": args.action_horizon,
            "direction_prompt_source": "frozen_mem_final_cup",
            "action_condition_mode": args.condition_mode,
            "current_only_action_input": current_only_action,
            "direction_prompt_template": _m6.PROMPT_TEMPLATE,
            "action_contract": "current_frame_same_anchor_relative_link6_eef10",
            "state_contract": "episode_first_relative_link6_eef10",
        }
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=M6CachedHistoryPolicy(
            memory_policy,
            action_policy,
            action_horizon=args.action_horizon,
            current_only_action=current_only_action,
        ),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
