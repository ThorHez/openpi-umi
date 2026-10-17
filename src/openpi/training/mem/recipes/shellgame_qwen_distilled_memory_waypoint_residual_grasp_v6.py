"""Train bounded visual XY residuals around the frozen ShellGame MEM waypoint."""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx

from openpi.shared import nnx_utils
from openpi.tasks.shellgame import pi0_qwen_event_memory_waypoint_residual_action as _residual
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_frame59_waypoint as _waypoint
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _grasp


DEFAULT_INIT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000/params"
)


def make_model_config() -> _residual.Pi0QwenEventMemoryWaypointResidualActionConfig:
    base = _waypoint.make_model_config()
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    return _residual.Pi0QwenEventMemoryWaypointResidualActionConfig(
        **values,
        waypoint_residual_normalized_limits=(0.395, 0.246),
    )


def make_train_config(
    *,
    config_module,
    exp_name: str,
    init_checkpoint: pathlib.Path = DEFAULT_INIT_CHECKPOINT,
    steps: int = 1_000,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    parent = _grasp.make_train_config(
        config_module=config_module,
        exp_name=exp_name,
        init_checkpoint=init_checkpoint,
        steps=steps,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    model = make_model_config()
    action_expert = nnx_utils.PathRegex(r".*PaliGemma/llm/.*_1.*")
    action_modules = nnx_utils.PathRegex(
        r".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*"
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_qwen_distilled_memory_waypoint_residual_grasp_v6_eef7_260827",
        model=model,
        freeze_filter=nnx.Not(nnx.Any(action_expert, action_modules)),
        save_interval=250,
        keep_period=250,
        eval_interval=125,
    )


make_index_filter = _grasp.make_index_filter
