"""Adapt the reproduced V10 action expert to the Qwen-memory waypoint bridge.

Initialization is a strict hybrid: all current semantic-memory and waypoint
parameters come from the validated waypoint-grasp V6 checkpoint, while only
the 19 Pi0 action-expert/projection leaves come from reproduced V10 step 1000.
Training keeps the complete memory-to-waypoint conditioner frozen and updates
the action branch on the audited V6 60/40 nominal/correction mixture.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import flax.traverse_util
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.tasks.shellgame import v10_action_weight_transplant as _transplant
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_waypoint_grasp_v6 as _grasp

DEFAULT_CURRENT_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/"
    "direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000/params"
)
DEFAULT_V10_ACTION_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/"
    "1000/params"
)


@dataclasses.dataclass(frozen=True)
class QwenMemoryV10ActionTransplantLoader:
    """Build an exact target tree from current MEM/waypoint plus V10 action."""

    current_params_path: str
    v10_params_path: str

    def load(self, params: at.Params) -> at.Params:
        current = _model.restore_params(self.current_params_path, restore_type=np.ndarray)
        v10 = _model.restore_params(self.v10_params_path, restore_type=np.ndarray)
        hybrid, report = _transplant.transplant_v10_action_params(current, v10)

        target = flax.traverse_util.flatten_dict(params, sep="/")
        source = flax.traverse_util.flatten_dict(hybrid, sep="/")
        result = {}
        missing = []
        mismatched = []
        for path, reference in target.items():
            candidate = source.get(path)
            if candidate is None:
                missing.append(path)
                continue
            if np.shape(candidate) != np.shape(reference):
                mismatched.append((path, np.shape(candidate), np.shape(reference)))
                continue
            result[path] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
        if missing or mismatched:
            raise ValueError(
                "Hybrid initialization does not exactly match the target model: "
                f"missing={missing[:8]} mismatched={mismatched[:8]}"
            )
        logging.info(
            "Initialized Qwen-memory/V10-action hybrid leaves=%d elements=%d action_expert=%d projections=%d",
            report.selected_leaves,
            report.selected_elements,
            report.action_expert_leaves,
            report.projection_leaves,
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


def make_train_config(
    *,
    config_module,
    exp_name: str,
    current_checkpoint: pathlib.Path = DEFAULT_CURRENT_CHECKPOINT,
    v10_action_checkpoint: pathlib.Path = DEFAULT_V10_ACTION_CHECKPOINT,
    steps: int = 1_000,
    peak_lr: float = 1e-6,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    current_checkpoint = current_checkpoint.expanduser().resolve()
    v10_action_checkpoint = v10_action_checkpoint.expanduser().resolve()
    if not current_checkpoint.is_dir():
        raise FileNotFoundError(current_checkpoint)
    if not v10_action_checkpoint.is_dir():
        raise FileNotFoundError(v10_action_checkpoint)
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")

    parent = _grasp.make_train_config(
        config_module=config_module,
        exp_name=exp_name,
        init_checkpoint=current_checkpoint,
        steps=steps,
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    return dataclasses.replace(
        parent,
        name="pi0_shellgame_qwen_memory_waypoint_v10_action_adapt_eef7_260827",
        weight_loader=QwenMemoryV10ActionTransplantLoader(
            current_params_path=str(current_checkpoint),
            v10_params_path=str(v10_action_checkpoint),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(100, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        save_interval=250,
        keep_period=250,
        eval_interval=125,
        eval_batches=20,
    )


make_index_filter = _grasp.make_index_filter
