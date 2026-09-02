"""Train only a semantic-memory adapter on the exact V10 action graph.

This is a controlled replacement experiment.  Every parameter and operation
from V10 is restored and frozen.  During training, the residual produced by
V10's old visual-memory cross attention is set to zero and a newly initialized
parallel adapter must condition the unchanged V10 action expert from frozen
Qwen-distilled semantic memory tokens.
"""

from __future__ import annotations

from argparse import Namespace
import dataclasses
import functools
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import numpy as np

from examples.shellgame import train_old_tracker_full_absolute_eef as _full_eef
from examples.shellgame import v10_exact_parallel_semantic_adapter as _exact
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_qwen_distilled_memory_action_v10 as _memory_data

DEFAULT_V10_CHECKPOINT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/"
    "absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/"
    "1000/params"
)


@dataclasses.dataclass(frozen=True)
class ExactV10ParallelAdapterLoader:
    """Strictly restore all V10 leaves and retain only fresh adapter leaves."""

    params_path: str

    def load(self, params):
        source = _model.restore_params(self.params_path, restore_type=np.ndarray)
        merged, counts = _exact.merge_exact_v10_with_fresh_parallel_adapter(params, source)
        logging.info(
            "Strict V10 initialization restored %d old leaves and initialized %d adapter leaves",
            counts["v10"],
            counts["parallel_adapter"],
        )
        return merged


def make_model_config() -> _exact.V10ExactParallelSemanticAdapterConfig:
    # Reconstruct the successful V10 architecture without importing an eval
    # script (those scripts intentionally depend on their launch directory).
    args = Namespace(
        exp_name="v10_exact_parallel_semantic_adapter",
        init_checkpoint="",
        steps=500,
        warmup_steps=50,
        peak_lr=1e-4,
        batch_size=12,
        num_workers=8,
        fsdp_devices=6,
        eval_interval=125,
        eval_batches=20,
        save_interval=125,
        keep_period=125,
        gripper_loss_weight=4.0,
        encoder_width=256,
        encoder_depth=2,
        encoder_heads=8,
        memory_width=64,
        memory_depth=2,
        memory_heads=4,
        adapter_heads=4,
        memory_tokens=128,
        current_tokens=256,
        residual_scale=1.0,
        video_mode="normal",
        initial_mode="normal",
        relation_mode="one_hot",
        raw_memory_mode="normal",
        query_tokens=16,
        query_width=256,
        query_depth=2,
        query_heads=4,
        action_cross_attention_heads=8,
        overwrite=False,
    )
    base = _full_eef.build_config(args).model
    model = _exact.make_config_from_v10(base)
    return dataclasses.replace(
        model,
        parallel_semantic_adapter_enabled=True,
        # Remove only the old-memory residual. The old tracker still runs, so
        # the graph, prefix cache and action expert are otherwise unchanged.
        old_memory_condition_strength=0.0,
        semantic_residual_gate_init=0.0,
    )


def _data_config(config_module: Any, root: pathlib.Path, memory_path: pathlib.Path):
    cls = _memory_data._base.data_config_type(config_module)  # noqa: SLF001
    return cls(
        repo_id=str(root),
        memory_path=str(memory_path.resolve()),
        assets=config_module.AssetsConfig(
            asset_id=".",
            assets_dir=str(_memory_data.NOMINAL_ROOT),
        ),
        base_config=config_module.UmiDataConfig(
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
            robot_type="ARM=1 G=0 H=0",
        ),
        # The exact V10 tracker contract is frames 0..59 plus the live frame.
        num_frames=61,
        frame_stride=1,
        video_layout="sliding",
    )


def make_index_filter(memory_banks: dict[pathlib.Path, pathlib.Path]):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).resolve()
        for root, path in memory_banks.items()
    }
    return functools.partial(_memory_data.filter_and_sample_indices, banks=banks)


def make_train_config(
    *,
    config_module: Any,
    exp_name: str,
    memory_banks: dict[pathlib.Path, pathlib.Path] | None = None,
    init_checkpoint: pathlib.Path = DEFAULT_V10_CHECKPOINT,
    steps: int = 500,
    peak_lr: float = 1e-4,
    batch_size: int = 12,
    fsdp_devices: int = 6,
    num_workers: int = 8,
    overwrite: bool = False,
):
    banks = {
        pathlib.Path(root).resolve(): pathlib.Path(path).resolve()
        for root, path in (
            _memory_data.DEFAULT_MEMORY_BANKS if memory_banks is None else memory_banks
        ).items()
    }
    init_checkpoint = pathlib.Path(init_checkpoint).resolve()
    if not init_checkpoint.is_dir():
        raise FileNotFoundError(init_checkpoint)
    for root in (
        _memory_data.NOMINAL_ROOT,
        _memory_data.V6_ROOT,
        _memory_data.V9_ROOT,
    ):
        if not root.is_dir():
            raise FileNotFoundError(root)
        if root not in banks or not banks[root].is_file():
            raise FileNotFoundError(f"No semantic memory bank for {root}: {banks.get(root)}")

    correct_counts = {
        root: _memory_data._correct_episode_count(root, banks)  # noqa: SLF001
        for root in banks
    }
    nominal_rows = (
        correct_counts[_memory_data.NOMINAL_ROOT]
        * _memory_data.ROWS_PER_CORRECT_EPISODE[_memory_data.NOMINAL_ROOT]
    )
    weights = {}
    for root in (
        _memory_data.NOMINAL_ROOT,
        _memory_data.V6_ROOT,
        _memory_data.V9_ROOT,
    ):
        rows = correct_counts[root] * _memory_data.ROWS_PER_CORRECT_EPISODE[root]
        weights[root] = (
            _memory_data.SOURCE_FRACTIONS[root]
            / _memory_data.SOURCE_FRACTIONS[_memory_data.NOMINAL_ROOT]
            * nominal_rows
            / rows
        )

    model = make_model_config()
    adapter = nnx_utils.PathRegex(r".*ParallelSemanticMemoryActionConditioner.*")
    logging.info(
        "Exact V10 semantic replacement: correct episodes nominal/v6/v9=%d/%d/%d "
        "weights=%s old_memory_strength=0 trainable=parallel_adapter_only",
        correct_counts[_memory_data.NOMINAL_ROOT],
        correct_counts[_memory_data.V6_ROOT],
        correct_counts[_memory_data.V9_ROOT],
        {root.name: weights[root] for root in weights},
    )
    return config_module.TrainConfig(
        name="pi0_shellgame_v10_exact_parallel_semantic_adapter_eef7_260827",
        exp_name=exp_name,
        model=model,
        freeze_filter=nnx.Not(adapter),
        data=config_module.MultiDataConfigFactory(
            state_pad_dim=96,
            datasets=[
                _data_config(config_module, _memory_data.V9_ROOT, banks[_memory_data.V9_ROOT]),
                _data_config(config_module, _memory_data.V6_ROOT, banks[_memory_data.V6_ROOT]),
                _data_config(
                    config_module,
                    _memory_data.NOMINAL_ROOT,
                    banks[_memory_data.NOMINAL_ROOT],
                ),
            ],
            weights=[
                weights[_memory_data.V9_ROOT],
                weights[_memory_data.V6_ROOT],
                weights[_memory_data.NOMINAL_ROOT],
            ],
            use_merged_norm_stats=False,
        ),
        weight_loader=ExactV10ParallelAdapterLoader(str(init_checkpoint)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(50, max(steps - 1, 0)),
            peak_lr=peak_lr,
            decay_steps=max(steps, 2),
            decay_lr=peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        log_interval=10,
        save_interval=125,
        keep_period=125,
        val_ratio=0.1,
        eval_interval=125,
        eval_batches=20,
        wandb_enabled=False,
        overwrite=overwrite,
        shellgame_memory_classifier=config_module.ShellgameMemoryClassifierConfig(enabled=False),
        shellgame_cup_eval=config_module.ShellgameCupEvalConfig(enabled=False),
    )
