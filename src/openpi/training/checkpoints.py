from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import pathlib
from typing import Protocol, TYPE_CHECKING

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
if TYPE_CHECKING:
    import openpi.training.data_loader as _data_loader
    import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def _save_norm_assets(directory: epath.Path, configs) -> None:
    """Validate every target before writing; shared assets are saved just once."""
    pending = {}
    for config in configs:
        stats = config.norm_stats
        if stats is None:
            continue
        asset_id = config.asset_id
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("A normalization asset needs an explicit checkpoint asset_id")
        target = pathlib.PurePosixPath(asset_id)
        if (target.is_absolute() or ".." in target.parts or "\\" in asset_id
                or any(ord(char) < 32 for char in asset_id)):
            raise ValueError("Normalization asset_id must stay inside the checkpoint assets directory")
        asset_id = target.as_posix()
        metadata = getattr(config, "norm_stats_metadata", None)
        document = json.loads(_normalize.serialize_json(stats))
        if metadata is not None:
            document["shared_normalization"] = metadata
        method = getattr(config, "normalization_method", None)
        if method is not None:
            if method not in ("minmax", "quantile", "zscore"):
                raise ValueError("Normalization method must be minmax, quantile or zscore")
            document["normalization_method"] = method
        identity = json.dumps(document, sort_keys=True, allow_nan=False)
        if asset_id in pending and pending[asset_id][0] != identity:
            raise ValueError(
                f"Conflicting normalization statistics target the same checkpoint asset '{asset_id}'; "
                "bind one shared raw-sample statistic or use distinct per-source asset IDs"
            )
        pending[asset_id] = (identity, stats, metadata, method)
    files = [pathlib.PurePosixPath(asset_id) / "norm_stats.json" for asset_id in pending]
    if any(left in right.parents for left in files for right in files if left != right):
        raise ValueError("Normalization asset paths conflict with another statistics file")
    for asset_id, (_, stats, metadata, method) in pending.items():
        explicit = {"normalization_method": method} if method is not None else {}
        _normalize.save(directory / asset_id, stats, shared_normalization=metadata, **explicit)


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        configs = data_loader.data_configs() if hasattr(data_loader, "data_configs") else [data_loader.data_config()]
        _save_norm_assets(directory, configs)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
    *,
    sharding=None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        items = {"train_state": train_state, "params": {"params": params}}
        if sharding is None:
            restored = checkpoint_manager.restore(step, items=items)
        else:
            state_sharding, params_sharding = _split_params(sharding)
            restore_args = ocp.checkpoint_utils.construct_restore_args(
                items, {"train_state": state_sharding, "params": {"params": params_sharding}},
            )
            restored = checkpoint_manager.restore(step, args=ocp.args.Composite(**{
                key: ocp.args.PyTreeRestore(item=item, restore_args=restore_args[key]) for key, item in items.items()
            }))
    return _merge_params(restored["train_state"], restored["params"])


def resolve_resume_checkpoint(checkpoint: str, destination: epath.Path | str, target_steps: int) -> pathlib.Path:
    """Check the explicit native source step before opening the destination writer."""
    source = pathlib.Path(checkpoint)
    if (not source.is_absolute() or not source.name.isascii() or not source.name.isdigit()
            or str(int(source.name)) != source.name):
        raise ValueError("RESUME_CHECKPOINT_INVALID: expected an absolute numbered OpenPI checkpoint directory")
    try:
        resolved = source.resolve(strict=True)
        destination = pathlib.Path(destination).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("RESUME_CHECKPOINT_INVALID: checkpoint or output path is unavailable") from exc
    if source != resolved or not source.is_dir():
        raise ValueError("RESUME_CHECKPOINT_INVALID: checkpoint must be an existing directory without symlinks")
    if destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination):
        raise ValueError("RESUME_CHECKPOINT_INVALID: checkpoint input and output directories must be separate")
    if target_steps <= int(source.name) + 1:
        raise ValueError("RESUME_STEP_INVALID: target total steps must exceed the checkpoint's completed steps")
    return source


def restore_state_from_checkpoint(
    source: pathlib.Path,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    *,
    sharding,
    target_steps: int,
) -> training_utils.TrainState:
    """Restore weights, optimizer and EMA from one finalized step using a read-only manager."""
    step = int(source.name)
    with ocp.CheckpointManager(
        source.parent,
        item_handlers={"train_state": ocp.PyTreeCheckpointHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(
            read_only=True,
            save_interval_steps=0,
            create=False,
            cleanup_tmp_directories=False,
            save_root_metadata=False,
            enable_async_checkpointing=False,
        ),
    ) as reader:
        if step not in reader.all_steps():
            raise ValueError("RESUME_CHECKPOINT_INVALID: selected checkpoint step is missing or incomplete")
        restored = restore_state(reader, state, data_loader, step=step, sharding=sharding)
    expected_leaves, expected_structure = jax.tree.flatten(state)
    restored_leaves, restored_structure = jax.tree.flatten(restored)
    if expected_structure != restored_structure or any(
        expected.shape != actual.shape or expected.dtype != actual.dtype
        for expected, actual in zip(expected_leaves, restored_leaves, strict=True)
    ):
        raise ValueError("RESUME_CHECKPOINT_INCOMPATIBLE: restored training state does not match the model or optimizer")
    actual_step = int(restored.step)
    # Native save_state labels the checkpoint with the loop index after applying
    # that update. The state carries the next iteration's index.
    if actual_step != step + 1 or target_steps <= actual_step:
        raise ValueError("RESUME_STEP_INVALID: restored train step does not match checkpoint label or target total")
    return restored


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
