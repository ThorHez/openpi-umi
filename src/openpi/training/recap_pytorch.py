"""Native PyTorch training and checkpoint handling for the ReCap stages."""

import contextlib
import dataclasses
import functools
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from datetime import timedelta

import numpy as np
import safetensors.torch
import torch

from openpi.models import model as observation_model
from openpi.models_pytorch import recap
from openpi.training import checkpoints, optimizer
from openpi.training.training_normalization import prepare_training_normalization


PLATFORM_STAGE_PROGRESS_VALIDATION_VERSION = 1


_OBSERVATION_FIELDS = ({field.name for field in dataclasses.fields(observation_model.Observation)}
                       - {"images", "image_masks", "frame_valid_masks"}
                       | {"image", "image_mask", "frame_valid_mask"})


def init_wandb(config, *, resuming=False, enabled=True):
    if enabled:
        import wandb

        wandb.init(name=config.exp_name, project=config.project_name,
                   config={"stage": config.recap_training_stage, "batch_size": config.batch_size,
                           "steps": config.num_train_steps, "precision": config.pytorch_training_precision})


def learning_rate(schedule, step):
    if not isinstance(schedule, (optimizer.CosineDecaySchedule, optimizer.RsqrtDecaySchedule)):
        raise ValueError("PyTorch ReCap 支持余弦或逆平方根学习率调度")
    if step < schedule.warmup_steps:
        start = schedule.peak_lr / (schedule.warmup_steps + 1)
        return start + (schedule.peak_lr - start) * step / schedule.warmup_steps
    if isinstance(schedule, optimizer.RsqrtDecaySchedule):
        return schedule.peak_lr / math.sqrt(1 + (step - schedule.warmup_steps) / schedule.timescale)
    progress = min(1.0, (step - schedule.warmup_steps) / max(1, schedule.decay_steps - schedule.warmup_steps))
    return schedule.decay_lr + (schedule.peak_lr - schedule.decay_lr) * (1 + math.cos(math.pi * progress)) / 2


def build_optimizer(config, parameters):
    lr = learning_rate(config.lr_schedule, 0)
    settings = config.optimizer
    if isinstance(settings, optimizer.AdamW):
        return torch.optim.AdamW(parameters, lr=lr, betas=(settings.b1, settings.b2), eps=settings.eps,
                                 weight_decay=settings.weight_decay)
    if isinstance(settings, optimizer.SGD):
        return torch.optim.SGD(parameters, lr=lr, momentum=settings.momentum, nesterov=settings.nesterov)
    raise ValueError("PyTorch ReCap 支持 AdamW 或 SGD 优化器")


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return torch.as_tensor(value).to(device, non_blocking=True)


def torch_observation(sample, device):
    fields = {key: value for key, value in sample.items() if key in _OBSERVATION_FIELDS and value is not None}
    return observation_model.Observation.from_dict(_to_device(fields, device))


def make_loader(dataset, config, *, batch_size, sampler=None, shuffle=False, drop_last=False, seed=None):
    from openpi.training import data_loader

    workers = config.num_workers
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, drop_last=drop_last,
        num_workers=workers, persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
        collate_fn=functools.partial(data_loader._collate_fn, image_keys=config.model.image_keys,
                                     depth_keys=config.model.depth_keys),
        generator=torch.Generator().manual_seed(config.seed if seed is None else seed),
    )


def _training_datasets(config, rank):
    if config.recap_training_stage == "stage_progress":
        from openpi.training import stage_progress

        source = stage_progress.StageProgressSource(config, skip_norm_stats=True)
        if not config.val_ratio:
            return stage_progress.StageProgressPairs(source, include_actions=True), None, [source.data_config], None
        if not 0 < config.val_ratio < 1:
            raise ValueError("阶段进度验证集比例须大于 0 且小于 1")
        episodes = [rows for rows in source.episodes if len(rows) > 1]
        if len(episodes) < 2:
            raise ValueError("阶段进度验证至少需要两条各含两帧的轨迹。请增加轨迹或将验证集比例设为 0")
        validation_size = min(len(episodes) - 1, max(1, round(len(episodes) * config.val_ratio)))
        order = np.random.default_rng(config.seed).permutation(len(episodes))
        held_out = set(order[:validation_size])
        train = stage_progress.StageProgressPairs(
            source, include_actions=True, episodes=[rows for index, rows in enumerate(episodes) if index not in held_out]
        )
        validation = stage_progress.StageProgressPairs(
            source, include_actions=True, episodes=[episodes[index] for index in order[:validation_size]],
            reference_seed=config.seed,
        ) if rank == 0 else None
        return train, validation, [source.data_config], None
    from openpi.training import data_loader
    from openpi.training.multi_data_loader import WeightedConcatDataset

    multi = hasattr(config.data, "datasets")
    configs = (config.data.create_all(config.assets_dirs, config.model) if multi
               else [config.data.create(config.assets_dirs, config.model)])
    configs = [data_loader._apply_global_normalization(data, config) for data in configs]
    datasets = [data_loader.transform_dataset(
        data_loader.create_torch_dataset(data, config.model.action_horizon, config.model), data
    ) for data in configs]
    weights = config.data.weights if multi else None
    dataset = WeightedConcatDataset(datasets, weights=weights)
    sample_weights = dataset.get_dataset_weights_for_sampler() if weights and len(set(weights)) > 1 else None
    return dataset, None, configs, sample_weights


def _training_loaders(config, rank, world_size):
    train, validation, configs, weights = _training_datasets(config, rank)
    if config.recap_training_stage == "value" and config.val_ratio:
        validation_size = max(1, round(len(train) * config.val_ratio))
        train, validation = torch.utils.data.random_split(
            train, [len(train) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
        if weights is not None:
            weights = [weights[index] for index in train.indices]
    if len(train) < config.batch_size:
        raise ValueError("PyTorch ReCap 训练样本不足以组成一个全局 batch，请减小 batch 或验证集比例")
    batch_size = config.batch_size // world_size
    sampler = None
    seed = config.seed + rank
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train) // world_size, replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    elif world_size > 1:
        sampler = torch.utils.data.DistributedSampler(train, world_size, rank, seed=config.seed, drop_last=True)
    train_loader = make_loader(train, config, batch_size=batch_size, sampler=sampler,
                               shuffle=sampler is None, drop_last=True, seed=seed)
    validation_loader = (make_loader(validation, config, batch_size=config.eval_batch_size or batch_size)
                         if validation is not None and rank == 0 else None)
    return train_loader, validation_loader, configs


def _batches(loader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _batch_loss(model, batch, stage, device):
    if stage == "stage_progress":
        return model(torch_observation(batch["reference"], device), torch_observation(batch["current"], device),
                     _to_device(batch["target"], device), _to_device(batch["actions"], device))
    observation = torch_observation(batch, device)
    if stage == "value":
        if observation.value_target is None:
            raise ValueError("PyTorch ReCap 价值训练要求数据提供 value_target")
        target = observation.value_target.reshape(-1)
        if not torch.isfinite(target).all():
            raise ValueError("value_target 含非有限值")
        return model(observation, target)
    return model(observation, _to_device(batch["actions"], device))


@torch.inference_mode()
def _evaluate(model, loader, config, device):
    import itertools

    stage = config.recap_training_stage
    progress_validation = stage == "stage_progress"
    devices = [device] if device.type == "cuda" else []
    rng = torch.random.fork_rng(devices=devices) if progress_validation else contextlib.nullcontext()
    total, count = 0.0, 0
    model.eval()
    try:
        with rng:
            if progress_validation:
                torch.random.default_generator.manual_seed(config.seed)
                if devices:
                    torch.cuda.manual_seed(config.seed)
            for batch in itertools.islice(loader, config.eval_batches):
                loss = _batch_loss(model, batch, stage, device)
                total += loss.float().sum().item()
                count += loss.numel()
    finally:
        model.train()
    return total / count


def save_checkpoint(config, model, optim, data_configs, step):
    root = config.checkpoint_dir
    destination = root / str(step)
    with tempfile.TemporaryDirectory(dir=root, prefix=".checkpoint-") as temporary:
        staged = Path(temporary) / "checkpoint"
        staged.mkdir()
        safetensors.torch.save_model(model, str(staged / "model.safetensors"))
        torch.save(optim.state_dict(), staged / "optimizer.pt")
        (staged / "assets").mkdir()
        checkpoints._save_norm_assets(staged / "assets", data_configs)
        metadata = {"schema_version": 1, "backend": "pytorch", "stage": config.recap_training_stage, "step": step}
        (staged / "recap_checkpoint.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        staged.rename(destination)
    for older in root.iterdir():
        if (older.name.isdecimal() and int(older.name) < step and (older / "recap_checkpoint.json").is_file()
                and not (config.keep_period and int(older.name) % config.keep_period == 0)):
            shutil.rmtree(older)
    logging.info("已保存 PyTorch ReCap checkpoint %s", destination)


def _device(rank):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    fraction = os.environ.get("VLA_RECAP_TORCH_MEMORY_FRACTION")
    if fraction is not None:
        limit = float(fraction)
        if not math.isfinite(limit) or not 0 < limit < 1:
            raise ValueError("PyTorch ReCap 共享显存比例必须在 0 和 1 之间")
        torch.cuda.set_per_process_memory_fraction(limit, device)
    return device


def _log_metrics(config, values, step):
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("NON_FINITE_METRIC: PyTorch ReCap 训练指标出现 NaN 或 Inf")
    print("VLA_EVENT:" + json.dumps({"type": "METRIC", "data": {**values, "step": step,
          "total_steps": config.num_train_steps}}, allow_nan=False), flush=True)
    if config.wandb_enabled:
        import wandb

        wandb.log(values, step=step)


def _train_rank(rank, world_size, config, rendezvous=None, worker_setup=None):
    logging.basicConfig(level=logging.INFO)
    if worker_setup is not None:
        config = worker_setup(config)
    device = _device(rank)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size,
                                             timeout=timedelta(minutes=30), device_id=device)
    try:
        _train(config, rank, world_size, device)
    finally:
        if distributed:
            torch.distributed.destroy_process_group()


def _train(config, rank, world_size, device):
    stage = config.recap_training_stage
    loader, validation, data_configs = _training_loaders(config, rank, world_size)
    precision = config.pytorch_training_precision if device.type == "cuda" else "float32"
    module = recap.create_recap_model(config.model, stage, precision)
    recap.load_recap_weights(module, config.pytorch_weight_path, stage)
    recap.configure_recap_trainable(module, config)
    module.to(device)
    module.gradient_checkpointing_enable()
    module.train()
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("PyTorch ReCap 配置没有可训练参数")
    distributed = world_size > 1
    model = (torch.nn.parallel.DistributedDataParallel(module, device_ids=[rank], broadcast_buffers=False,
                                                       find_unused_parameters=True) if distributed else module)
    optim = build_optimizer(config, parameters)
    if rank == 0:
        init_wandb(config, enabled=config.wandb_enabled)
    batches = _batches(loader)
    for step in range(config.num_train_steps):
        batch = next(batches)
        lr = learning_rate(config.lr_schedule, step)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, stage, device).float().mean()
        loss.backward()
        clip = getattr(config.optimizer, "clip_gradient_norm", math.inf)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optim.step()
        final = step == config.num_train_steps - 1
        log = step % config.log_interval == 0 or final
        evaluate = (stage in {"value", "stage_progress"} and config.val_ratio > 0
                    and (step % config.eval_interval == 0 or final))
        save = final or (step > 0 and step % config.save_interval == 0)
        if log:
            metrics = torch.stack((loss.detach(), grad_norm.detach().float()))
            if distributed:
                torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.AVG)
        if rank == 0 and (log or evaluate):
            values = {"loss": float(metrics[0]), "grad_norm": float(metrics[1]), "learning_rate": lr} if log else {}
            if evaluate:
                values["val_loss"] = _evaluate(module, validation, config, device)
            _log_metrics(config, values, step)
        if rank == 0 and save:
            save_checkpoint(config, module, optim, data_configs, step)
        if distributed and (evaluate or save):
            torch.distributed.barrier(device_ids=[rank])
    if rank == 0 and config.wandb_enabled:
        import wandb

        wandb.finish()


def load_scoring_model(config, checkpoint, stage, device):
    precision = config.pytorch_training_precision if device.type == "cuda" else "float32"
    model = recap.create_recap_model(config.model, stage, precision)
    recap.load_recap_weights(model, checkpoint, stage, require_trained_head=True)
    return model.to(device).eval()


def run(config, *, worker_setup=None):
    stage = config.recap_training_stage
    if stage not in {"value", "stage_progress", "offline_rl"} or config.pytorch_weight_path is None:
        raise ValueError("PyTorch ReCap 要求选择训练阶段和包含 model.safetensors 的初始权重")
    if config.resume or config.resume_checkpoint is not None or config.overwrite:
        raise ValueError("PyTorch ReCap 使用独立的新训练目录")
    if config.ema_decay is not None:
        raise ValueError("PyTorch ReCap 训练配置需将 ema_decay 设为 None")
    world_size = max(1, torch.cuda.device_count())
    if config.batch_size < 1 or config.batch_size % world_size or config.num_train_steps < 1:
        raise ValueError("PyTorch ReCap 的全局 batch 必须整除设备数量，且训练步数必须为正")
    config = prepare_training_normalization(config)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if world_size == 1:
        _train_rank(0, 1, config)
    else:
        with tempfile.TemporaryDirectory(dir=config.checkpoint_dir, prefix=".ddp-") as temporary:
            rendezvous = (Path(temporary) / "store").as_uri()
            torch.multiprocessing.spawn(_train_rank, args=(world_size, config, rendezvous, worker_setup), nprocs=world_size)
