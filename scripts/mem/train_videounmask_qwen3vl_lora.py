#!/usr/bin/env python3
# ruff: noqa: E402
"""LoRA SFT for Qwen3-VL RoboMME visual memory tasks.

The manifest is disk-light: VideoUnmask frames are decoded lazily from HDF5.
Optional ShellGame rows are accepted for rehearsal, and an existing adapter can
be loaded trainably for continued LoRA experiment B.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

from accelerate import Accelerator
from accelerate.utils import set_seed
import h5py
import numpy as np
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup
from transformers.video_utils import VideoMetadata

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import SYSTEM_PROMPT as PICK_SYSTEM_PROMPT
from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import prompt_for_task as pick_prompt
from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import validate_compact_response as validate_pick
from openpi.tasks.robomme.pickxtimes.qwen3vl_local_event_contract import SYSTEM_PROMPT as PICK_LOCAL_SYSTEM_PROMPT
from openpi.tasks.robomme.pickxtimes.qwen3vl_local_event_contract import prompt_for_task as pick_local_prompt
from openpi.tasks.robomme.pickxtimes.qwen3vl_local_event_contract import validate_compact_response as validate_pick_local
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import SYSTEM_PROMPT as SWING_SYSTEM_PROMPT
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import prompt_for_task as swing_prompt
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import validate_compact_response as validate_swing
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import SYSTEM_PROMPT as ORDER_SYSTEM_PROMPT
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import prompt_for_task as order_prompt
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import validate_compact_response as validate_order
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import SYSTEM_PROMPT as ORDER_LOCAL_SYSTEM_PROMPT
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import prompt_for_local_event as order_local_prompt
from openpi.tasks.robomme.videoplaceorder.qwen3vl_local_event_contract import validate_compact_response as validate_order_local
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import SYSTEM_PROMPT as VIDEO_SYSTEM_PROMPT
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import prompt_for_target
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import validate_compact_response
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import SYSTEM_PROMPT as UNMASK_SWAP_SYSTEM_PROMPT
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import prompt_for_local_event as unmask_swap_prompt
from openpi.tasks.robomme.videounmaskswap.qwen3vl_local_event_contract import validate_compact_response as validate_unmask_swap
from openpi.tasks.robomme.qwen3vl_unified_event_contract import SYSTEM_PROMPT as UNIFIED_SYSTEM_PROMPT
from openpi.tasks.robomme.qwen3vl_unified_event_contract import prompt_for_goal as unified_prompt
from openpi.tasks.robomme.qwen3vl_unified_event_contract import validate_compact_response as validate_unified
from openpi.tasks.shellgame.qwen3vl_sft_contract import SYSTEM_PROMPT as SHELLGAME_SYSTEM_PROMPT
from openpi.tasks.shellgame.qwen3vl_sft_contract import prompt_for_sample_type as shellgame_prompt
from openpi.tasks.shellgame.qwen3vl_sft_contract import validate_compact_response as validate_shellgame

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST_DIR = _ROOT / "artifacts/videounmask_qwen3vl_sft_seed260823"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=15)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int, default=720)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ManifestDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = None, shuffle_seed: int | None = None):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rows)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No samples in {path}")
        for row in self.rows:
            source = str(row.get("source", "videounmask"))
            if row.get("contract") == "unified_causal_event_v1":
                validate_unified(str(row["target"]))
            elif source == "videounmask":
                validate_compact_response(str(row["target"]))
            elif source == "videounmask_variable_demo":
                validate_compact_response(str(row["target"]))
            elif source == "videounmaskswap_local_event":
                validate_unmask_swap(str(row["target"]))
            elif source == "swingxtimes":
                validate_swing(
                    str(row["target"]), target_round_trips=int(row["target_round_trips"])
                )
            elif source == "pickxtimes":
                validate_pick(str(row["target"]), required_count=int(row["required_count"]))
            elif source == "pickxtimes_local_event":
                validate_pick_local(str(row["target"]))
            elif source == "videoplaceorder":
                validate_order(str(row["target"]))
            elif source == "videoplaceorder_local_event":
                validate_order_local(str(row["target"]))
            elif source == "shellgame":
                validate_shellgame(str(row["target"]))
            else:
                raise ValueError(f"Unsupported manifest source: {source}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class VideoSFTCollator:
    def __init__(self, processor: Any):
        self.processor = processor
        self._h5: dict[str, h5py.File] = {}

    def _h5_frames(self, row: dict[str, Any]) -> list[Image.Image]:
        path = str(row["h5_path"])
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        episode = self._h5[path][str(row["episode_name"])]
        indices = [int(value) for value in row["frame_indices"]]
        if len(indices) != 12 or min(indices) < 0:
            raise ValueError(f"HDF5 inputs must contain twelve causal frames: {indices}")
        if row.get("source") == "videounmask" and max(indices) > 65:
            raise ValueError(f"VideoUnmask input leaks beyond the demo prefix: {indices}")
        if row.get("source") in {
            "videounmask_variable_demo",
            "videounmaskswap_local_event",
            "videoplaceorder",
            "videoplaceorder_local_event",
        } and max(indices) >= int(row["demo_end"]):
            raise ValueError(f"VideoPlaceOrder input leaks into robot execution: {indices}")
        return [Image.fromarray(episode[f"timestep_{index}/obs/front_rgb"][()]).convert("RGB") for index in indices]

    @staticmethod
    def _shellgame_frames(row: dict[str, Any]) -> list[Image.Image]:
        indices = np.asarray(row["frame_indices"], dtype=np.int64)
        with np.load(row["trajectory_path"], allow_pickle=False) as trajectory:
            array = np.asarray(trajectory["third_person_images"][indices], dtype=np.uint8)
        frames = [Image.fromarray(frame).convert("RGB") for frame in array]
        # Keep one common temporal tensor shape per batch. Repeating endpoints
        # preserves the learned ten-frame event while matching VideoUnmask's 12.
        return [frames[0], *frames, frames[-1]]

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations, metadata, targets, source_ids = [], [], [], []
        for row in rows:
            source = str(row.get("source", "videounmask"))
            if row.get("contract") == "unified_causal_event_v1":
                frames = (
                    self._shellgame_frames(row)
                    if source == "shellgame_unified_replay"
                    else self._h5_frames(row)
                )
                system = UNIFIED_SYSTEM_PROMPT
                prompt = unified_prompt(
                    str(row["goal"]),
                    focus_entity=row.get("focus_entity"),
                    candidate_region_count=row.get("candidate_region_count"),
                )
                source_ids.append(
                    {
                        "videounmask_variable_demo": 5,
                        "videounmaskswap_local_event": 6,
                        "pickxtimes_local_event": 7,
                        "videoplaceorder_local_event": 8,
                        "shellgame_unified_replay": 1,
                    }[source]
                )
            elif source == "videounmask":
                frames = self._h5_frames(row)
                system = VIDEO_SYSTEM_PROMPT
                prompt = prompt_for_target(str(row["target_color"]))
                source_ids.append(0)
            elif source == "videounmask_variable_demo":
                frames = self._h5_frames(row)
                system = VIDEO_SYSTEM_PROMPT
                prompt = prompt_for_target(str(row["target_color"]))
                source_ids.append(5)
            elif source == "videounmaskswap_local_event":
                frames = self._h5_frames(row)
                system = UNMASK_SWAP_SYSTEM_PROMPT
                prompt = unmask_swap_prompt(int(row["num_containers"]))
                source_ids.append(6)
            elif source == "swingxtimes":
                frames = self._h5_frames(row)
                system = SWING_SYSTEM_PROMPT
                prompt = swing_prompt(
                    str(row["target_color"]), int(row["target_round_trips"])
                )
                source_ids.append(2)
            elif source == "pickxtimes":
                frames = self._h5_frames(row)
                system = PICK_SYSTEM_PROMPT
                prompt = pick_prompt(str(row["target_color"]), int(row["required_count"]))
                source_ids.append(3)
            elif source == "pickxtimes_local_event":
                frames = self._h5_frames(row)
                system = PICK_LOCAL_SYSTEM_PROMPT
                prompt = pick_local_prompt(str(row["target_color"]))
                source_ids.append(7)
            elif source == "videoplaceorder":
                frames = self._h5_frames(row)
                system = ORDER_SYSTEM_PROMPT
                prompt = order_prompt(str(row["target_color"]), int(row["ordinal"]))
                source_ids.append(4)
            elif source == "videoplaceorder_local_event":
                frames = self._h5_frames(row)
                system = ORDER_LOCAL_SYSTEM_PROMPT
                prompt = order_local_prompt()
                source_ids.append(8)
            else:
                frames = self._shellgame_frames(row)
                system = SHELLGAME_SYSTEM_PROMPT
                prompt = shellgame_prompt(str(row["sample_type"]))
                source_ids.append(1)
            target = str(row["target"])
            targets.append(target)
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {
                        "role": "user",
                        "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ]
            )
            metadata.append(
                VideoMetadata(
                    total_num_frames=12,
                    fps=10.0,
                    width=int(frames[0].width),
                    height=int(frames[0].height),
                    duration=1.2,
                    frames_indices=list(range(12)),
                )
            )
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=12,
            video_metadata=metadata,
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for index, target in enumerate(targets):
            target_ids = self.processor.tokenizer(target, add_special_tokens=False).input_ids
            input_ids = batch["input_ids"][index].tolist()
            starts = [
                start
                for start in range(len(input_ids) - len(target_ids) + 1)
                if input_ids[start : start + len(target_ids)] == target_ids
            ]
            if not starts:
                raise ValueError(f"Could not locate assistant target tokens: {target!r}")
            start = starts[-1]
            labels[index, start:] = batch["input_ids"][index, start:]
            labels[index, ~batch["attention_mask"][index].bool()] = -100
        batch["labels"] = labels
        batch["source_id"] = torch.tensor(source_ids, dtype=torch.long)
        return dict(batch)


@dataclass
class EvalMetrics:
    loss: float
    token_accuracy: float
    samples: int


@torch.no_grad()
def evaluate(accelerator: Accelerator, model: torch.nn.Module, loader: DataLoader, max_batches: int) -> EvalMetrics:
    model.eval()
    totals = torch.zeros(5, device=accelerator.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch.pop("source_id")
        outputs = model(**batch)
        labels = batch["labels"][:, 1:]
        predictions = outputs.logits[:, :-1].argmax(dim=-1)
        mask = labels != -100
        count = mask.sum().double()
        totals += torch.stack(
            (outputs.loss.detach().double() * count, count, ((predictions == labels) & mask).sum().double(), count, torch.tensor(batch["input_ids"].shape[0], device=totals.device, dtype=torch.float64))
        )
    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    return EvalMetrics(float((totals[0] / totals[1].clamp_min(1)).item()), float((totals[2] / totals[3].clamp_min(1)).item()), int(totals[4].item()))


def _save(accelerator: Accelerator, model: torch.nn.Module, processor: Any, path: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        print(f"saved adapter: {path}", flush=True)
    accelerator.wait_for_everyone()


def _install_non_tp_peft_load_compat() -> None:
    """Work around PEFT 0.19 importing a newer Transformers TP symbol.

    PEFT 0.19.1 imports ``EmbeddingParallel`` before it checks whether the
    model is tensor parallel. Transformers 4.57.1 does not expose that symbol,
    so an ordinary DDP adapter load fails even though none of the TP code is
    needed. Keep the upstream implementation for actual TP models and skip it
    only when no module has both TP metadata fields.
    """
    import peft.utils.save_and_load as peft_save_and_load
    import transformers.integrations.tensor_parallel as tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return
    original = peft_save_and_load._maybe_shard_state_dict_for_tp

    def non_tp_compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], adapter_name: str) -> None:
        has_tensor_parallel_layer = any(
            getattr(module, "_hf_tp_plan", None) is not None
            and getattr(module, "_hf_device_mesh", None) is not None
            for module in model.modules()
        )
        if not has_tensor_parallel_layer:
            return
        original(model, state_dict, adapter_name)

    peft_save_and_load._maybe_shard_state_dict_for_tp = non_tp_compatible


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, min_pixels=224**2, max_pixels=224**2)
    processor.video_processor.fps = None
    train_dataset = ManifestDataset(args.train_manifest, args.max_train_samples)
    val_dataset = ManifestDataset(args.val_manifest, args.max_val_samples, args.seed + 1)
    collator = VideoSFTCollator(processor)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, collate_fn=collator)

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.initial_adapter is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=("qkv", "proj", "linear_fc1", "linear_fc2", "q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        )
    else:
        _install_non_tp_peft_load_compat()
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    if accelerator.is_main_process:
        config = {**vars(args), "world_size": accelerator.num_processes, "global_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps * accelerator.num_processes, "train_samples": len(train_dataset), "val_samples": len(val_dataset)}
        config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path = args.output_dir / "metrics.jsonl"
    global_step = 0
    started = time.perf_counter()
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            batch.pop("source_id")
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                # ``accelerator.accumulate`` still executes this block for
                # every micro-batch. The wrapped optimizer suppresses its
                # update until gradients synchronize, but a manually stepped
                # scheduler does not. Advance LR exactly once per real update.
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step == 1 or global_step % args.logging_steps == 0:
                reduced_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                if accelerator.is_main_process:
                    elapsed = time.perf_counter() - started
                    record = {"step": global_step, "train_loss": float(reduced_loss.item()), "learning_rate": float(scheduler.get_last_lr()[0]), "elapsed_seconds": elapsed, "steps_per_second": global_step / max(elapsed, 1e-6)}
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(accelerator, model, val_loader, args.eval_batches)
                if accelerator.is_main_process:
                    record = {"step": global_step, **{f"val_{key}": value for key, value in asdict(metrics).items()}}
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save(accelerator, model, processor, args.output_dir / f"checkpoint-{global_step:06d}")
            if global_step >= args.max_steps:
                break
    _save(accelerator, model, processor, args.output_dir / "final")
    if accelerator.is_main_process:
        print(json.dumps({"status": "complete", "steps": global_step, "wall_time_seconds": time.perf_counter() - started, "final_adapter": str(args.output_dir / "final")}, sort_keys=True), flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
