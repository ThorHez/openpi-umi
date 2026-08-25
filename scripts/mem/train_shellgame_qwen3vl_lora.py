# ruff: noqa: E402
"""LoRA SFT for Qwen3-VL ShellGame reveal/exchange grounding.

Run only in the isolated Qwen environment.  Raw videos are decoded lazily from
the manifest's NPZ paths, and loss is applied only to the compact assistant JSON.
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
import numpy as np
from peft import LoraConfig
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

from openpi.tasks.shellgame.qwen3vl_sft_contract import SAMPLE_TYPES
from openpi.tasks.shellgame.qwen3vl_sft_contract import SYSTEM_PROMPT
from openpi.tasks.shellgame.qwen3vl_sft_contract import prompt_for_sample_type
from openpi.tasks.shellgame.qwen3vl_sft_contract import validate_compact_response

DEFAULT_MODEL = Path("/data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct")
DEFAULT_MANIFEST_DIR = _ROOT / "artifacts/shellgame_qwen3vl_gt_event_sft_v1"
DEFAULT_OUTPUT_DIR = _ROOT / "checkpoints/qwen3vl_shellgame_gt_event_lora_v1"
TYPE_TO_ID = {name: index for index, name in enumerate(SAMPLE_TYPES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_MANIFEST_DIR / "val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=260825)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ManifestDataset(Dataset):
    def __init__(self, path: Path, *, max_samples: int | None = None, shuffle_seed: int | None = None):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rows)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No samples loaded from {path}")
        for row in self.rows:
            if str(row["sample_type"]) not in TYPE_TO_ID:
                raise ValueError(f"Invalid sample type in {path}: {row['sample_type']!r}")
            validate_compact_response(str(row["target"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class QwenVideoSFTCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    @staticmethod
    def _frames(row: dict[str, Any]) -> list[Image.Image]:
        indices = np.asarray(row["frame_indices"], dtype=np.int64)
        if indices.shape != (10,) or not np.all(np.diff(indices) == 1):
            raise ValueError(f"Expected ten consecutive frame indices, got {indices.tolist()}")
        with np.load(row["trajectory_path"], allow_pickle=False) as trajectory:
            frames = np.asarray(trajectory["third_person_images"][indices], dtype=np.uint8)
        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations = []
        video_metadata = []
        targets = []
        for row in rows:
            frames = self._frames(row)
            target = str(row["target"])
            targets.append(target)
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames},
                            {"type": "text", "text": prompt_for_sample_type(str(row["sample_type"]))},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ]
            )
            video_metadata.append(
                VideoMetadata(
                    total_num_frames=10,
                    fps=10.0,
                    width=int(frames[0].width),
                    height=int(frames[0].height),
                    duration=1.0,
                    frames_indices=list(range(10)),
                )
            )
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=10,
            video_metadata=video_metadata,
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, target in enumerate(targets):
            target_ids = self.processor.tokenizer(target, add_special_tokens=False).input_ids
            input_ids = batch["input_ids"][row_index].tolist()
            matches = [
                start
                for start in range(len(input_ids) - len(target_ids) + 1)
                if input_ids[start : start + len(target_ids)] == target_ids
            ]
            if not matches:
                raise ValueError(f"Could not locate assistant target tokens: target={target!r}")
            # The shared exchange prompt explicitly enumerates the two reject
            # JSON objects.  The assistant response is always the final match.
            start = matches[-1]
            valid = batch["attention_mask"][row_index].bool()
            labels[row_index, start:] = batch["input_ids"][row_index, start:]
            labels[row_index, ~valid] = -100
        batch["labels"] = labels
        batch["sample_type_id"] = torch.tensor([TYPE_TO_ID[str(row["sample_type"])] for row in rows], dtype=torch.long)
        return dict(batch)


@dataclass
class EvalMetrics:
    loss: float
    token_accuracy: float
    samples: int


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    max_batches: int,
) -> EvalMetrics:
    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    loss_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    correct = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    tokens = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    samples = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch.pop("sample_type_id")
        outputs = model(**batch)
        labels = batch["labels"][:, 1:]
        predictions = outputs.logits[:, :-1].argmax(dim=-1)
        mask = labels != -100
        token_count = mask.sum()
        loss_sum += outputs.loss.detach().double() * token_count.double()
        loss_count += token_count.double()
        correct += ((predictions == labels) & mask).sum().double()
        tokens += token_count.double()
        samples += batch["input_ids"].shape[0]
    packed = torch.stack((loss_sum, loss_count, correct, tokens, samples))
    packed = accelerator.reduce(packed, reduction="sum")
    model.train()
    return EvalMetrics(
        loss=float((packed[0] / packed[1].clamp_min(1)).item()),
        token_accuracy=float((packed[2] / packed[3].clamp_min(1)).item()),
        samples=int(packed[4].item()),
    )


def _save_adapter(accelerator: Accelerator, model: torch.nn.Module, processor: Any, path: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        print(f"saved adapter: {path}", flush=True)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        # This loop defines max_steps in optimizer steps.  Letting Accelerate
        # scale scheduler.step() by world size would finish warmup/decay N times
        # too early under DDP.
        step_scheduler_with_optimizer=False,
    )
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
    )
    processor.video_processor.fps = None
    train_dataset = ManifestDataset(args.train_manifest, max_samples=args.max_train_samples)
    val_dataset = ManifestDataset(
        args.val_manifest,
        max_samples=args.max_val_samples,
        shuffle_seed=args.seed + 1,
    )
    collator = QwenVideoSFTCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            # Vision temporal/spatial attention and MLP, including the merger.
            "qkv",
            "proj",
            "linear_fc1",
            "linear_fc2",
            # A light language-side adaptation for the compact output contract.
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
    )
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        config = vars(args).copy()
        config.update(
            {
                "model_path": str(args.model_path),
                "train_manifest": str(args.train_manifest),
                "val_manifest": str(args.val_manifest),
                "output_dir": str(args.output_dir),
                "world_size": accelerator.num_processes,
                "global_batch_size": (
                    args.per_device_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
                ),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            }
        )
        (args.output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metrics_path = args.output_dir / "metrics.jsonl"
    global_step = 0
    micro_step = 0
    started = time.perf_counter()
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            micro_step += 1
            batch.pop("sample_type_id")
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step % args.logging_steps == 0 or global_step == 1:
                gathered_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                if accelerator.is_main_process:
                    elapsed = time.perf_counter() - started
                    record = {
                        "step": global_step,
                        "train_loss": float(gathered_loss.item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "elapsed_seconds": elapsed,
                        "steps_per_second": global_step / max(elapsed, 1e-6),
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(accelerator, model, val_loader, max_batches=args.eval_batches)
                if accelerator.is_main_process:
                    record = {"step": global_step, **{f"val_{k}": v for k, v in asdict(metrics).items()}}
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_adapter(
                    accelerator,
                    model,
                    processor,
                    args.output_dir / f"checkpoint-{global_step:06d}",
                )
            if global_step >= args.max_steps:
                break
    _save_adapter(accelerator, model, processor, args.output_dir / "final")
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "complete",
                    "steps": global_step,
                    "wall_time_seconds": time.perf_counter() - started,
                    "final_adapter": str(args.output_dir / "final"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
