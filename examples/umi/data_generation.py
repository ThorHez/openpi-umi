"""
Task Description Augmentation for UMI Dataset using DeepSeek API

This script reads task descriptions from a UMI LeRobot dataset, calls the DeepSeek
model to generate diverse paraphrased versions, and writes them back into the dataset.
This improves VLA (Vision-Language-Action) instruction-following ability during training
by exposing the model to varied but semantically equivalent task descriptions.

For each original task, the script generates N paraphrased versions while preserving
the core semantic meaning. The paraphrased tasks are added to the dataset and randomly
assigned to episodes so that the same manipulation task can be described in different ways.

Usage:
    python examples/umi/data_generation.py \
        --dataset_path /path/to/umi_lerobot_dataset \
        --api_key YOUR_DEEPSEEK_API_KEY \
        --num_augments 5

Author: OpenPI Team
"""

import argparse
import json
import logging
import pathlib
import random
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

os.environ["DEEPSEEK_API_KEY"] = "sk-1fd70bfefebf430a859af1feda0908cf"

# ─────────────────────────── Dataset I/O ────────────────────────────

def load_tasks(dataset_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load tasks from meta/tasks.jsonl."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def save_tasks(dataset_path: pathlib.Path, tasks: list[dict[str, Any]]):
    """Save tasks to meta/tasks.jsonl."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def load_episodes(dataset_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load episodes from meta/episodes.jsonl."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    if episodes_path.exists():
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
    return episodes


def save_episodes(dataset_path: pathlib.Path, episodes: list[dict[str, Any]]):
    """Save episodes to meta/episodes.jsonl."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def load_info(dataset_path: pathlib.Path) -> dict:
    """Load info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_info(dataset_path: pathlib.Path, info: dict):
    """Save info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


# ─────────────────────── DeepSeek Paraphrase ────────────────────────

SYSTEM_PROMPT = """\
You are an expert at paraphrasing robotic manipulation task descriptions.

Rules:
1. Keep the EXACT same semantic meaning — the robot must perform the same actions on the same objects.
2. Vary sentence structure, vocabulary, and phrasing style.
3. You may change between imperative ("pick up …"), descriptive ("the robot should …"), \
   step-by-step ("first … then …"), or casual ("grab the … and put it …") styles.
4. Keep each paraphrase concise (1–2 sentences).
5. Do NOT add extra actions or remove any existing actions.
6. Output ONLY the paraphrased descriptions, one per line, without numbering or bullet points.
"""


def paraphrase_task_deepseek(
    client: "OpenAI",
    task_description: str,
    num_augments: int = 5,
    model: str = "deepseek-chat",
    temperature: float = 0.9,
    max_retries: int = 3,
) -> list[str]:
    """Call DeepSeek to generate paraphrased task descriptions.

    Args:
        client: OpenAI-compatible client (pointing at DeepSeek endpoint).
        task_description: The original task description.
        num_augments: Number of paraphrased versions to generate.
        model: DeepSeek model name.
        temperature: Sampling temperature for diversity.
        max_retries: Number of retries on failure.

    Returns:
        List of paraphrased task descriptions (may include duplicates removed).
    """
    user_prompt = (
        f"Original task description:\n\"{task_description}\"\n\n"
        f"Please generate {num_augments} different paraphrased versions of this robotic "
        f"manipulation task description. Output one paraphrase per line."
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
            )

            raw_text = response.choices[0].message.content.strip()
            # Parse: one paraphrase per line, skip empty lines
            paraphrases = [
                line.strip().strip('"').strip("'")
                for line in raw_text.splitlines()
                if line.strip() and line.strip() != task_description
            ]
            # Remove numbering prefixes like "1. ", "1) ", "- "
            cleaned = []
            for p in paraphrases:
                # Strip leading numbers/bullets
                for prefix in ["- ", "• "]:
                    if p.startswith(prefix):
                        p = p[len(prefix):]
                if len(p) > 2 and p[0].isdigit() and p[1] in ".)" :
                    p = p[2:].strip()
                if len(p) > 3 and p[:2].isdigit() and p[2] in ".)" :
                    p = p[3:].strip()
                if p:
                    cleaned.append(p)

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for p in cleaned:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(p)

            return unique[:num_augments]

        except (KeyError, AttributeError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning("DeepSeek API call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error("Failed to paraphrase task after %d retries: %s", max_retries, task_description)
    return []


# ──────────────────── Write Back to Dataset ─────────────────────────

def augment_dataset(
    dataset_path: pathlib.Path,
    original_tasks: list[dict[str, Any]],
    augmented_map: dict[int, list[str]],
    seed: int = 42,
):
    """Write augmented tasks back into the LeRobot dataset.

    For each episode, the original task_index in every frame is randomly replaced
    with one of the augmented variants (including the original).

    Args:
        dataset_path: Path to the LeRobot dataset.
        original_tasks: Original task list from tasks.jsonl.
        augmented_map: Mapping from original task_index to list of new paraphrased texts.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    # ── 1. Build new task list ──────────────────────────────────────
    # Keep original tasks, append new ones
    new_tasks = list(original_tasks)
    next_idx = max(t["task_index"] for t in original_tasks) + 1

    # old_task_index -> list of ALL valid task_indices (including original)
    index_group: dict[int, list[int]] = {}
    for task in original_tasks:
        index_group[task["task_index"]] = [task["task_index"]]

    for old_idx, paraphrases in augmented_map.items():
        for text in paraphrases:
            new_tasks.append({"task_index": next_idx, "task": text})
            index_group[old_idx].append(next_idx)
            next_idx += 1

    save_tasks(dataset_path, new_tasks)
    logger.info("Saved %d tasks (%d original + %d augmented)",
                len(new_tasks), len(original_tasks), len(new_tasks) - len(original_tasks))

    # ── 2. Update episodes.jsonl ────────────────────────────────────
    episodes = load_episodes(dataset_path)
    for ep in episodes:
        # Expand task list to include augmented variants
        old_task_list = ep.get("tasks", [])
        if isinstance(old_task_list, list) and old_task_list:
            # Find matching text entries
            new_task_list = set()
            for t_text in old_task_list:
                new_task_list.add(t_text)
                # Find original index for this text
                for task in original_tasks:
                    if task["task"] == t_text:
                        for aug_text in augmented_map.get(task["task_index"], []):
                            new_task_list.add(aug_text)
            ep["tasks"] = sorted(new_task_list)
    save_episodes(dataset_path, episodes)

    # ── 3. Update parquet files: randomly swap task_index ───────────
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info("Updating task_index in %d parquet files...", len(parquet_files))

    for pq_path in tqdm(parquet_files, desc="Updating parquet"):
        table = pq.read_table(pq_path)
        if "task_index" not in table.column_names:
            continue

        old_indices = table.column("task_index").to_pylist()
        new_indices = []
        for idx in old_indices:
            candidates = index_group.get(idx, [idx])
            new_indices.append(rng.choice(candidates))

        # Replace column
        col_idx = table.column_names.index("task_index")
        table = table.set_column(col_idx, "task_index", pa.array(new_indices, type=pa.int64()))
        pq.write_table(table, pq_path)

    logger.info("Dataset augmentation complete.")


# ──────────────────────────── Main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Augment UMI dataset task descriptions using DeepSeek for improved instruction following"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the UMI LeRobot dataset",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.deepseek.com",
        help="DeepSeek API base URL",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="DeepSeek model name",
    )
    parser.add_argument(
        "--num_augments", type=int, default=5,
        help="Number of paraphrased variants per task",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature for paraphrase diversity",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible task_index assignment",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only generate paraphrases and print them, do not modify the dataset",
    )

    args = parser.parse_args()

    # ── Resolve API key ────────────────────────────────────────────
    import os
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("Provide --api_key or set the DEEPSEEK_API_KEY environment variable.")

    if OpenAI is None:
        parser.error("openai package is required. Install with: pip install openai")

    dataset_path = pathlib.Path(args.dataset_path)
    if not dataset_path.exists():
        parser.error(f"Dataset path does not exist: {dataset_path}")

    # ── Load existing tasks ────────────────────────────────────────
    original_tasks = load_tasks(dataset_path)
    logger.info("Loaded %d tasks from dataset", len(original_tasks))
    for t in original_tasks:
        logger.info("  [%d] %s", t["task_index"], t["task"])

    # ── Call DeepSeek for each task ────────────────────────────────
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    augmented_map: dict[int, list[str]] = {}
    for task in original_tasks:
        task_idx = task["task_index"]
        task_text = task["task"]

        logger.info("Paraphrasing task [%d]: \"%s\"", task_idx, task_text)
        paraphrases = paraphrase_task_deepseek(
            client,
            task_text,
            num_augments=args.num_augments,
            model=args.model,
            temperature=args.temperature,
        )

        if paraphrases:
            augmented_map[task_idx] = paraphrases
            logger.info("  Generated %d paraphrases:", len(paraphrases))
            for i, p in enumerate(paraphrases):
                logger.info("    %d. %s", i + 1, p)
        else:
            logger.warning("  No paraphrases generated for task [%d]", task_idx)

    total_augmented = sum(len(v) for v in augmented_map.values())
    logger.info("Total: %d original tasks → %d new paraphrases",
                len(original_tasks), total_augmented)

    # ── Write back or dry-run ──────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run mode — dataset was NOT modified.")
        return

    augment_dataset(
        dataset_path=dataset_path,
        original_tasks=original_tasks,
        augmented_map=augmented_map,
        seed=args.seed,
    )

    logger.info("=" * 60)
    logger.info("Done! Dataset has been augmented with paraphrased task descriptions.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
