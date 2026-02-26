"""
Compute Conditional Advantage Scores for UMI Data using Pi0Value Model

This script computes advantage scores for each sample in the UMI dataset using
the Pi0.6-style distributional value model. The advantage A = R - V is computed
where R is the Monte-Carlo return target and V is the model's value prediction.

The computed scores are written back into the UMI LeRobot dataset as new columns:
- advantage: the advantage score A = R - V
- value: the predicted value V
- return_target: the Monte-Carlo return target R

Usage:
    python examples/umi/rl/compute_advantage_scores.py \
        --dataset_path /path/to/umi_lerobot_dataset \
        --checkpoint_dir /path/to/value_model_checkpoint

Author: OpenPI Team
"""

import argparse
import csv
import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from openpi.models import model as _model
from openpi.models import pi0_value
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as transforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExtendedObservation:
    """Extended observation with value model required fields.
    
    This wraps the standard Observation and adds fields needed by Pi0Value:
    - step_index: current timestep in the episode
    - episode_T: terminal timestep index
    - reward: terminal reward (1 for success, 0 for failure)
    """
    base_obs: _model.Observation
    step_index: jnp.ndarray  # [B]
    episode_T: jnp.ndarray   # [B]
    reward: jnp.ndarray      # [B]
    
    # Proxy all attributes to base_obs
    def __getattr__(self, name):
        return getattr(self.base_obs, name)


def load_value_model(
    checkpoint_dir: str,
    config: pi0_value.Pi0ValueConfig,
) -> pi0_value.Pi0Value:
    """Load Pi0Value model from checkpoint.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory containing 'params' folder.
        config: Model configuration.
        
    Returns:
        Loaded Pi0Value model.
    """
    params_path = pathlib.Path(checkpoint_dir) / "params"
    if not params_path.exists():
        # Try without 'params' suffix
        params_path = pathlib.Path(checkpoint_dir)
    
    logger.info("Loading model parameters from: %s", params_path)
    
    # Restore parameters
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    
    # Create model from config and load parameters
    model = config.load(params)
    
    logger.info("Model loaded successfully")
    return model


def get_episode_info_from_dataset(dataset_path: str) -> dict:
    """Extract episode boundaries and success labels from LeRobot dataset.
    
    Args:
        dataset_path: Path to the LeRobot dataset.
        
    Returns:
        Dict with episode_ends, episode_starts, and success labels.
    """
    from openpi.training import lerobot_dataset
    
    meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_path)
    
    # Get episode ends
    episode_ends = np.array(meta.episode_ends)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    # Get episode lengths
    episode_lengths = episode_ends - episode_starts
    
    logger.info("Found %d episodes", len(episode_ends))
    logger.info("Total samples: %d", episode_ends[-1])
    logger.info("Episode lengths: min=%d, max=%d, mean=%.1f", 
                episode_lengths.min(), episode_lengths.max(), episode_lengths.mean())
    
    return {
        "episode_ends": episode_ends,
        "episode_starts": episode_starts,
        "episode_lengths": episode_lengths,
        "num_episodes": len(episode_ends),
        "total_samples": int(episode_ends[-1]),
    }


def compute_step_and_episode_info(
    global_index: int,
    episode_info: dict,
) -> tuple[int, int, int]:
    """Compute step_index and episode_T for a given global sample index.
    
    Args:
        global_index: The global index of the sample in the dataset.
        episode_info: Episode boundary information.
        
    Returns:
        Tuple of (step_index, episode_T, episode_idx).
    """
    episode_ends = episode_info["episode_ends"]
    episode_starts = episode_info["episode_starts"]
    
    # Find which episode this sample belongs to
    episode_idx = np.searchsorted(episode_ends, global_index, side='right')
    
    # Compute step index within episode
    step_index = global_index - episode_starts[episode_idx]
    
    # Episode T is the last step index in the episode
    episode_T = (episode_ends[episode_idx] - episode_starts[episode_idx]) - 1
    
    return step_index, episode_T, episode_idx


def compute_advantages_for_dataset(
    model: pi0_value.Pi0Value,
    dataset_path: str,
    train_config: _config.TrainConfig,
    batch_size: int = 8,
    success_labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute advantage scores for all samples in the dataset.
    
    Args:
        model: The Pi0Value model.
        dataset_path: Path to the LeRobot dataset.
        train_config: Training configuration for data loading.
        batch_size: Batch size for processing.
        success_labels: Optional per-episode success labels (1=success, 0=fail).
                       If None, assumes all episodes are failures (reward=0).
    
    Returns:
        Dict containing advantages, returns, values for each sample.
    """
    from openpi.training import lerobot_dataset
    import torch
    
    # Get episode info
    episode_info = get_episode_info_from_dataset(dataset_path)
    num_samples = episode_info["total_samples"]
    num_episodes = episode_info["num_episodes"]
    
    # Default: all episodes failed (conservative assumption)
    if success_labels is None:
        logger.warning("No success labels provided. Assuming all episodes failed (reward=0).")
        success_labels = np.zeros(num_episodes)
    
    # Create dataset
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_path)
    dataset = lerobot_dataset.LeRobotDataset(
        dataset_path,
        delta_timestamps={
            key: [t / meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    
    # Apply transforms
    transforms_list = []
    if data_config.prompt_from_task:
        transforms_list.append(transforms.PromptFromLeRobotTask(meta.tasks))
    transforms_list.extend(data_config.data_transforms.inputs)
    transforms_list.extend(data_config.model_transforms.inputs)
    
    dataset = _data_loader.TransformedDataset(dataset, transforms_list)
    
    # Create PyTorch data loader directly (without openpi wrapper to avoid drop_last=True)
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,  # Important: keep all samples
        collate_fn=_data_loader._collate_fn,
    )
    
    # Storage for results
    all_advantages = []
    all_returns = []
    all_values = []
    all_indices = []
    
    rng = jax.random.key(42)
    sample_idx = 0
    
    logger.info("Processing %d samples in batches of %d...", num_samples, batch_size)
    
    for batch_data in tqdm(torch_loader, desc="Computing advantages"):
        # Get actual batch size (may be smaller for last batch)
        if isinstance(batch_data, dict):
            first_value = next(iter(batch_data.values()))
            if isinstance(first_value, dict):
                batch_size_actual = len(next(iter(first_value.values())))
            else:
                batch_size_actual = len(first_value)
        elif isinstance(batch_data, (list, tuple)):
            batch_size_actual = len(batch_data[0]) if hasattr(batch_data[0], '__len__') else 1
        else:
            batch_size_actual = batch_size
        
        # Prepare batch indices
        batch_indices = np.arange(sample_idx, sample_idx + batch_size_actual)
        
        # Compute step_index, episode_T, reward for each sample in batch
        step_indices = []
        episode_Ts = []
        rewards = []
        
        for idx in batch_indices:
            if idx >= num_samples:
                break
            step_idx, ep_T, ep_idx = compute_step_and_episode_info(idx, episode_info)
            step_indices.append(step_idx)
            episode_Ts.append(ep_T)
            # Reward is 1 if success, 0 if failure (at terminal step)
            rewards.append(float(success_labels[ep_idx]))
        
        actual_batch = len(step_indices)
        if actual_batch == 0:
            break
            
        step_indices = jnp.array(step_indices)
        episode_Ts = jnp.array(episode_Ts)
        rewards = jnp.array(rewards)
        
        # Convert batch data to observation dict
        if isinstance(batch_data, tuple):
            obs_data, _ = batch_data  # Ignore actions
        else:
            obs_data = batch_data
        
        # Convert to JAX arrays
        obs_data = jax.tree.map(
            lambda x: jnp.array(x.numpy()) if hasattr(x, 'numpy') else jnp.array(x), 
            obs_data
        )
        
        # Create observation
        observation = _model.Observation.from_dict(obs_data)
        
        # Create extended observation with value model fields
        extended_obs = ExtendedObservation(
            base_obs=observation,
            step_index=step_indices[:actual_batch],
            episode_T=episode_Ts[:actual_batch],
            reward=rewards[:actual_batch],
        )
        
        # Compute advantage
        rng, step_rng = jax.random.split(rng)
        try:
            A_norm, R_norm, V_norm = model.compute_advantage(
                step_rng,
                extended_obs,
                train=False,
                value_reduce="expectation",
                stopgrad_value=True,
            )
            
            all_advantages.append(np.array(A_norm))
            all_returns.append(np.array(R_norm))
            all_values.append(np.array(V_norm))
            all_indices.extend(batch_indices[:actual_batch].tolist())
            
        except (RuntimeError, ValueError) as e:
            logger.error("Error processing batch at index %d: %s", sample_idx, e)
            # Fill with NaN for failed batches
            all_advantages.append(np.full(actual_batch, np.nan))
            all_returns.append(np.full(actual_batch, np.nan))
            all_values.append(np.full(actual_batch, np.nan))
            all_indices.extend(batch_indices[:actual_batch].tolist())
        
        sample_idx += batch_size_actual
    
    # Concatenate results
    results = {
        "advantages": np.concatenate(all_advantages),
        "returns": np.concatenate(all_returns),
        "values": np.concatenate(all_values),
        "sample_indices": np.array(all_indices),
        "episode_info": episode_info,
    }
    
    logger.info("Processed %d samples", len(results['advantages']))
    logger.info("Advantage stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                results['advantages'].mean(), results['advantages'].std(),
                results['advantages'].min(), results['advantages'].max())
    
    return results


def write_scores_to_dataset(
    dataset_path: str,
    advantages: np.ndarray,
    returns: np.ndarray,
    values: np.ndarray,
    episode_info: dict,
):
    """Write advantage scores back into the LeRobot dataset parquet files.
    
    Args:
        dataset_path: Path to the LeRobot dataset.
        advantages: Array of advantage scores for all samples.
        returns: Array of return targets for all samples.
        values: Array of value predictions for all samples.
        episode_info: Episode boundary information.
    """
    dataset_path = pathlib.Path(dataset_path)
    data_dir = dataset_path / "data"
    
    episode_ends = episode_info["episode_ends"]
    episode_starts = episode_info["episode_starts"]
    num_episodes = episode_info["num_episodes"]
    
    logger.info("Writing advantage scores to dataset parquet files...")
    
    for ep_idx in tqdm(range(num_episodes), desc="Writing to parquet"):
        # Find the parquet file for this episode
        chunk_idx = ep_idx // 1000
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        
        if not parquet_path.exists():
            logger.warning("Parquet file not found: %s", parquet_path)
            continue
        
        # Read existing parquet file
        table = pq.read_table(parquet_path)
        
        # Get indices for this episode
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        
        # Extract scores for this episode
        ep_advantages = advantages[start_idx:end_idx].astype(np.float32)
        ep_returns = returns[start_idx:end_idx].astype(np.float32)
        ep_values = values[start_idx:end_idx].astype(np.float32)
        
        # Verify lengths match
        if len(ep_advantages) != table.num_rows:
            logger.warning(
                "Length mismatch for episode %d: scores=%d, parquet=%d",
                ep_idx, len(ep_advantages), table.num_rows
            )
            # Pad or truncate if necessary
            if len(ep_advantages) < table.num_rows:
                pad_len = table.num_rows - len(ep_advantages)
                ep_advantages = np.concatenate([ep_advantages, np.full(pad_len, np.nan)])
                ep_returns = np.concatenate([ep_returns, np.full(pad_len, np.nan)])
                ep_values = np.concatenate([ep_values, np.full(pad_len, np.nan)])
            else:
                ep_advantages = ep_advantages[:table.num_rows]
                ep_returns = ep_returns[:table.num_rows]
                ep_values = ep_values[:table.num_rows]
        
        # Add new columns to table
        new_table = table.append_column("advantage", pa.array(ep_advantages))
        new_table = new_table.append_column("value", pa.array(ep_values))
        new_table = new_table.append_column("return_target", pa.array(ep_returns))
        
        # Write back to parquet
        pq.write_table(new_table, parquet_path)
    
    # Update info.json to include new columns in features
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        # Add new features if not already present
        if "features" in info:
            if "advantage" not in info["features"]:
                info["features"]["advantage"] = {"dtype": "float32", "shape": [1]}
            if "value" not in info["features"]:
                info["features"]["value"] = {"dtype": "float32", "shape": [1]}
            if "return_target" not in info["features"]:
                info["features"]["return_target"] = {"dtype": "float32", "shape": [1]}
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
    
    logger.info("Successfully wrote advantage scores to %d episodes", num_episodes)


def save_results_to_file(results: dict, output_path: str):
    """Save computed advantage scores to external files (optional backup)."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as npz
    np.savez(
        output_path,
        advantages=results["advantages"],
        returns=results["returns"],
        values=results["values"],
        sample_indices=results["sample_indices"],
    )
    
    logger.info("Results saved to: %s", output_path)
    
    # Also save a summary CSV
    summary_path = output_path.with_suffix(".csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "advantage", "return", "value"])
        for i, idx in enumerate(results["sample_indices"]):
            writer.writerow([idx, results["advantages"][i], results["returns"][i], results["values"][i]])
    
    logger.info("Summary CSV saved to: %s", summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute conditional advantage scores for UMI data and write back to dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the UMI LeRobot dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the Pi0Value model checkpoint directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_umi",
        help="Training config name (for data transforms)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional: also save results to external file (npz/csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--task_max_steps",
        type=int,
        default=1500,
        help="Maximum steps for task normalization",
    )
    parser.add_argument(
        "--success_file",
        type=str,
        default=None,
        help="Optional CSV/NPZ file with per-episode success labels",
    )
    parser.add_argument(
        "--all_success",
        action="store_true",
        help="Treat all episodes as successful (reward=1)",
    )
    parser.add_argument(
        "--all_failure",
        action="store_true", 
        help="Treat all episodes as failures (reward=0)",
    )
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="Do not write scores back to dataset (only compute and optionally save to file)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Compute Conditional Advantage Scores")
    logger.info("=" * 60)
    logger.info("Dataset: %s", args.dataset_path)
    logger.info("Checkpoint: %s", args.checkpoint_dir)
    logger.info("Config: %s", args.config_name)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Task max steps: %d", args.task_max_steps)
    logger.info("Write to dataset: %s", "No" if args.no_write else "Yes")
    if args.output_path:
        logger.info("Output file: %s", args.output_path)
    logger.info("=" * 60)
    
    # Get training config for data transforms
    train_config = _config.get_config(args.config_name)
    
    # Override dataset path in config
    if hasattr(train_config.data, 'repo_id'):
        # Update repo_id to use the provided dataset path
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, repo_id=args.dataset_path),
        )
    
    # Get episode info first
    episode_info = get_episode_info_from_dataset(args.dataset_path)
    num_episodes = episode_info["num_episodes"]
    
    # Determine success labels
    if args.all_success:
        success_labels = np.ones(num_episodes)
        logger.info("Using all_success: treating all episodes as successful")
    elif args.all_failure:
        success_labels = np.zeros(num_episodes)
        logger.info("Using all_failure: treating all episodes as failures")
    elif args.success_file:
        # Load success labels from file
        if args.success_file.endswith(".npz"):
            data = np.load(args.success_file)
            success_labels = data["success_labels"]
        elif args.success_file.endswith(".csv"):
            with open(args.success_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                success_labels = np.array([int(row["success"]) for row in reader])
        else:
            success_labels = np.loadtxt(args.success_file)
        logger.info("Loaded success labels from: %s", args.success_file)
        logger.info("Success rate: %.2f%%", success_labels.mean() * 100)
    else:
        success_labels = None
        logger.info("No success labels provided, assuming all failures")
    
    # Create value model config
    value_config = pi0_value.Pi0ValueConfig(
        action_dim=train_config.model.action_dim,
        action_horizon=train_config.model.action_horizon,
        paligemma_variant=getattr(train_config.model, "paligemma_variant", "gemma_2b"),
        task_max_steps=args.task_max_steps,
    )
    
    # Load model
    model = load_value_model(args.checkpoint_dir, value_config)
    
    # Compute advantages
    results = compute_advantages_for_dataset(
        model=model,
        dataset_path=args.dataset_path,
        train_config=train_config,
        batch_size=args.batch_size,
        success_labels=success_labels,
    )
    
    # Write scores back to dataset
    if not args.no_write:
        write_scores_to_dataset(
            dataset_path=args.dataset_path,
            advantages=results["advantages"],
            returns=results["returns"],
            values=results["values"],
            episode_info=results["episode_info"],
        )
    
    # Optionally save to external file
    if args.output_path:
        save_results_to_file(results, args.output_path)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
