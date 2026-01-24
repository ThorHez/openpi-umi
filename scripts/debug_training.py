"""Debug script to save intermediate training values for analysis.

This script will run a few training steps and save actions, squared_error, 
and squared_error_masked for debugging purposes.
"""

import pathlib
import pickle
import logging

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.shared import array_typing as at

logging.basicConfig(level=logging.INFO)


@at.typecheck
def debug_step(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[at.Float[at.Array, ""], dict]:
    """Run one step and collect debug info."""
    observation, actions = batch
    
    # Call compute_loss with debug flag
    loss_per_timestep, debug_info = model.compute_loss(
        rng, observation, actions, train=False, return_debug_info=True
    )
    loss = jnp.mean(loss_per_timestep)
    
    return loss, debug_info


def main(
    config_name: str,
    num_steps: int = 10,
    output_dir: str = "/root/openpi/debug_output",
):
    """Run debug training steps and save intermediate values.
    
    Args:
        config_name: Name of the training config to use
        num_steps: Number of steps to run
        output_dir: Directory to save debug outputs
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = _config.get_config(config_name)
    
    # Initialize model
    rng = jax.random.key(config.seed)
    init_rng, data_rng = jax.random.split(rng)
    
    logging.info("Creating model...")
    model = config.model.create(init_rng)
    model.eval()
    
    # Create data loader
    logging.info("Creating data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=False,
        num_batches=num_steps,
        skip_norm_stats=False,
        framework="jax",
    )
    
    # Collect debug info
    debug_data = []
    
    logging.info(f"Running {num_steps} debug steps...")
    for step_idx, batch in enumerate(data_loader):
        step_rng = jax.random.fold_in(data_rng, step_idx)
        
        logging.info(f"Step {step_idx + 1}/{num_steps}")
        loss, debug_info = debug_step(model, step_rng, batch)
        
        # Convert to numpy for saving
        debug_entry = {
            "step": step_idx,
            "loss": float(loss),
            "actions": np.array(debug_info["actions"]),
            "squared_error": np.array(debug_info["squared_error"]),
            "squared_error_masked": np.array(debug_info["squared_error_masked"]) if debug_info["squared_error_masked"] is not None else None,
            "v_t": np.array(debug_info["v_t"]),
            "u_t": np.array(debug_info["u_t"]),
        }
        
        # Print statistics
        logging.info(f"  Loss: {debug_entry['loss']:.6f}")
        logging.info(f"  Actions shape: {debug_entry['actions'].shape}")
        logging.info(f"  Actions mean: {debug_entry['actions'].mean():.6f}")
        logging.info(f"  Actions std: {debug_entry['actions'].std():.6f}")
        logging.info(f"  Actions range: [{debug_entry['actions'].min():.6f}, {debug_entry['actions'].max():.6f}]")
        logging.info(f"  Squared error mean: {debug_entry['squared_error'].mean():.6f}")
        
        debug_data.append(debug_entry)
        
        if step_idx + 1 >= num_steps:
            break
    
    # Save all debug data
    output_file = output_path / f"debug_data_{config_name}.pkl"
    logging.info(f"Saving debug data to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(debug_data, f)
    
    # Save summary statistics
    summary_file = output_path / f"debug_summary_{config_name}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Debug Summary for {config_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, entry in enumerate(debug_data):
            f.write(f"Step {i}:\n")
            f.write(f"  Loss: {entry['loss']:.6f}\n")
            f.write(f"  Actions shape: {entry['actions'].shape}\n")
            f.write(f"  Actions stats:\n")
            f.write(f"    Mean: {entry['actions'].mean():.6f}\n")
            f.write(f"    Std: {entry['actions'].std():.6f}\n")
            f.write(f"    Min: {entry['actions'].min():.6f}\n")
            f.write(f"    Max: {entry['actions'].max():.6f}\n")
            f.write(f"  Squared error stats:\n")
            f.write(f"    Mean: {entry['squared_error'].mean():.6f}\n")
            f.write(f"    Std: {entry['squared_error'].std():.6f}\n")
            if entry['squared_error_masked'] is not None:
                f.write(f"  Masked squared error stats:\n")
                f.write(f"    Mean: {entry['squared_error_masked'].mean():.6f}\n")
                f.write(f"    Std: {entry['squared_error_masked'].std():.6f}\n")
            f.write("\n")
    
    logging.info(f"Summary saved to {summary_file}")
    logging.info(f"\nTo analyze the data in Python:")
    logging.info(f"  import pickle")
    logging.info(f"  with open('{output_file}', 'rb') as f:")
    logging.info(f"      data = pickle.load(f)")
    logging.info(f"  # data is a list of dicts with keys: step, loss, actions, squared_error, etc.")


if __name__ == "__main__":
    tyro.cli(main)


