"""Analyze debug data collected during training.

This script loads the debug data saved by debug_training.py and provides
analysis and visualization.
"""

import pickle
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_debug_data(filepath: str):
    """Load debug data from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def analyze_actions(data):
    """Analyze action statistics across all steps."""
    print("\n" + "=" * 80)
    print("ACTION ANALYSIS")
    print("=" * 80)
    
    all_actions = []
    for entry in data:
        all_actions.append(entry['actions'])
    
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\nTotal samples: {all_actions.shape[0]}")
    print(f"Action shape: {all_actions.shape}")
    print(f"\nOverall statistics:")
    print(f"  Mean: {all_actions.mean():.6f}")
    print(f"  Std: {all_actions.std():.6f}")
    print(f"  Min: {all_actions.min():.6f}")
    print(f"  Max: {all_actions.max():.6f}")
    
    # Per dimension statistics
    print(f"\nPer-dimension statistics:")
    for dim in range(min(7, all_actions.shape[-1])):
        actions_dim = all_actions[..., dim]
        print(f"  Dim {dim}: mean={actions_dim.mean():.6f}, "
              f"std={actions_dim.std():.6f}, "
              f"range=[{actions_dim.min():.6f}, {actions_dim.max():.6f}]")
    
    return all_actions


def analyze_errors(data):
    """Analyze error statistics."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    losses = [entry['loss'] for entry in data]
    squared_errors = [entry['squared_error'] for entry in data]
    
    print(f"\nLoss progression:")
    print(f"  First: {losses[0]:.6f}")
    print(f"  Last: {losses[-1]:.6f}")
    print(f"  Mean: {np.mean(losses):.6f}")
    print(f"  Std: {np.std(losses):.6f}")
    
    all_squared_errors = np.concatenate(squared_errors, axis=0)
    print(f"\nSquared error statistics:")
    print(f"  Mean: {all_squared_errors.mean():.6f}")
    print(f"  Std: {all_squared_errors.std():.6f}")
    print(f"  RMSE: {np.sqrt(all_squared_errors.mean()):.6f}")
    
    # Per dimension
    print(f"\nPer-dimension RMSE:")
    for dim in range(min(7, all_squared_errors.shape[-1])):
        rmse = np.sqrt(all_squared_errors[..., dim].mean())
        print(f"  Dim {dim}: {rmse:.6f}")
    
    return losses, squared_errors


def plot_analysis(data, output_dir: str):
    """Create visualization plots."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss over steps
    plt.figure(figsize=(10, 6))
    losses = [entry['loss'] for entry in data]
    plt.plot(losses, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss over Debug Steps')
    plt.grid(True)
    plt.savefig(output_path / 'loss_progression.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path / 'loss_progression.png'}")
    
    # Plot 2: Action distribution per dimension
    all_actions = np.concatenate([entry['actions'] for entry in data], axis=0)
    num_dims = min(7, all_actions.shape[-1])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for dim in range(num_dims):
        actions_dim = all_actions[..., dim].flatten()
        axes[dim].hist(actions_dim, bins=50, alpha=0.7, edgecolor='black')
        axes[dim].set_xlabel('Action Value')
        axes[dim].set_ylabel('Frequency')
        axes[dim].set_title(f'Dimension {dim}')
        axes[dim].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for dim in range(num_dims, len(axes)):
        axes[dim].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'action_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path / 'action_distributions.png'}")
    
    # Plot 3: Squared error per dimension
    all_squared_errors = np.concatenate([entry['squared_error'] for entry in data], axis=0)
    rmse_per_dim = [np.sqrt(all_squared_errors[..., dim].mean()) for dim in range(num_dims)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_dims), rmse_per_dim)
    plt.xlabel('Action Dimension')
    plt.ylabel('RMSE')
    plt.title('RMSE per Action Dimension')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'rmse_per_dimension.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path / 'rmse_per_dimension.png'}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Analyze debug training data')
    parser.add_argument('debug_file', type=str, help='Path to debug data pickle file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', type=str, default='/root/openpi/debug_output/plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print(f"Loading debug data from: {args.debug_file}")
    data = load_debug_data(args.debug_file)
    
    print(f"Loaded {len(data)} debug steps")
    
    # Analyze
    all_actions = analyze_actions(data)
    losses, squared_errors = analyze_errors(data)
    
    # Generate plots if requested
    if args.plot:
        plot_analysis(data, args.output_dir)
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    # Check for common issues
    action_mean_abs = np.abs(all_actions).mean()
    if action_mean_abs < 0.001:
        print("\n⚠️  WARNING: Actions are very small (mean abs < 0.001)")
        print("   This suggests:")
        print("   1. Model hasn't learned yet (need more training)")
        print("   2. Normalization issue")
        print("   3. Data issue (double delta?)")
    elif action_mean_abs < 0.01:
        print("\n✓ Actions are in reasonable range for delta actions")
    else:
        print("\n⚠️  Actions seem large - check if this is expected")
    
    avg_loss = np.mean(losses)
    if avg_loss > 0.5:
        print(f"\n⚠️  Loss is high ({avg_loss:.3f}) - model not converged")
    elif avg_loss > 0.1:
        print(f"\n⚠️  Loss is moderate ({avg_loss:.3f}) - continue training")
    else:
        print(f"\n✓ Loss is reasonable ({avg_loss:.3f})")


if __name__ == "__main__":
    main()


