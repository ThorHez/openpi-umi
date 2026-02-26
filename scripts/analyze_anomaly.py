#!/usr/bin/env python3
"""
Analyze anomaly data saved during training.

Usage:
    python scripts/analyze_anomaly.py path/to/anomaly/file.pkl
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_anomaly(anomaly_file: Path):
    """Load and analyze an anomaly file."""
    print(f"\n{'='*80}")
    print(f"📊 Analyzing anomaly: {anomaly_file.name}")
    print(f"{'='*80}\n")
    
    with open(anomaly_file, "rb") as f:
        data = pickle.load(f)
    
    # Print basic info
    print(f"Step: {data['step']}")
    print(f"Loss: {data['loss']:.6f}")
    print(f"Reason: {data['reason']}")
    print()
    
    # Plot loss history
    if len(data['loss_history']) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(data['loss_history'], marker='o')
        plt.axhline(y=data['loss'], color='r', linestyle='--', label=f"Anomaly loss: {data['loss']:.4f}")
        plt.xlabel("Recent steps")
        plt.ylabel("Loss")
        plt.title(f"Loss History (Step {data['step']})")
        plt.legend()
        plt.grid(True)
        
        plot_file = anomaly_file.with_suffix('.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📈 Loss plot saved to: {plot_file}")
        plt.close()
    
    # Analyze observation
    obs = data['observation']
    print(f"\n🔍 Observation Analysis:")
    print(f"   State shape: {obs.state.shape}")
    print(f"   State stats: mean={np.mean(obs.state):.4f}, std={np.std(obs.state):.4f}, "
          f"min={np.min(obs.state):.4f}, max={np.max(obs.state):.4f}")
    print(f"   State has NaN: {np.any(np.isnan(obs.state))}")
    print(f"   State has Inf: {np.any(np.isinf(obs.state))}")
    
    print(f"\n   Images:")
    for key, img in obs.images.items():
        print(f"     {key}: shape={img.shape}, "
              f"mean={np.mean(img):.4f}, std={np.std(img):.4f}, "
              f"min={np.min(img):.4f}, max={np.max(img):.4f}")
        print(f"       Has NaN: {np.any(np.isnan(img))}, Has Inf: {np.any(np.isinf(img))}")
    
    # Analyze actions
    actions = data['actions']
    print(f"\n🎯 Actions Analysis:")
    print(f"   Shape: {actions.shape}")
    print(f"   Stats: mean={np.mean(actions):.4f}, std={np.std(actions):.4f}, "
          f"min={np.min(actions):.4f}, max={np.max(actions):.4f}")
    print(f"   Has NaN: {np.any(np.isnan(actions))}")
    print(f"   Has Inf: {np.any(np.isinf(actions))}")
    
    # Show actions distribution
    print(f"\n   Actions per dimension:")
    for i in range(min(actions.shape[-1], 10)):  # Show first 10 dimensions
        dim_data = actions[..., i]
        print(f"     Dim {i}: mean={np.mean(dim_data):.4f}, std={np.std(dim_data):.4f}, "
              f"min={np.min(dim_data):.4f}, max={np.max(dim_data):.4f}")
    
    # Additional info
    if 'info' in data:
        print(f"\n📋 Training Info:")
        for key, value in data['info'].items():
            print(f"   {key}: {value}")
    
    print(f"\n{'='*80}\n")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Analyze training anomaly data")
    parser.add_argument("anomaly_file", type=Path, help="Path to anomaly pickle file")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Drop into interactive Python shell after analysis")
    
    args = parser.parse_args()
    
    if not args.anomaly_file.exists():
        print(f"❌ Error: File not found: {args.anomaly_file}")
        return 1
    
    data = analyze_anomaly(args.anomaly_file)
    
    if args.interactive:
        print("🐍 Dropping into interactive shell...")
        print("   Available variables: 'data' (anomaly data dict)")
        import IPython
        IPython.embed()
    
    return 0


if __name__ == "__main__":
    exit(main())












