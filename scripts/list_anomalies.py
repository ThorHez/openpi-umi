#!/usr/bin/env python3
"""
List all anomaly files in a checkpoint directory.

Usage:
    python scripts/list_anomalies.py path/to/checkpoint_dir
"""

import argparse
import pickle
from pathlib import Path
from tabulate import tabulate


def list_anomalies(checkpoint_dir: Path):
    """List all anomaly files in the checkpoint directory."""
    anomaly_dir = checkpoint_dir / "anomalies"
    
    if not anomaly_dir.exists():
        print(f"❌ No anomalies directory found in {checkpoint_dir}")
        return []
    
    anomaly_files = sorted(anomaly_dir.glob("*.pkl"))
    
    if not anomaly_files:
        print(f"✓ No anomalies found in {anomaly_dir}")
        return []
    
    print(f"\n{'='*100}")
    print(f"🚨 Found {len(anomaly_files)} anomalies in {anomaly_dir}")
    print(f"{'='*100}\n")
    
    # Load and display info
    table_data = []
    for anomaly_file in anomaly_files:
        try:
            with open(anomaly_file, "rb") as f:
                data = pickle.load(f)
            
            table_data.append([
                anomaly_file.name,
                data['step'],
                f"{data['loss']:.6f}",
                data['reason'][:60] + "..." if len(data['reason']) > 60 else data['reason'],
            ])
        except Exception as e:
            table_data.append([
                anomaly_file.name,
                "ERROR",
                "ERROR",
                str(e)[:60],
            ])
    
    print(tabulate(
        table_data,
        headers=["File", "Step", "Loss", "Reason"],
        tablefmt="grid"
    ))
    
    print(f"\n💡 To analyze a specific anomaly:")
    print(f"   python scripts/analyze_anomaly.py {anomaly_dir}/FILENAME.pkl")
    print()
    
    return anomaly_files


def main():
    parser = argparse.ArgumentParser(description="List all training anomalies")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to checkpoint directory")
    
    args = parser.parse_args()
    
    if not args.checkpoint_dir.exists():
        print(f"❌ Error: Directory not found: {args.checkpoint_dir}")
        return 1
    
    list_anomalies(args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    exit(main())












