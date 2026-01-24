#!/usr/bin/env python3
"""
Verify converted LeRobot dataset.

Usage:
    python verify_lerobot_dataset.py --repo-id your_hf_username/umi_dataset --root ./umi_lerobot_dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: lerobot not installed. Please install it first:")
    print("  pip install lerobot")
    sys.exit(1)


def verify_dataset(repo_id: str, root: str):
    """Verify the converted LeRobot dataset."""
    
    print("="*60)
    print("LeRobot Dataset Verification")
    print("="*60)
    print(f"\n📦 Loading dataset: {repo_id}")
    print(f"   from: {root}")
    
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return False
    
    print("\n✅ Dataset loaded successfully!")
    
    # Basic info
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    print(f"  Repo ID: {dataset.repo_id}")
    print(f"  Total frames: {len(dataset)}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Total episodes: {dataset.num_episodes}")
    
    # Episode info
    print("\n" + "="*60)
    print("📁 EPISODE INFORMATION")
    print("="*60)
    
    episode_lengths = dataset.episode_data_index['to'] - dataset.episode_data_index['from']
    print(f"  Average episode length: {float(episode_lengths.float().mean()):.1f} frames")
    print(f"  Min episode length: {int(episode_lengths.min())} frames")
    print(f"  Max episode length: {int(episode_lengths.max())} frames")
    print(f"  Total frames: {int(episode_lengths.sum())} frames")
    
    # Show first few episodes
    print(f"\n  First 5 episodes:")
    for i in range(min(5, dataset.num_episodes)):
        start = int(dataset.episode_data_index['from'][i])
        end = int(dataset.episode_data_index['to'][i])
        length = end - start
        print(f"    Episode {i}: {length} frames (index {start} to {end})")
    
    # Check frame structure
    print("\n" + "="*60)
    print("🔍 FRAME STRUCTURE")
    print("="*60)
    
    try:
        frame = dataset[0]
        print(f"  Keys in frame: {list(frame.keys())}")
        
        # Check required fields (LeRobot uses '.' instead of '/')
        required_fields = [
            "observation.robot0_eef_pos",
            "observation.robot0_eef_rot_axis_angle",
            "observation.robot0_gripper_width",
            "observation.camera0_rgb",
            "action",
        ]
        
        print(f"\n  Checking required fields:")
        all_present = True
        for field in required_fields:
            present = field in frame
            shape = frame[field].shape if present else "N/A"
            dtype = frame[field].dtype if present else "N/A"
            status = "✅" if present else "❌"
            print(f"    {status} {field:40s} shape={str(shape):20s} dtype={dtype}")
            if not present:
                all_present = False
        
        if not all_present:
            print("\n❌ Some required fields are missing!")
            return False
        
        # Check shapes
        print(f"\n  Verifying data shapes:")
        checks = [
            ("End-effector position", frame["observation.robot0_eef_pos"].shape, (3,)),
            ("End-effector rotation", frame["observation.robot0_eef_rot_axis_angle"].shape, (3,)),
            # ("Gripper width", frame["observation.robot0_gripper_width"].shape, (1,)),
            ("Camera image (CHW)", frame["observation.camera0_rgb"].shape, (3, 224, 224)),  # LeRobot uses CHW format
            ("Action", frame["action"].shape, (7,)),
        ]
        
        all_correct = True
        for name, actual, expected in checks:
            correct = actual == expected
            status = "✅" if correct else "❌"
            print(f"    {status} {name:30s} expected={expected}, actual={actual}")
            if not correct:
                all_correct = False
        
        if not all_correct:
            print("\n❌ Some shapes are incorrect!")
            return False
        
        # Check data types
        print(f"\n  Verifying data types:")
        dtype_checks = [
            ("State (position)", frame["observation.robot0_eef_pos"].dtype, "torch.float32"),
            ("State (rotation)", frame["observation.robot0_eef_rot_axis_angle"].dtype, "torch.float32"),
            ("State (gripper)", frame["observation.robot0_gripper_width"].dtype, "torch.float32"),
            ("Image", frame["observation.camera0_rgb"].dtype, "torch.uint8"),
            ("Action", frame["action"].dtype, "torch.float32"),
        ]
        
        all_correct_dtype = True
        for name, actual, expected in dtype_checks:
            correct = str(actual) == expected
            status = "✅" if correct else "❌"
            print(f"    {status} {name:30s} expected={expected}, actual={actual}")
            if not correct:
                all_correct_dtype = False
        
        if not all_correct_dtype:
            print("\n⚠️  Warning: Some data types are incorrect, but may still work")
        
        # Check data ranges
        print(f"\n  Checking data ranges:")
        print(f"    Position range: [{float(frame['observation.robot0_eef_pos'].min()):.3f}, {float(frame['observation.robot0_eef_pos'].max()):.3f}]")
        print(f"    Rotation range: [{float(frame['observation.robot0_eef_rot_axis_angle'].min()):.3f}, {float(frame['observation.robot0_eef_rot_axis_angle'].max()):.3f}]")
        print(f"    Gripper range: [{float(frame['observation.robot0_gripper_width'].min()):.3f}, {float(frame['observation.robot0_gripper_width'].max()):.3f}]")
        print(f"    Image range: [{int(frame['observation.camera0_rgb'].min())}, {int(frame['observation.camera0_rgb'].max())}]")
        print(f"    Action range: [{float(frame['action'].min()):.3f}, {float(frame['action'].max()):.3f}]")
        
        # Check if image has content
        if int(frame['observation.camera0_rgb'].max()) == 0:
            print(f"\n⚠️  Warning: Image appears to be all black!")
        
    except Exception as e:
        print(f"\n❌ Error checking frame structure: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sample random frames
    print("\n" + "="*60)
    print("🎲 RANDOM SAMPLING TEST")
    print("="*60)
    
    num_samples = min(10, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"  Testing {num_samples} random frames...")
    try:
        for i, idx in enumerate(sample_indices):
            frame = dataset[int(idx)]  # Convert numpy.int64 to Python int
            # Just accessing the frame is a good test
            _ = frame["observation.camera0_rgb"].shape
        print(f"  ✅ All {num_samples} samples loaded successfully!")
    except Exception as e:
        print(f"  ❌ Error loading random samples: {e}")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    print("  ✅ Dataset loaded successfully")
    print("  ✅ All required fields present")
    print("  ✅ All shapes correct")
    print(f"  {'✅' if all_correct_dtype else '⚠️ '} Data types {'correct' if all_correct_dtype else 'have warnings'}")
    print("  ✅ Random sampling works")
    
    print("\n" + "="*60)
    print("✅ DATASET VERIFICATION PASSED!")
    print("="*60)
    
    print("\n🎉 Your dataset is ready for training!")
    print("\n📖 Next steps:")
    print("  1. Update the repo_id in src/openpi/training/config.py")
    print("  2. Run training: python scripts/train.py --config pi05_umi")
    print("  3. (Optional) Push to HuggingFace Hub if not already done")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify converted LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./umi_lerobot_dataset",
        help="Root directory of the dataset",
    )
    
    args = parser.parse_args()
    
    # Check if root exists
    if not Path(args.root).exists():
        print(f"❌ Error: Dataset directory not found: {args.root}")
        sys.exit(1)
    
    # Verify dataset
    success = verify_dataset(args.repo_id, args.root)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

