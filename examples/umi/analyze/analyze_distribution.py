#!/usr/bin/env python3
"""
Analyze and visualize feature distributions across multiple LeRobot datasets.

This script compares the distribution of features across multiple datasets and generates
visualization plots for each dimension.

Usage:
    python analyze_distribution.py \
        --datasets /path/to/dataset1 /path/to/dataset2 \
        --features observation.robot0_eef_pos actions \
        --output_dir ./distribution_plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def flatten_nested(nested):
    """Flatten nested list to 1D"""
    result = []
    if isinstance(nested, (list, tuple, np.ndarray)):
        for item in nested:
            result.extend(flatten_nested(item))
    else:
        result.append(nested)
    return result


def load_feature_data(
    dataset_path: Path,
    features: List[str],
    max_episodes: Optional[int] = None,
    sample_rate: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Load feature data from a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        features: List of feature names to load
        max_episodes: Maximum number of episodes to process
        sample_rate: Fraction of frames to sample (0.0-1.0)
    
    Returns:
        Dict mapping feature names to numpy arrays of all values
    """
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]
    
    # Initialize data containers
    feature_data = {feat: [] for feat in features}
    
    for parquet_file in tqdm(parquet_files, desc=f"Loading {dataset_path.name}"):
        table = pq.read_table(parquet_file)
        
        for feature in features:
            if feature not in table.column_names:
                continue
            
            col = table.column(feature)
            
            for row_idx in range(table.num_rows):
                # Apply sampling
                if sample_rate < 1.0 and np.random.random() > sample_rate:
                    continue
                
                try:
                    data = col[row_idx].as_py()
                    flat_data = flatten_nested(data)
                    feature_data[feature].append(flat_data)
                except Exception:
                    continue
    
    # Convert to numpy arrays
    result = {}
    for feat, data_list in feature_data.items():
        if data_list:
            result[feat] = np.array(data_list)
    
    return result


def compute_distribution_stats(data: np.ndarray) -> Dict[str, Any]:
    """Compute distribution statistics for each dimension."""
    stats = {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q25": np.percentile(data, 25, axis=0).tolist(),
        "q50": np.percentile(data, 50, axis=0).tolist(),
        "q75": np.percentile(data, 75, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist(),
        "count": len(data),
        "num_dims": data.shape[1] if len(data.shape) > 1 else 1,
    }
    return stats


def plot_histogram_comparison(
    datasets_data: Dict[str, np.ndarray],
    feature_name: str,
    dim_idx: int,
    output_path: Path,
    bins: int = 100
):
    """
    Plot histogram comparison for a specific dimension across datasets.
    
    Args:
        datasets_data: Dict mapping dataset names to their data arrays
        feature_name: Name of the feature
        dim_idx: Dimension index to plot
        output_path: Path to save the plot
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets_data)))
    
    for (dataset_name, data), color in zip(datasets_data.items(), colors):
        if data.ndim == 1:
            dim_data = data
        else:
            if dim_idx >= data.shape[1]:
                continue
            dim_data = data[:, dim_idx]
        
        # Remove outliers for better visualization (keep 99% of data)
        q01, q99 = np.percentile(dim_data, [1, 99])
        mask = (dim_data >= q01) & (dim_data <= q99)
        filtered_data = dim_data[mask]
        
        ax.hist(filtered_data, bins=bins, alpha=0.5, label=dataset_name, color=color, density=True)
    
    ax.set_xlabel(f"Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{feature_name} - Dimension {dim_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplot_comparison(
    datasets_data: Dict[str, np.ndarray],
    feature_name: str,
    output_path: Path,
    max_dims: int = 20
):
    """
    Plot boxplot comparison for all dimensions across datasets.
    
    Args:
        datasets_data: Dict mapping dataset names to their data arrays
        feature_name: Name of the feature
        output_path: Path to save the plot
        max_dims: Maximum number of dimensions to show
    """
    # Determine number of dimensions
    num_dims = 1
    for data in datasets_data.values():
        if data.ndim > 1:
            num_dims = min(data.shape[1], max_dims)
            break
    
    fig, axes = plt.subplots(1, num_dims, figsize=(3 * num_dims, 6))
    if num_dims == 1:
        axes = [axes]
    
    dataset_names = list(datasets_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    for dim_idx, ax in enumerate(axes):
        box_data = []
        labels = []
        
        for dataset_name, data in datasets_data.items():
            if data.ndim == 1:
                dim_data = data
            else:
                if dim_idx >= data.shape[1]:
                    continue
                dim_data = data[:, dim_idx]
            
            # Remove outliers
            q01, q99 = np.percentile(dim_data, [1, 99])
            mask = (dim_data >= q01) & (dim_data <= q99)
            box_data.append(dim_data[mask])
            labels.append(dataset_name)
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(f"Dim {dim_idx}")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"{feature_name} - Box Plot Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_violin_comparison(
    datasets_data: Dict[str, np.ndarray],
    feature_name: str,
    output_path: Path,
    max_dims: int = 10
):
    """
    Plot violin comparison for dimensions across datasets.
    """
    # Determine number of dimensions
    num_dims = 1
    for data in datasets_data.values():
        if data.ndim > 1:
            num_dims = min(data.shape[1], max_dims)
            break
    
    fig, axes = plt.subplots(1, num_dims, figsize=(4 * num_dims, 6))
    if num_dims == 1:
        axes = [axes]
    
    dataset_names = list(datasets_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    for dim_idx, ax in enumerate(axes):
        violin_data = []
        
        for dataset_name, data in datasets_data.items():
            if data.ndim == 1:
                dim_data = data
            else:
                if dim_idx >= data.shape[1]:
                    continue
                dim_data = data[:, dim_idx]
            
            # Remove outliers and sample for performance
            q01, q99 = np.percentile(dim_data, [1, 99])
            mask = (dim_data >= q01) & (dim_data <= q99)
            filtered = dim_data[mask]
            
            # Sample if too large
            if len(filtered) > 10000:
                filtered = np.random.choice(filtered, 10000, replace=False)
            
            violin_data.append(filtered)
        
        if violin_data:
            parts = ax.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        
        ax.set_title(f"Dim {dim_idx}")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"{feature_name} - Violin Plot Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_statistics_table(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    feature_name: str,
    output_path: Path
):
    """
    Plot statistics comparison as a table image.
    
    Args:
        all_stats: Nested dict: {dataset_name: {feature_name: stats}}
        feature_name: Feature to visualize
        output_path: Path to save the plot
    """
    # Collect data for table
    dataset_names = list(all_stats.keys())
    
    # Get number of dimensions
    num_dims = 1
    for stats in all_stats.values():
        if feature_name in stats:
            num_dims = stats[feature_name].get("num_dims", 1)
            break
    
    # Create figure with subplots for each dimension
    fig, ax = plt.subplots(figsize=(14, 4 + 0.3 * len(dataset_names)))
    ax.axis('off')
    
    # Prepare table data
    columns = ["Dataset", "Count"] + [f"Dim {i}" for i in range(min(num_dims, 10))]
    
    rows_data = []
    for dataset_name in dataset_names:
        if feature_name not in all_stats[dataset_name]:
            continue
        
        stats = all_stats[dataset_name][feature_name]
        row = [dataset_name, stats["count"]]
        
        means = stats["mean"] if isinstance(stats["mean"], list) else [stats["mean"]]
        stds = stats["std"] if isinstance(stats["std"], list) else [stats["std"]]
        
        for i in range(min(num_dims, 10)):
            if i < len(means):
                row.append(f"{means[i]:.4f}\n±{stds[i]:.4f}")
            else:
                row.append("N/A")
        
        rows_data.append(row)
    
    if rows_data:
        table = ax.table(
            cellText=rows_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] + [0.08] + [0.077] * min(num_dims, 10)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', weight='bold')
    
    ax.set_title(f"{feature_name} - Statistics Comparison (Mean ± Std)", fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    features: List[str],
    output_path: Path
):
    """
    Plot a summary dashboard showing global statistics for all features and datasets.
    """
    dataset_names = list(all_stats.keys())
    
    fig, axes = plt.subplots(len(features), 1, figsize=(14, 4 * len(features)))
    if len(features) == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, features):
        # Collect global min/max for each dataset
        x_labels = []
        global_mins = []
        global_maxs = []
        global_q01s = []
        global_q99s = []
        
        for dataset_name in dataset_names:
            if feature not in all_stats[dataset_name]:
                continue
            
            stats = all_stats[dataset_name][feature]
            x_labels.append(dataset_name)
            
            mins = stats["min"] if isinstance(stats["min"], list) else [stats["min"]]
            maxs = stats["max"] if isinstance(stats["max"], list) else [stats["max"]]
            q01s = stats["q01"] if isinstance(stats["q01"], list) else [stats["q01"]]
            q99s = stats["q99"] if isinstance(stats["q99"], list) else [stats["q99"]]
            
            global_mins.append(min(mins))
            global_maxs.append(max(maxs))
            global_q01s.append(min(q01s))
            global_q99s.append(max(q99s))
        
        if not x_labels:
            continue
        
        x = np.arange(len(x_labels))
        width = 0.2
        
        ax.bar(x - 1.5*width, global_mins, width, label='Min', color='blue', alpha=0.7)
        ax.bar(x - 0.5*width, global_q01s, width, label='Q01', color='cyan', alpha=0.7)
        ax.bar(x + 0.5*width, global_q99s, width, label='Q99', color='orange', alpha=0.7)
        ax.bar(x + 1.5*width, global_maxs, width, label='Max', color='red', alpha=0.7)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Value')
        ax.set_title(f'{feature} - Global Range Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_distributions(
    datasets: List[str],
    features: List[str],
    output_dir: str,
    max_episodes: Optional[int] = None,
    sample_rate: float = 0.1,
    verbose: bool = True
):
    """
    Main function to analyze and visualize distributions across datasets.
    
    Args:
        datasets: List of dataset paths
        features: List of feature names to analyze
        output_dir: Directory to save output plots
        max_episodes: Maximum episodes per dataset
        sample_rate: Fraction of frames to sample
        verbose: Print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data from all datasets
    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    all_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for dataset_path_str in datasets:
        dataset_path = Path(dataset_path_str)
        dataset_name = dataset_path.name
        
        if verbose:
            print(f"\n📊 Loading dataset: {dataset_name}")
        
        try:
            feature_data = load_feature_data(
                dataset_path, features, max_episodes, sample_rate
            )
            all_data[dataset_name] = feature_data
            
            # Compute statistics
            all_stats[dataset_name] = {}
            for feat, data in feature_data.items():
                all_stats[dataset_name][feat] = compute_distribution_stats(data)
                if verbose:
                    print(f"  • {feat}: {data.shape}, {data.shape[0]} samples")
        
        except Exception as e:
            print(f"  ❌ Error loading {dataset_name}: {e}")
            continue
    
    if not all_data:
        print("❌ No data loaded from any dataset")
        return
    
    if verbose:
        print(f"\n🎨 Generating visualizations...")
    
    # Generate plots for each feature
    for feature in features:
        if verbose:
            print(f"\n  Processing: {feature}")
        
        # Collect data for this feature across datasets
        feature_datasets = {}
        for dataset_name, data_dict in all_data.items():
            if feature in data_dict:
                feature_datasets[dataset_name] = data_dict[feature]
        
        if not feature_datasets:
            if verbose:
                print(f"    ⚠️  No data found for {feature}")
            continue
        
        # Create feature output directory
        feature_dir = output_path / feature.replace(".", "_")
        feature_dir.mkdir(exist_ok=True)
        
        # Get number of dimensions
        num_dims = 1
        for data in feature_datasets.values():
            if data.ndim > 1:
                num_dims = data.shape[1]
                break
        
        # Generate histogram for each dimension
        if verbose:
            print(f"    📈 Generating histograms ({num_dims} dimensions)...")
        
        for dim_idx in range(min(num_dims, 30)):  # Limit to 30 dimensions
            hist_path = feature_dir / f"histogram_dim_{dim_idx:02d}.png"
            plot_histogram_comparison(feature_datasets, feature, dim_idx, hist_path)
        
        # Generate boxplot comparison
        if verbose:
            print(f"    📦 Generating boxplot...")
        boxplot_path = feature_dir / "boxplot_comparison.png"
        plot_boxplot_comparison(feature_datasets, feature, boxplot_path)
        
        # Generate violin plot
        if verbose:
            print(f"    🎻 Generating violin plot...")
        violin_path = feature_dir / "violin_comparison.png"
        plot_violin_comparison(feature_datasets, feature, violin_path)
        
        # Generate statistics table
        if verbose:
            print(f"    📋 Generating statistics table...")
        stats_path = feature_dir / "statistics_table.png"
        plot_statistics_table(all_stats, feature, stats_path)
    
    # Generate summary dashboard
    if verbose:
        print(f"\n  📊 Generating summary dashboard...")
    dashboard_path = output_path / "summary_dashboard.png"
    plot_summary_dashboard(all_stats, features, dashboard_path)
    
    # Save statistics to JSON
    stats_json_path = output_path / "distribution_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    
    if verbose:
        print(f"\n✅ Analysis complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Statistics saved: {stats_json_path}")
        print(f"  Plots generated for {len(features)} features across {len(all_data)} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize feature distributions across multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Paths to dataset directories"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["observation.robot0_eef_pos", "actions"],
        help="Feature names to analyze (default: observation.robot0_eef_pos actions)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distribution_analysis",
        help="Output directory for plots (default: ./distribution_analysis)"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process per dataset"
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="Fraction of frames to sample (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    analyze_distributions(
        datasets=args.datasets,
        features=args.features,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        sample_rate=args.sample_rate,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

