import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import zarr
from zarr.storage import ZipStore


def _load_array(root: zarr.Group, key: str) -> np.ndarray:
    arr = root[key][...]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _finite_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mask = np.isfinite(x).all(axis=1)
    return x[mask]


def plot_xyz_hist(axs, xyz: np.ndarray, title_prefix: str, bins: int = 120):
    labels = ["x", "y", "z"]
    for i in range(3):
        ax = axs[i]
        ax.hist(xyz[:, i], bins=bins, alpha=0.75, color="#4C78A8")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("count")
        ax.set_title(f"{title_prefix} {labels[i]}")
        ax.grid(True, alpha=0.2)


def plot_xyz_scatter3d(ax, xyz0: np.ndarray, xyz1: np.ndarray, subsample: int = 8):
    if subsample > 1:
        xyz0 = xyz0[::subsample]
        xyz1 = xyz1[::subsample]

    ax.scatter(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], s=1, alpha=0.35, label="robot0_eef_pos")
    ax.scatter(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], s=1, alpha=0.35, label="robot1_eef_pos")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("EEF position 3D scatter (subsampled)")
    ax.legend(loc="best")


def plot_axis_angle(axs, aa: np.ndarray, title_prefix: str, bins: int = 120):
    labels = ["ax", "ay", "az"]
    for i in range(3):
        ax = axs[i]
        ax.hist(aa[:, i], bins=bins, alpha=0.75, color="#F58518")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("count")
        ax.set_title(f"{title_prefix} {labels[i]}")
        ax.grid(True, alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_advantage_messy_demostration_20260409_201357_to_20260409_211241_ep51.zarr.zip",
        help="Path to .zarr.zip dataset",
    )
    ap.add_argument("--outdir", default="./eef_dist_plots", help="Output directory for PNGs")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins")
    ap.add_argument("--subsample", type=int, default=8, help="Scatter subsample stride")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    store = ZipStore(args.dataset, mode="r")
    root = zarr.open_group(store=store, mode="r")

    r0_pos = _finite_rows(_load_array(root, "data/robot0_eef_pos"))
    r1_pos = _finite_rows(_load_array(root, "data/robot1_eef_pos"))
    r0_aa = _finite_rows(_load_array(root, "data/robot0_eef_rot_axis_angle"))
    r1_aa = _finite_rows(_load_array(root, "data/robot1_eef_rot_axis_angle"))
    store.close()

    # --- Position histograms ---
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    plot_xyz_hist(axs[0], r0_pos, "robot0_eef_pos", bins=args.bins)
    plot_xyz_hist(axs[1], r1_pos, "robot1_eef_pos", bins=args.bins)
    fig.suptitle("EEF position distributions (per-dimension hist)")
    pos_hist_path = outdir / "eef_pos_hist.png"
    fig.savefig(pos_hist_path, dpi=180)
    plt.close(fig)

    # --- Position 3D scatter ---
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plot_xyz_scatter3d(ax, r0_pos, r1_pos, subsample=max(1, args.subsample))
    pos_scatter_path = outdir / "eef_pos_scatter3d.png"
    fig.savefig(pos_scatter_path, dpi=180)
    plt.close(fig)

    # --- Axis-angle histograms + magnitude ---
    fig, axs = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    plot_axis_angle(axs[0, :3], r0_aa, "robot0_eef_rot_axis_angle", bins=args.bins)
    plot_axis_angle(axs[1, :3], r1_aa, "robot1_eef_rot_axis_angle", bins=args.bins)

    r0_mag = np.linalg.norm(r0_aa, axis=1)
    r1_mag = np.linalg.norm(r1_aa, axis=1)
    axs[0, 3].hist(r0_mag, bins=args.bins, alpha=0.75, color="#54A24B")
    axs[0, 3].set_title("robot0 axis-angle |w|")
    axs[1, 3].hist(r1_mag, bins=args.bins, alpha=0.75, color="#54A24B")
    axs[1, 3].set_title("robot1 axis-angle |w|")
    for a in (axs[0, 3], axs[1, 3]):
        a.set_xlabel("|w| (rad)")
        a.set_ylabel("count")
        a.grid(True, alpha=0.2)
    fig.suptitle("EEF rotation distributions (axis-angle)")
    rot_hist_path = outdir / "eef_rot_axis_angle_hist.png"
    fig.savefig(rot_hist_path, dpi=180)
    plt.close(fig)

    print("Saved:")
    print(" -", pos_hist_path.resolve())
    print(" -", pos_scatter_path.resolve())
    print(" -", rot_hist_path.resolve())


if __name__ == "__main__":
    main()

