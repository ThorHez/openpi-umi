#!/usr/bin/env python3
"""
分析 dataset_no_filter_1.zarr.zip 的桌面平面与每个点的绝对高度

流程:
  1. 加载 zarr 数据，提取 robot0_eef_pos 和 robot1_eef_pos
  2. 将所有点合并，基于多参数扫描选取“桌面候选点”
  3. 用 RANSAC 稳健拟合桌面平面（可选 force_horizontal）
  4. 对数据集中的每个点计算到桌面平面的有符号距离（相对桌面高度）
  5. 输出平面参数、统计信息，并保存结果到 .npz 文件

用法:
    python analyze_desk_height.py
    python analyze_desk_height.py --zarr /path/to/dataset.zarr.zip
    python analyze_desk_height.py --percentile 1
    python analyze_desk_height.py --force_horizontal on
    python analyze_desk_height.py --force_horizontal auto --auto_h_median_deg 5 --auto_h_p90_deg 10
    # 将相对桌面高度写回数据集（输出新 .zarr.zip）：双臂写 robot0/robot1_eef_height_above_desk，单臂仅写 robot0_eef_height_above_desk
    python analyze_desk_height.py --zarr /path/to/dataset.zarr.zip --write_back /path/to/dataset_with_desk_height.zarr.zip
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import zarr


# ===================== 平面拟合基础函数 =====================

def _fit_plane_svd(points3: np.ndarray):
    """
    用最小二乘(SVD)拟合平面: n·x + d = 0
    返回 (n_unit, d, centroid)
    """
    assert points3.ndim == 2 and points3.shape[1] == 3
    centroid = points3.mean(axis=0)
    X = points3 - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return None
    n = n / n_norm
    d = -np.dot(n, centroid)
    return n, d, centroid


def plane_signed_distance(points: np.ndarray, n: np.ndarray, d: float):
    """
    有符号距离：dist = n·p + d （n 已归一化时，单位为米）
    """
    return points @ n + d


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold: float = 0.003,
    max_iters: int = 2000,
    min_inliers: int = 50,
    seed: int = 0,
    refine: bool = True,
):
    """
    RANSAC 平面拟合
    输入:
        points: (N,3)
        threshold: 内点距离阈值（米）
        max_iters: RANSAC 迭代次数
        min_inliers: 认为有效模型的最少内点数
        refine: 是否用最佳内点集再做一次 SVD 精修
    输出:
        best_n: (3,) 单位法向量
        best_d: float
        best_inliers: (N,) bool
        best_signed_dist: (N,) float  每个点到平面的有符号距离
    """
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    N = points.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    rng = np.random.default_rng(seed)

    best_inliers = None
    best_score = -1
    best_model = None

    for _ in range(max_iters):
        ids = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[ids]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        d_val = -np.dot(n, p1)

        signed_dist = plane_signed_distance(points, n, d_val)
        inliers = np.abs(signed_dist) < threshold
        score = int(inliers.sum())

        if score > best_score and score >= min_inliers:
            best_score = score
            best_inliers = inliers
            best_model = (n, d_val)

    if best_model is None:
        raise RuntimeError(
            f"RANSAC failed: no plane found with >= {min_inliers} inliers. "
            f"Try increasing max_iters, threshold, or providing cleaner points."
        )

    best_n, best_d = best_model
    best_signed = plane_signed_distance(points, best_n, best_d)

    if refine:
        inlier_pts = points[best_inliers]
        refined = _fit_plane_svd(inlier_pts)
        if refined is not None:
            best_n, best_d, _ = refined
            best_signed = plane_signed_distance(points, best_n, best_d)
            best_inliers = np.abs(best_signed) < threshold

    return best_n, float(best_d), best_inliers, best_signed


# ===================== 稳健扫描与评分 =====================

def _angle_deg_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    """两个法向量夹角（度），忽略方向符号（n 与 -n 视为同向）"""
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    c = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _robust_mad(x: np.ndarray) -> float:
    """Median Absolute Deviation"""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _xy_coverage_ratio(points_xy: np.ndarray, mask: np.ndarray, grid_size: int = 16) -> float:
    """
    计算内点在 XY 平面的覆盖率：
    把点云包围盒分成 grid_size x grid_size 网格，
    coverage = 被内点击中的格子数 / 被所有点击中的格子数
    """
    pts = points_xy
    in_pts = points_xy[mask]
    if pts.shape[0] < 10 or in_pts.shape[0] < 10:
        return 0.0

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # 防止退化
    if (x_max - x_min) < 1e-9 or (y_max - y_min) < 1e-9:
        return 0.0

    def to_cell(p):
        x = (p[:, 0] - x_min) / (x_max - x_min + 1e-12)
        y = (p[:, 1] - y_min) / (y_max - y_min + 1e-12)
        ix = np.clip((x * grid_size).astype(np.int32), 0, grid_size - 1)
        iy = np.clip((y * grid_size).astype(np.int32), 0, grid_size - 1)
        return ix, iy

    ix_all, iy_all = to_cell(pts)
    ix_in, iy_in = to_cell(in_pts)

    occ_all = np.zeros((grid_size, grid_size), dtype=bool)
    occ_in = np.zeros((grid_size, grid_size), dtype=bool)
    occ_all[ix_all, iy_all] = True
    occ_in[ix_in, iy_in] = True

    denom = int(occ_all.sum())
    if denom == 0:
        return 0.0
    return float(occ_in.sum() / denom)


def _plane_quality_score(
    inlier_ratio: float,
    coverage: float,
    mad: float,
    normal_stability_penalty: float = 0.0,
) -> float:
    """
    综合评分（越大越好）
    可按数据集微调权重
    """
    return (
        2.0 * inlier_ratio
        + 1.2 * coverage
        - 40.0 * mad
        - 0.02 * normal_stability_penalty
    )


def fit_desk_plane_robust(
    all_points: np.ndarray,
    *,
    percentiles=(0.3, 0.5, 1.0, 2.0, 3.0, 5.0),
    thresholds=(0.004, 0.006, 0.008, 0.010),
    max_iters: int = 5000,
    min_inlier_ratios=(0.20, 0.15, 0.10, 0.07, 0.05),
    seed: int = 42,
    force_upward: bool = True,
    verbose: bool = True,
):
    """
    在多组 percentile + threshold + min_inlier_ratio 上扫描，
    选择最稳健的桌面平面。返回:
      best_n, best_d, best_info, trials
    """
    all_points = np.asarray(all_points, dtype=np.float64)
    assert all_points.ndim == 2 and all_points.shape[1] == 3
    N_total = all_points.shape[0]
    if N_total < 50:
        raise RuntimeError(f"Not enough points: {N_total}")

    trials = []
    rng = np.random.default_rng(seed)

    for p in percentiles:
        z_thr = np.percentile(all_points[:, 2], p)
        cand_mask = all_points[:, 2] <= z_thr
        cand_pts = all_points[cand_mask]
        n_cand = cand_pts.shape[0]

        if n_cand < 100:
            continue

        for th in thresholds:
            for ratio in min_inlier_ratios:
                min_inliers = max(30, int(n_cand * ratio))

                try:
                    n, d, inliers, signed = fit_plane_ransac(
                        cand_pts,
                        threshold=float(th),
                        max_iters=int(max_iters),
                        min_inliers=int(min_inliers),
                        seed=int(rng.integers(0, 1_000_000)),
                        refine=True,
                    )
                except Exception:
                    continue

                if force_upward and n[2] < 0:
                    n = -n
                    d = -d
                    signed = -signed

                inlier_ratio = float(inliers.mean())
                abs_res = np.abs(signed[inliers]) if inliers.any() else np.abs(signed)
                mad = _robust_mad(abs_res) if abs_res.size > 0 else 1e9
                rmse = float(np.sqrt(np.mean(abs_res**2))) if abs_res.size > 0 else 1e9
                coverage = _xy_coverage_ratio(cand_pts[:, :2], inliers, grid_size=16)

                info = {
                    "percentile": float(p),
                    "threshold": float(th),
                    "min_inlier_ratio": float(ratio),
                    "z_threshold": float(z_thr),
                    "num_candidates": int(n_cand),
                    "num_inliers": int(inliers.sum()),
                    "inlier_ratio": float(inlier_ratio),
                    "mad": float(mad),
                    "rmse": float(rmse),
                    "coverage": float(coverage),
                    "normal": n.copy(),
                    "d": float(d),
                }
                trials.append(info)

    if len(trials) == 0:
        raise RuntimeError(
            "fit_desk_plane_robust: no valid plane found. "
            "Try larger thresholds or broader percentile range."
        )

    for t in trials:
        t["score"] = _plane_quality_score(
            t["inlier_ratio"], t["coverage"], t["mad"], normal_stability_penalty=0.0
        )

    trials_sorted = sorted(trials, key=lambda x: x["score"], reverse=True)
    top_k = trials_sorted[: min(12, len(trials_sorted))]

    normals = np.stack([t["normal"] for t in top_k], axis=0)
    n_ref = normals.mean(axis=0)
    n_ref = n_ref / (np.linalg.norm(n_ref) + 1e-12)
    if force_upward and n_ref[2] < 0:
        n_ref = -n_ref

    angles = np.array([_angle_deg_between_normals(t["normal"], n_ref) for t in trials], dtype=np.float64)
    for t, a in zip(trials, angles):
        t["normal_angle_to_ref_deg"] = float(a)
        t["score_refined"] = _plane_quality_score(
            t["inlier_ratio"], t["coverage"], t["mad"], normal_stability_penalty=float(a)
        )

    trials_sorted = sorted(trials, key=lambda x: x["score_refined"], reverse=True)
    best = trials_sorted[0]
    best_n = best["normal"].copy()
    best_d = float(best["d"])

    best["normal_stability_deg_median"] = float(np.median(angles))
    best["normal_stability_deg_p90"] = float(np.percentile(angles, 90))

    if verbose:
        print("\n========== Robust Desk Plane Scan Report ==========")
        print(f"Total trials: {len(trials)}")
        print(f"Best config: percentile={best['percentile']}, threshold={best['threshold']}, "
              f"min_inlier_ratio={best['min_inlier_ratio']}")
        print(f"Candidates: {best['num_candidates']}, Inliers: {best['num_inliers']} "
              f"(ratio={best['inlier_ratio']:.3f})")
        print(f"MAD={best['mad']:.6f} m, RMSE={best['rmse']:.6f} m, Coverage={best['coverage']:.3f}")
        print(f"Normal=[{best_n[0]:.6f}, {best_n[1]:.6f}, {best_n[2]:.6f}], d={best_d:.6f}")
        print(f"Normal stability: median={best['normal_stability_deg_median']:.3f}°, "
              f"p90={best['normal_stability_deg_p90']:.3f}°")

    return best_n, best_d, best, trials_sorted


# ===================== force_horizontal 逻辑 =====================

def fit_horizontal_plane_from_points(
    points: np.ndarray,
    *,
    up_axis: str = "z",
    z_stat: str = "median",
    trim_ratio: float = 0.1,
):
    """
    强制水平平面拟合：
      - 法向固定为 up 轴
      - 只估计 d，使平面位置匹配点云在 up 轴上的中心趋势
    平面: n·x + d = 0
    """
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.shape[0] >= 3

    axis_map = {"x": 0, "y": 1, "z": 2}
    if up_axis not in axis_map:
        raise ValueError(f"up_axis must be one of {list(axis_map.keys())}, got {up_axis}")
    k = axis_map[up_axis]

    n = np.zeros(3, dtype=np.float64)
    n[k] = 1.0

    vals = points[:, k]
    if z_stat == "median":
        h0 = float(np.median(vals))
    elif z_stat == "mean":
        h0 = float(np.mean(vals))
    elif z_stat == "trimmed_mean":
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0, 0.5)")
        lo = np.quantile(vals, trim_ratio)
        hi = np.quantile(vals, 1.0 - trim_ratio)
        core = vals[(vals >= lo) & (vals <= hi)]
        h0 = float(np.mean(core)) if core.size >= 3 else float(np.median(vals))
    else:
        raise ValueError("z_stat must be one of: median, mean, trimmed_mean")

    d = -h0
    signed = plane_signed_distance(points, n, d)
    return n, float(d), signed


def maybe_force_horizontal(
    best_n: np.ndarray,
    best_d: float,
    best_info: dict,
    desk_points: np.ndarray,
    *,
    force_horizontal_mode: str = "off",   # off / on / auto
    auto_median_deg_th: float = 5.0,
    auto_p90_deg_th: float = 10.0,
    up_axis: str = "z",
    z_stat: str = "median",
):
    """
    根据开关决定是否使用水平约束平面。
    Returns:
        n, d, used_horizontal(bool), reason(str), extra(dict)
    """
    mode = (force_horizontal_mode or "off").lower()
    if mode not in ("off", "on", "auto"):
        raise ValueError("force_horizontal_mode must be one of: off/on/auto")

    use_horizontal = False
    reason = "free-plane"

    if mode == "on":
        use_horizontal = True
        reason = "forced-by-user"
    elif mode == "auto":
        med = float(best_info.get("normal_stability_deg_median", 999.0))
        p90 = float(best_info.get("normal_stability_deg_p90", 999.0))
        if med > auto_median_deg_th or p90 > auto_p90_deg_th:
            use_horizontal = True
            reason = f"auto-triggered: stability bad (median={med:.2f}, p90={p90:.2f})"

    if use_horizontal:
        n_h, d_h, signed_h = fit_horizontal_plane_from_points(
            desk_points, up_axis=up_axis, z_stat=z_stat, trim_ratio=0.1
        )
        extra = {
            "horizontal_fit_abs_median": float(np.median(np.abs(signed_h))),
            "horizontal_fit_abs_mean": float(np.mean(np.abs(signed_h))),
        }
        return n_h, d_h, True, reason, extra

    return best_n, float(best_d), False, reason, {}


# ===================== 数据加载 =====================

def load_zarr(zarr_zip_path: str, tmp_dir: str = "/tmp/zarr_desk_height"):
    """加载 zarr.zip 并返回 root group"""
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"解压 {zarr_zip_path} → {tmp_dir}")
    with zipfile.ZipFile(zarr_zip_path, 'r') as z:
        z.extractall(tmp_dir)

    return zarr.open(tmp_dir, mode='r'), tmp_dir


def get_episode_slices(meta_group):
    """返回 [(start, end), ...] 列表"""
    ends = np.array(meta_group['episode_ends'])
    slices = []
    start = 0
    for end in ends:
        slices.append((int(start), int(end)))
        start = end
    return slices


# ===================== 主逻辑 =====================

def analyze_desk_height(
    zarr_zip_path: str,
    z_percentile: float = 1.0,
    ransac_threshold: float = 0.005,
    ransac_iters: int = 3000,
    output_dir: str = None,
    force_horizontal: str = "off",   # off/on/auto
    up_axis: str = "z",
    horizontal_stat: str = "median",
    auto_h_median_deg: float = 5.0,
    auto_h_p90_deg: float = 10.0,
    write_back_path: str = None,
):
    # ---- 1. 加载数据 ----
    root, _tmp_dir = load_zarr(zarr_zip_path)
    data = root['data']
    meta = root['meta']

    episodes = get_episode_slices(meta)
    n_episodes = len(episodes)
    print(f"\n数据集: {zarr_zip_path}")
    print(f"Episode 数: {n_episodes}")

    robot0_pos = np.array(data['robot0_eef_pos'], dtype=np.float64)
    has_robot1 = 'robot1_eef_pos' in data
    robot1_pos = np.array(data['robot1_eef_pos'], dtype=np.float64) if has_robot1 else None

    N = robot0_pos.shape[0]
    print(f"总数据点: {N}")
    print(f"Robot0 eef_pos shape: {robot0_pos.shape}")
    if has_robot1:
        print(f"Robot1 eef_pos shape: {robot1_pos.shape}")
    else:
        print("Robot1 eef_pos: 不存在，仅使用 Robot0")

    all_points = np.vstack([robot0_pos, robot1_pos]) if has_robot1 else robot0_pos
    print(f"\n合并后总点数: {all_points.shape[0]}")
    print(f"  X 范围: [{all_points[:, 0].min():.6f}, {all_points[:, 0].max():.6f}]")
    print(f"  Y 范围: [{all_points[:, 1].min():.6f}, {all_points[:, 1].max():.6f}]")
    print(f"  Z 范围: [{all_points[:, 2].min():.6f}, {all_points[:, 2].max():.6f}]")

    # ---- 2&3. 多参数扫描 + 稳健拟合 ----
    print("\n正在进行稳健桌面平面拟合（多参数扫描）...")

    p0 = float(z_percentile)
    percentile_grid = sorted(set([
        max(0.2, p0 * 0.5),
        max(0.3, p0 * 0.8),
        p0,
        min(8.0, p0 * 1.5),
        min(10.0, p0 * 2.0),
    ]))
    if len(percentile_grid) < 4:
        percentile_grid = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

    threshold_grid = (
        max(0.003, ransac_threshold * 0.8),
        ransac_threshold,
        max(0.008, ransac_threshold * 1.5),
        0.010
    )

    plane_n, plane_d, best_info, trials = fit_desk_plane_robust(
        all_points,
        percentiles=tuple(percentile_grid),
        thresholds=threshold_grid,
        max_iters=max(3000, ransac_iters),
        min_inlier_ratios=(0.20, 0.15, 0.10, 0.07, 0.05),
        seed=42,
        force_upward=True,
        verbose=True,
    )

    z_thr_best = best_info["z_threshold"]
    desk_mask = all_points[:, 2] <= z_thr_best
    desk_points = all_points[desk_mask]

    # 只为打印一致性：按最佳参数复算一次 inliers
    n_tmp, d_tmp, inliers, _ = fit_plane_ransac(
        desk_points,
        threshold=best_info["threshold"],
        max_iters=max(3000, ransac_iters),
        min_inliers=max(30, int(desk_points.shape[0] * best_info["min_inlier_ratio"])),
        seed=123,
        refine=True,
    )
    if n_tmp[2] < 0:
        n_tmp, d_tmp = -n_tmp, -d_tmp

    print("\n桌面采样(最佳配置):")
    print(f"  Z 轴 percentile: {best_info['percentile']:.3f}%")
    print(f"  Z 阈值: {z_thr_best:.6f} m")
    print(f"  候选点数: {desk_points.shape[0]}")
    print(f"  内点数: {inliers.sum()} / {desk_points.shape[0]} (ratio={inliers.mean():.3f})")

    # ---- 3.5 force_horizontal ----
    plane_n, plane_d, used_horizontal, h_reason, h_extra = maybe_force_horizontal(
        plane_n,
        plane_d,
        best_info,
        desk_points,
        force_horizontal_mode=force_horizontal,
        auto_median_deg_th=auto_h_median_deg,
        auto_p90_deg_th=auto_h_p90_deg,
        up_axis=up_axis,
        z_stat=horizontal_stat,
    )

    print(f"\n水平约束模式: {force_horizontal} | used_horizontal={used_horizontal} | reason={h_reason}")
    if used_horizontal and h_extra:
        print(f"  horizontal abs median dist: {h_extra['horizontal_fit_abs_median']:.6f} m")
        print(f"  horizontal abs mean dist:   {h_extra['horizontal_fit_abs_mean']:.6f} m")

    print("\n========== 桌面平面拟合结果(最终) ==========")
    print(f"  法向量 n: [{plane_n[0]:.6f}, {plane_n[1]:.6f}, {plane_n[2]:.6f}]")
    print(f"  偏移量 d: {plane_d:.6f}")
    print(f"  平面方程: {plane_n[0]:.6f}*x + {plane_n[1]:.6f}*y + {plane_n[2]:.6f}*z + {plane_d:.6f} = 0")
    if abs(plane_n[2]) > 1e-6:
        z_at_origin = -plane_d / plane_n[2]
        print(f"  原点处桌面 z 值: {z_at_origin:.6f} m")

    # ---- 4. 高度计算 ----
    print("\n正在计算每个点相对于桌面的绝对高度...")

    robot0_height = plane_signed_distance(robot0_pos, plane_n, plane_d)
    robot1_height = plane_signed_distance(robot1_pos, plane_n, plane_d) if has_robot1 else None

    print("\n========== 高度统计 (signed distance to desk plane) ==========")
    print("  正值 = 桌面上方, 负值 = 桌面下方\n")

    print("  Robot0:")
    print(f"    范围: [{robot0_height.min():.6f}, {robot0_height.max():.6f}] m")
    print(f"    均值: {robot0_height.mean():.6f} m")
    print(f"    中位数: {np.median(robot0_height):.6f} m")
    print(f"    标准差: {robot0_height.std():.6f} m")

    if has_robot1:
        print("  Robot1:")
        print(f"    范围: [{robot1_height.min():.6f}, {robot1_height.max():.6f}] m")
        print(f"    均值: {robot1_height.mean():.6f} m")
        print(f"    中位数: {np.median(robot1_height):.6f} m")
        print(f"    标准差: {robot1_height.std():.6f} m")

    # ---- 5. 按 Episode 统计 ----
    print("\n========== 每个 Episode 的高度统计 ==========")
    if has_robot1:
        print(f"{'Ep':>4s} | {'Start':>6s}-{'End':>6s} | "
              f"{'R0 min':>8s} {'R0 mean':>8s} {'R0 max':>8s} | "
              f"{'R1 min':>8s} {'R1 mean':>8s} {'R1 max':>8s}")
        print("-" * 90)
    else:
        print(f"{'Ep':>4s} | {'Start':>6s}-{'End':>6s} | "
              f"{'R0 min':>8s} {'R0 mean':>8s} {'R0 max':>8s}")
        print("-" * 55)

    ep_stats = []
    for i, (s, e) in enumerate(episodes):
        r0h = robot0_height[s:e]
        stat = {
            'episode': i,
            'start': s,
            'end': e,
            'robot0_height_min': float(r0h.min()),
            'robot0_height_mean': float(r0h.mean()),
            'robot0_height_max': float(r0h.max()),
        }
        line = (f"  {i:2d} | {s:6d}-{e:6d} | "
                f"{r0h.min():8.4f} {r0h.mean():8.4f} {r0h.max():8.4f}")
        if has_robot1:
            r1h = robot1_height[s:e]
            stat.update({
                'robot1_height_min': float(r1h.min()),
                'robot1_height_mean': float(r1h.mean()),
                'robot1_height_max': float(r1h.max()),
            })
            line += f" | {r1h.min():8.4f} {r1h.mean():8.4f} {r1h.max():8.4f}"
        ep_stats.append(stat)
        print(line)

    # ---- 6. 保存 ----
    if output_dir is None:
        zarr_stem = Path(zarr_zip_path).name.replace('.zarr.zip', '')
        output_dir = f"/root/openpi-umi/data/analyze/analyze_desk_height/{zarr_stem}_desk_height_analysis"
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join("/root/openpi-umi/data/analyze/analyze_desk_height", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "desk_height_results.npz")

    save_dict = {
        'plane_normal': plane_n.astype(np.float64),
        'plane_d': np.array(plane_d, dtype=np.float64),
        'robot0_height': robot0_height.astype(np.float32),
        'robot0_eef_pos': robot0_pos.astype(np.float32),
        'episode_ends': np.array(meta['episode_ends']),
        'z_percentile_input': np.array(z_percentile, dtype=np.float32),
        'ransac_threshold_input': np.array(ransac_threshold, dtype=np.float32),
        'ransac_iters_input': np.array(ransac_iters, dtype=np.int32),
    }

    save_dict.update({
        'desk_fit_best_percentile': np.array(best_info['percentile'], dtype=np.float32),
        'desk_fit_best_threshold': np.array(best_info['threshold'], dtype=np.float32),
        'desk_fit_best_min_inlier_ratio': np.array(best_info['min_inlier_ratio'], dtype=np.float32),
        'desk_fit_inlier_ratio': np.array(best_info['inlier_ratio'], dtype=np.float32),
        'desk_fit_mad': np.array(best_info['mad'], dtype=np.float32),
        'desk_fit_rmse': np.array(best_info['rmse'], dtype=np.float32),
        'desk_fit_coverage': np.array(best_info['coverage'], dtype=np.float32),
        'desk_fit_normal_stability_median_deg': np.array(best_info['normal_stability_deg_median'], dtype=np.float32),
        'desk_fit_normal_stability_p90_deg': np.array(best_info['normal_stability_deg_p90'], dtype=np.float32),
        'desk_fit_z_threshold': np.array(best_info['z_threshold'], dtype=np.float32),
    })

    save_dict.update({
        'force_horizontal_mode': np.array(force_horizontal),
        'force_horizontal_used': np.array(1 if used_horizontal else 0, dtype=np.int32),
        'force_horizontal_reason': np.array(h_reason),
        'up_axis': np.array(up_axis),
        'horizontal_stat': np.array(horizontal_stat),
        'auto_h_median_deg': np.array(auto_h_median_deg, dtype=np.float32),
        'auto_h_p90_deg': np.array(auto_h_p90_deg, dtype=np.float32),
    })
    if used_horizontal and h_extra:
        save_dict['horizontal_fit_abs_median'] = np.array(h_extra['horizontal_fit_abs_median'], dtype=np.float32)
        save_dict['horizontal_fit_abs_mean'] = np.array(h_extra['horizontal_fit_abs_mean'], dtype=np.float32)

    if has_robot1:
        save_dict['robot1_height'] = robot1_height.astype(np.float32)
        save_dict['robot1_eef_pos'] = robot1_pos.astype(np.float32)

    np.savez(output_file, **save_dict)

    # ---- 6.5 将相对桌面高度写回数据集（可选）----
    if write_back_path:
        root_rw = zarr.open(_tmp_dir, mode="a")
        data_rw = root_rw["data"]
        data_rw["robot0_eef_height_desk"] = robot0_height.astype(np.float32)
        if has_robot1:
            data_rw["robot1_eef_height_desk"] = robot1_height.astype(np.float32)
        # 将修改后的 zarr 目录打包为新的 .zarr.zip
        with zipfile.ZipFile(write_back_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_dir, _dirs, files in os.walk(_tmp_dir):
                for f in files:
                    full = os.path.join(root_dir, f)
                    arcname = os.path.relpath(full, _tmp_dir)
                    zf.write(full, arcname)
        print(f"\n相对桌面高度已写回数据集: {write_back_path}")
        print("  新增字段: robot0_eef_height_above_desk" + (" , robot1_eef_height_above_desk" if has_robot1 else ""))
    fields = list(save_dict.keys())
    print(f"\n结果已保存到: {output_file}")
    print(f"  包含字段: {', '.join(fields)}")

    # ---- 可视化 ----
    try:
        _plot_height_distribution(robot0_height, robot1_height, episodes, output_dir)
        _plot_height_over_time(robot0_height, robot1_height, episodes, output_dir)
        _plot_3d_with_plane(robot0_pos, robot1_pos, plane_n, plane_d, desk_points, output_dir)
    except Exception as ex:
        print(f"  可视化生成失败（不影响数据结果）: {ex}")

    return plane_n, plane_d, robot0_height, (robot1_height if has_robot1 else None)


# ===================== 可视化 =====================

def _plot_height_distribution(r0h, r1h, episodes, output_dir):
    """绘制高度分布直方图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_plots = 2 if r1h is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_data = [('Robot0', r0h)]
    if r1h is not None:
        plot_data.append(('Robot1', r1h))

    for ax, (label, heights) in zip(axes, plot_data):
        ax.hist(heights, bins=100, edgecolor='black', alpha=0.7, density=True, color='steelblue')
        ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Desk plane (h=0)')
        ax.axvline(heights.mean(), color='orange', linewidth=1.5, linestyle='-',
                   label=f'Mean={heights.mean():.4f}m')
        ax.set_xlabel('Height above desk (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} Height Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('End-effector Height Relative to Desk Plane', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'height_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def _plot_height_over_time(r0h, r1h, episodes, output_dir):
    """绘制每个 Episode 的高度时序图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_eps = len(episodes)
    n_cols = min(4, max(1, n_eps))
    n_rows = (n_eps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    if n_eps == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, (s, e) in enumerate(episodes):
        ax = axes[i]
        t = np.arange(e - s)
        ax.plot(t, r0h[s:e], label='Robot0', alpha=0.8, linewidth=0.8)
        if r1h is not None:
            ax.plot(t, r1h[s:e], label='Robot1', alpha=0.8, linewidth=0.8)
        ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_title(f'Episode {i}', fontsize=9)
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Height (m)', fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_eps, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Height Above Desk Over Time (per Episode)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'height_over_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


def _plot_3d_with_plane(r0_pos, r1_pos, plane_n, plane_d, desk_pts, output_dir):
    """绘制 3D 散点图 + 拟合平面"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    n_sample0 = min(3000, r0_pos.shape[0])
    idx0 = np.random.default_rng(0).choice(r0_pos.shape[0], n_sample0, replace=False)
    ax.scatter(r0_pos[idx0, 0], r0_pos[idx0, 1], r0_pos[idx0, 2],
               s=1, alpha=0.3, c='blue', label='Robot0')

    if r1_pos is not None:
        n_sample1 = min(3000, r1_pos.shape[0])
        idx1 = np.random.default_rng(1).choice(r1_pos.shape[0], n_sample1, replace=False)
        ax.scatter(r1_pos[idx1, 0], r1_pos[idx1, 1], r1_pos[idx1, 2],
                   s=1, alpha=0.3, c='green', label='Robot1')

    if desk_pts is not None and desk_pts.shape[0] > 0:
        show = min(5000, desk_pts.shape[0])
        didx = np.random.default_rng(2).choice(desk_pts.shape[0], show, replace=False)
        ax.scatter(desk_pts[didx, 0], desk_pts[didx, 1], desk_pts[didx, 2],
                   s=3, alpha=0.6, c='red', label='Desk candidate points')

    all_pts = np.vstack([r0_pos, r1_pos]) if r1_pos is not None else r0_pos
    x_range = np.linspace(all_pts[:, 0].min(), all_pts[:, 0].max(), 20)
    y_range = np.linspace(all_pts[:, 1].min(), all_pts[:, 1].max(), 20)
    xx, yy = np.meshgrid(x_range, y_range)

    if abs(plane_n[2]) > 1e-6:
        zz = -(plane_n[0] * xx + plane_n[1] * yy + plane_d) / plane_n[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='orange')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory with Fitted Desk Plane')
    ax.legend(loc='upper right', fontsize=8)

    path = os.path.join(output_dir, '3d_with_desk_plane.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description='分析 UMI 数据集中的桌面平面并计算末端执行器的相对高度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--zarr', type=str,
        default='/root/openpi-umi/data/dataset_no_filter_1.zarr.zip',
        help='zarr.zip 文件路径 (默认: /root/openpi-umi/data/dataset_no_filter_1.zarr.zip)',
    )
    parser.add_argument(
        '--percentile', type=float, default=1.0,
        help='用于构建扫描网格的中心 z 百分位 (默认: 1%%)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.005,
        help='RANSAC 内点距离阈值（米）中心值 (默认: 0.005)',
    )
    parser.add_argument(
        '--iters', type=int, default=3000,
        help='RANSAC 最大迭代次数 (默认: 3000)',
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='输出目录（默认: /root/openpi-umi/data/analyze/analyze_desk_height/<zarr_stem>_desk_height_analysis）',
    )
    parser.add_argument(
        '--force_horizontal',
        type=str,
        default='auto',
        choices=['off', 'on', 'auto'],
        help='是否强制使用水平桌面平面: off/on/auto (默认: off)',
    )
    parser.add_argument(
        '--up_axis',
        type=str,
        default='z',
        choices=['x', 'y', 'z'],
        help='force_horizontal 使用的“向上轴” (默认: z)',
    )
    parser.add_argument(
        '--horizontal_stat',
        type=str,
        default='median',
        choices=['median', 'mean', 'trimmed_mean'],
        help='force_horizontal 估计桌面高度时使用的统计量 (默认: median)',
    )
    parser.add_argument(
        '--auto_h_median_deg',
        type=float,
        default=5.0,
        help='force_horizontal=auto 时，normal 稳定性 median 触发阈值(度)',
    )
    parser.add_argument(
        '--auto_h_p90_deg',
        type=float,
        default=10.0,
        help='force_horizontal=auto 时，normal 稳定性 p90 触发阈值(度)',
    )
    parser.add_argument(
        '--write_back',
        type=str,
        default=None,
        metavar='PATH',
        help='将计算后的相对桌面高度写回数据集并保存为新 .zarr.zip 文件路径；双臂时写 robot0/robot1_eef_height_above_desk，单臂时仅写 robot0_eef_height_above_desk',
    )
    args = parser.parse_args()

    try:
        analyze_desk_height(
            zarr_zip_path=args.zarr,
            z_percentile=args.percentile,
            ransac_threshold=args.threshold,
            ransac_iters=args.iters,
            output_dir=args.output,
            force_horizontal=args.force_horizontal,
            up_axis=args.up_axis,
            horizontal_stat=args.horizontal_stat,
            auto_h_median_deg=args.auto_h_median_deg,
            auto_h_p90_deg=args.auto_h_p90_deg,
            write_back_path=args.write_back,
        )
    finally:
        shutil.rmtree("/tmp/zarr_desk_height", ignore_errors=True)


if __name__ == "__main__":
    main()
