import numpy as np

def _fit_plane_svd(points3: np.ndarray):
    """
    用最小二乘(SVD)拟合平面: n·x + d = 0
    返回 (n_unit, d, centroid)
    """
    assert points3.ndim == 2 and points3.shape[1] == 3
    centroid = points3.mean(axis=0)
    X = points3 - centroid
    # SVD: 最小奇异向量对应法向量
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
    threshold: float = 0.003,      # 内点阈值：3mm（按你的噪声可改）
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

    # 预计算一点加速：也可不做
    idx_all = np.arange(N)

    for _ in range(max_iters):
        # 随机采样 3 个不共线点
        ids = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[ids]

        # 通过叉乘得到法向量
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        d = -np.dot(n, p1)

        # 计算距离并判内点
        signed_dist = plane_signed_distance(points, n, d)
        inliers = np.abs(signed_dist) < threshold
        score = int(inliers.sum())

        if score > best_score and score >= min_inliers:
            best_score = score
            best_inliers = inliers
            best_model = (n, d)

            # 可选：早停（比如内点比例很高）
            # if best_score > 0.9 * N:
            #     break

    if best_model is None:
        raise RuntimeError(
            f"RANSAC failed: no plane found with >= {min_inliers} inliers. "
            f"Try increasing max_iters, threshold, or providing cleaner points."
        )

    best_n, best_d = best_model
    best_signed = plane_signed_distance(points, best_n, best_d)

    # 精修：对最佳内点集用 SVD 再拟合
    if refine:
        inlier_pts = points[best_inliers]
        refined = _fit_plane_svd(inlier_pts)
        if refined is not None:
            best_n, best_d, _ = refined
            best_signed = plane_signed_distance(points, best_n, best_d)
            best_inliers = np.abs(best_signed) < threshold

    return best_n, float(best_d), best_inliers, best_signed


# ------------------ 使用示例 ------------------
if __name__ == "__main__":
    # 假设你从 UMI 里拿到了 (N,3) 的 eef 位置点，单位米
    # points = eef_pos  # shape (N,3)

    # 这里用合成数据演示
    rng = np.random.default_rng(42)
    # 真平面：z = 0.7 + 0.1x - 0.05y
    x = rng.uniform(-0.5, 0.5, 2000)
    y = rng.uniform(-0.5, 0.5, 2000)
    z = 0.7 + 0.1 * x - 0.05 * y + rng.normal(0, 0.001, size=x.shape)  # 1mm noise
    points = np.stack([x, y, z], axis=1)
    # 加一些离群点
    outliers = rng.uniform([-0.5, -0.5, 0.4], [0.5, 0.5, 1.2], size=(200, 3))
    points = np.concatenate([points, outliers], axis=0)

    n, d, inliers, signed = fit_plane_ransac(
        points,
        threshold=0.003,     # 3mm
        max_iters=1500,
        min_inliers=500,
        seed=0,
        refine=True,
    )

    print("Plane n:", n)
    print("Plane d:", d)
    print("signed:", signed)
    print("Inliers:", inliers.sum(), "/", points.shape[0])
    # 任何点的“相对桌面高度”可以用 signed distance（如果你把桌面点用于拟合）
    # h_table = signed  (桌面附近≈0，上方为正/负取决于 n 朝向)
