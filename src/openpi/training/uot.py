"""论文 Ours/UOT 的纯 JAX 实现。

逻辑：视觉距离与机器人状态距离共同决定非平衡最优传输计划；计划停止梯度，
训练梯度只通过计划加权后的视觉距离回到 π0.5 原有视觉编码器。
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _squared_cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    """计算稳定的两两平方欧氏距离。"""
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    distance = jnp.sum(jnp.square(x), axis=-1)[:, None] + jnp.sum(jnp.square(y), axis=-1)[None, :]
    return jnp.maximum(distance - 2.0 * (x @ y.T), 0.0)


def _dimension_normalized_squared_cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    """Squared L2 normalized by block width so differently sized blocks have equal weight."""
    return _squared_cdist(x, y) / x.shape[-1]


def _equal_block_cost(
    source: Mapping[str, jax.Array], target: Mapping[str, jax.Array], block_names: tuple[str, ...]
) -> jax.Array:
    if set(source) != set(target):
        raise ValueError(f"Source/target blocks differ: {set(source)} != {set(target)}")
    missing = set(block_names) - set(source)
    if missing:
        raise ValueError(f"Missing required blocks: {sorted(missing)}")
    costs = [_dimension_normalized_squared_cdist(source[name], target[name]) for name in block_names]
    return sum(costs) / len(costs)


def unbalanced_sinkhorn_plan(
    cost: jax.Array,
    *,
    entropy_reg: float,
    marginal_relaxation: float,
    iterations: int,
    diagonal_reference: bool,
) -> jax.Array:
    """在 log 域复现官方 POT ``sinkhorn_knopp_unbalanced`` 的 KL 正则 UOT。"""
    if entropy_reg <= 0 or marginal_relaxation <= 0:
        raise ValueError("entropy_reg and marginal_relaxation must be positive")
    rows, cols = cost.shape
    log_a = -jnp.log(jnp.asarray(rows, dtype=jnp.float32)) * jnp.ones((rows,), dtype=jnp.float32)
    log_b = -jnp.log(jnp.asarray(cols, dtype=jnp.float32)) * jnp.ones((cols,), dtype=jnp.float32)
    # === [论文 Ours/UOT] 官方实现默认以 a⊗b 为 KL 参考测度；启用 heuristic 时改用 I/N。 ===
    # 当前 DTW loader 的第 i 个 source 与第 i 个 target 是一组时序配对，因此主实验使用
    # 官方 base_config.json 中 heuristic=true 对应的对角参考测度。
    if diagonal_reference:
        if rows != cols:
            raise ValueError("Official diagonal UOT reference requires equal source and target batch sizes")
        log_reference = jnp.where(
            jnp.eye(rows, dtype=jnp.bool_),
            -jnp.log(jnp.asarray(rows, dtype=jnp.float32)),
            -jnp.inf,
        )
    else:
        log_reference = log_a[:, None] + log_b[None, :]
    log_kernel = -cost.astype(jnp.float32) / entropy_reg + log_reference
    exponent = marginal_relaxation / (marginal_relaxation + entropy_reg)

    def update(_, potentials):
        log_u, log_v = potentials
        log_u = exponent * (log_a - jsp.special.logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = exponent * (log_b - jsp.special.logsumexp(log_kernel.T + log_u[None, :], axis=1))
        return log_u, log_v

    log_u, log_v = jax.lax.fori_loop(
        0,
        iterations,
        update,
        (jnp.zeros((rows,), dtype=jnp.float32), jnp.zeros((cols,), dtype=jnp.float32)),
    )
    return jnp.exp(log_u[:, None] + log_kernel + log_v[None, :])


def compute_uot_alignment_loss(
    source_features: Mapping[str, jax.Array],
    target_features: Mapping[str, jax.Array],
    source_labels: Mapping[str, jax.Array],
    target_labels: Mapping[str, jax.Array],
    *,
    embedding_cost_scale: float,
    label_cost_scale: float,
    entropy_reg: float,
    marginal_relaxation: float,
    sinkhorn_iterations: int,
    diagonal_reference: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """计算论文联合训练中的视觉 UOT 损失及可观测指标。"""
    # 每个同名相机保留完整网格并展平后计算平方 L2；所有相机等权。
    camera_names = tuple(sorted(source_features))
    if not camera_names or set(camera_names) != set(target_features):
        raise ValueError("Source and target must expose the same non-empty camera set")
    embedding_cost = embedding_cost_scale * _equal_block_cost(
        source_features, target_features, camera_names
    )
    # translation、SO(3) rotation matrix、absolute gripper 三块按维度归一后等权。
    label_cost = label_cost_scale * _equal_block_cost(
        source_labels, target_labels, ("translation", "rotation", "gripper")
    )
    matching_cost = embedding_cost + label_cost

    # 论文实现先根据当前特征求运输计划，再将该计划视作伪标签；不反传 Sinkhorn 迭代。
    plan = jax.lax.stop_gradient(
        unbalanced_sinkhorn_plan(
            jax.lax.stop_gradient(matching_cost),
            entropy_reg=entropy_reg,
            marginal_relaxation=marginal_relaxation,
            iterations=sinkhorn_iterations,
            diagonal_reference=diagonal_reference,
        )
    )
    # === [论文 Ours/UOT] 与官方开源实现一致，外层损失使用完整 ground cost <Pi, M>。 ===
    # label 不依赖模型参数，因此仍只有视觉 cost 向编码器反传梯度。
    alignment_loss = jnp.sum(plan * matching_cost)
    metrics = {
        "uot_loss": alignment_loss,
        "uot_plan_mass": jnp.sum(plan),
        "uot_embedding_cost": jnp.mean(embedding_cost),
        "uot_label_cost": jnp.mean(label_cost),
    }
    return alignment_loss, metrics
