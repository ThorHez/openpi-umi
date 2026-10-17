"""论文 Ours/UOT 的纯 JAX 实现。

逻辑：视觉距离与机器人状态距离共同决定非平衡最优传输计划；计划停止梯度，
训练梯度只通过计划加权后的视觉距离回到 π0.5 原有视觉编码器。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _squared_cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    """计算稳定的两两平方欧氏距离。"""
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    distance = jnp.sum(jnp.square(x), axis=-1)[:, None] + jnp.sum(jnp.square(y), axis=-1)[None, :]
    return jnp.maximum(distance - 2.0 * (x @ y.T), 0.0)


def unbalanced_sinkhorn_plan(
    cost: jax.Array,
    *,
    entropy_reg: float,
    marginal_relaxation: float,
    iterations: int,
) -> jax.Array:
    """在 log 域求解 KL 边缘松弛的 UOT 运输计划。"""
    if entropy_reg <= 0 or marginal_relaxation <= 0:
        raise ValueError("entropy_reg and marginal_relaxation must be positive")
    rows, cols = cost.shape
    log_a = -jnp.log(jnp.asarray(rows, dtype=jnp.float32)) * jnp.ones((rows,), dtype=jnp.float32)
    log_b = -jnp.log(jnp.asarray(cols, dtype=jnp.float32)) * jnp.ones((cols,), dtype=jnp.float32)
    log_kernel = -cost.astype(jnp.float32) / entropy_reg
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
    source_features: jax.Array,
    target_features: jax.Array,
    source_labels: jax.Array,
    target_labels: jax.Array,
    *,
    embedding_cost_scale: float,
    label_cost_scale: float,
    entropy_reg: float,
    marginal_relaxation: float,
    sinkhorn_iterations: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """计算论文联合训练中的视觉 UOT 损失及可观测指标。"""
    embedding_cost = embedding_cost_scale * _squared_cdist(source_features, target_features)
    label_cost = label_cost_scale * _squared_cdist(source_labels, target_labels)
    matching_cost = embedding_cost + label_cost

    # 论文实现先根据当前特征求运输计划，再将该计划视作伪标签；不反传 Sinkhorn 迭代。
    plan = jax.lax.stop_gradient(
        unbalanced_sinkhorn_plan(
            jax.lax.stop_gradient(matching_cost),
            entropy_reg=entropy_reg,
            marginal_relaxation=marginal_relaxation,
            iterations=sinkhorn_iterations,
        )
    )
    alignment_loss = jnp.sum(plan * embedding_cost)
    metrics = {
        "uot_loss": alignment_loss,
        "uot_plan_mass": jnp.sum(plan),
        "uot_embedding_cost": jnp.mean(embedding_cost),
        "uot_label_cost": jnp.mean(label_cost),
    }
    return alignment_loss, metrics
