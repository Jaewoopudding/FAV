"""
ALIGEN loss in JAX (amortized MLE with SVGD).
Particles are replicated across devices so the pairwise SVGD kernel sees the
global set; inner grads run on stop_gradient'd copies.
"""
import jax
import jax.numpy as jnp


def _ensure_list(temp):
    if isinstance(temp, (int, float)):
        return [temp]
    return list(temp)


def _pairwise_sq_dist(a, b):
    """Squared L2 distance matrix: dist[i,j] = ||a[i] - b[j]||^2."""
    a_sq = jnp.sum(a ** 2, axis=-1, keepdims=True)   # (M, 1)
    b_sq = jnp.sum(b ** 2, axis=-1, keepdims=True).T  # (1, N)
    return jnp.maximum(a_sq + b_sq - 2 * (a @ b.T), 0.0)


def _gather_all(x):
    """Replicate *x* across all devices so every shard sees the full array."""
    try:
        from utils.hsdp_util import get_global_mesh
        from jax.sharding import NamedSharding, PartitionSpec as P
        mesh = get_global_mesh()
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    except Exception:
        return x  # single-device or mesh not initialised


def jax_amortized_mle_loss(
    gen,            # (B, D)  generated particles, differentiable w.r.t. model params
    ref,            # (B', D) reference particles  (should already be stop-gradiented)
    reward_fn,      # callable (features, params) -> (B,)
    reward_params,  # frozen reward-model params (dict)
    beta,           # reward scaling factor (multiplied onto nabla_r; larger = stronger reward)
    temp_kde,       # KDE temperature(s) — float or list[float]
    temp_stein,     # Stein kernel temperature(s) — float or list[float]
    gradient_estimator=None,  # if not None, use NES instead of first-order gradient
):
    """ALIGEN SVGD pseudo-loss: builds a Stein-variational target for ``gen`` and
    returns the MSE whose gradient pushes particles toward higher-reward,
    reference-matching regions."""
    temps_kde = _ensure_list(temp_kde)
    temps_stein = _ensure_list(temp_stein)

    global_ref = _gather_all(jax.lax.stop_gradient(ref))
    global_gen = _gather_all(jax.lax.stop_gradient(gen))
    n_global = global_gen.shape[0]
    gen_det = jax.lax.stop_gradient(gen)

    # Reward gradient nabla_x r(x).
    if gradient_estimator is not None:
        nabla_r = gradient_estimator(gen_det, reward_fn, reward_params)
    else:
        def _reward_sum(x):
            return reward_fn(x, reward_params).sum()
        nabla_r = jax.grad(_reward_sum)(gen_det)

    # Multi-scale KDE prior score.
    dist_pos = _pairwise_sq_dist(gen_det, global_ref)        # (B, B')
    score_num = jnp.zeros_like(gen_det)
    Z_total = jnp.zeros((gen_det.shape[0], 1))
    for t in temps_kde:
        k = jnp.exp(-dist_pos / t)                           # (B, B')
        Z_t = k.sum(axis=-1, keepdims=True)                  # (B, 1)
        Z_total = Z_total + Z_t
        score_num = score_num + 2.0 * (k @ global_ref - Z_t * gen_det) / t
    Z_total = jnp.maximum(Z_total, 1e-6)
    prior_score = score_num / Z_total                         # (B, D)

    # SVGD total score = prior + beta * reward grad.
    total_score_local = prior_score + nabla_r * beta
    global_total_score = _gather_all(jax.lax.stop_gradient(total_score_local))

    # Multi-scale Stein kernel, then driving + repulsive terms.
    dist_gen = _pairwise_sq_dist(gen_det, global_gen)         # (B, B_global)
    kernel_matrix = sum(jnp.exp(-dist_gen / t) for t in temps_stein)
    driving_term = (kernel_matrix @ global_total_score) / n_global

    def _kernel_sum_fn(x):
        d = _pairwise_sq_dist(x, global_gen)
        return sum(jnp.exp(-d / t) for t in temps_stein).sum()
    grad_k = jax.grad(_kernel_sum_fn)(gen_det)
    repulsive_term = -grad_k / (2.0 * n_global)

    stein_velocity = driving_term + repulsive_term
    target = jax.lax.stop_gradient(gen_det + stein_velocity)
    loss = jnp.mean((gen - target) ** 2)
    return loss


# Chunked FAV loss with full-K coupling: the O(K^2) particle interaction runs
# against a detached full-K global set while the autograd graph back to the
# generator is built for only a micro-chunk at a time. Summed per-chunk losses
# (trainer divides by n_accum) recover the monolithic per-class gradient.


def jax_fav_total_score(
    gen_global,        # (K, D) full-class generator features (detached)
    ref_global,        # (K, D) full-class reference features (detached)
    reward_fn,         # callable (features, params) -> (K,)
    reward_params,
    beta,
    temp_kde,
    gradient_estimator=None,
):
    """Per-particle SVGD total score (prior + beta * reward grad) over the full
    detached global set; no generator graph (inputs are detached features)."""
    temps_kde = _ensure_list(temp_kde)

    global_ref = _gather_all(jax.lax.stop_gradient(ref_global))
    gen_det = _gather_all(jax.lax.stop_gradient(gen_global))

    # Reward gradient nabla r(gen) (per-particle independent).
    if gradient_estimator is not None:
        nabla_r = gradient_estimator(gen_det, reward_fn, reward_params)
    else:
        def _reward_sum(x):
            return reward_fn(x, reward_params).sum()
        nabla_r = jax.grad(_reward_sum)(gen_det)

    # Multi-scale KDE prior score over the full reference set.
    dist_pos = _pairwise_sq_dist(gen_det, global_ref)
    score_num = jnp.zeros_like(gen_det)
    Z_total = jnp.zeros((gen_det.shape[0], 1))
    for t in temps_kde:
        k = jnp.exp(-dist_pos / t)
        Z_t = k.sum(axis=-1, keepdims=True)
        Z_total = Z_total + Z_t
        score_num = score_num + 2.0 * (k @ global_ref - Z_t * gen_det) / t
    Z_total = jnp.maximum(Z_total, 1e-6)
    prior_score = score_num / Z_total

    total_score = prior_score + nabla_r * beta
    return jax.lax.stop_gradient(total_score)


def jax_fav_loss_chunked(
    gen_chunk,             # (m, D) LIVE generator features — carries the graph
    gen_global,            # (K, D) detached full-class generator features
    total_score_global,    # (K, D) detached, from jax_fav_total_score
    temp_stein,
    n_global,              # K — the SVGD kernel normaliser
):
    """SVGD pseudo-loss for one chunk of generator particles against the detached
    full-K set."""
    temps_stein = _ensure_list(temp_stein)

    global_gen = _gather_all(jax.lax.stop_gradient(gen_global))
    gts = _gather_all(jax.lax.stop_gradient(total_score_global))
    gen_det = jax.lax.stop_gradient(gen_chunk)

    dist_gen = _pairwise_sq_dist(gen_det, global_gen)         # (m, K)
    kernel_matrix = sum(jnp.exp(-dist_gen / t) for t in temps_stein)
    driving_term = (kernel_matrix @ gts) / n_global           # (m, D)

    def _kernel_sum_fn(x):
        d = _pairwise_sq_dist(x, global_gen)
        return sum(jnp.exp(-d / t) for t in temps_stein).sum()
    grad_k = jax.grad(_kernel_sum_fn)(gen_det)
    repulsive_term = -grad_k / (2.0 * n_global)

    stein_velocity = driving_term + repulsive_term
    target = jax.lax.stop_gradient(gen_det + stein_velocity)
    return jnp.mean((gen_chunk - target) ** 2)


def nes_gradient_estimator(particles, reward_fn, reward_params,
                           sigma=0.01, n_samples=16, rng=None):
    """Antithetic zeroth-order NES estimate of the reward gradient:
    nabla_r approx (1/2N) sum_i [r(x+sigma*u_i) - r(x-sigma*u_i)] / sigma * u_i."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    B, D = particles.shape
    u = jax.random.normal(rng, (n_samples, B, D))

    z_plus = (particles[None] + sigma * u).reshape(-1, D)    # (N*B, D)
    z_minus = (particles[None] - sigma * u).reshape(-1, D)   # (N*B, D)

    r_plus = reward_fn(z_plus, reward_params).reshape(n_samples, B)
    r_minus = reward_fn(z_minus, reward_params).reshape(n_samples, B)

    diff = ((r_plus - r_minus) / (2.0 * sigma))[:, :, None]  # (N, B, 1)
    nabla_r = (diff * u).mean(axis=0)                         # (B, D)
    return nabla_r
