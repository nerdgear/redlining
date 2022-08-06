import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.ops import index_add
from .additive_kernels_jax import newton_girard_combination
from jax import vmap

EPS = 1e-12
DEFAULT_JITTER = 1e-5


@jit
def compute_weighted_square_distances(x1, x2, lengthscales):

    z1 = x1 / jnp.expand_dims(lengthscales, axis=0)
    z2 = x2 / jnp.expand_dims(lengthscales, axis=0)

    cross_contrib = -2 * z1 @ z2.T

    z1_sq = jnp.sum(z1 ** 2, axis=1)
    z2_sq = jnp.sum(z2 ** 2, axis=1)

    combined = (
        jnp.expand_dims(z1_sq, axis=1) + cross_contrib + jnp.expand_dims(z2_sq, axis=0)
    )

    # This seems required as some elements can become smaller than zero.
    # Would be nice to fix this another way & make sure all OK.
    return jnp.maximum(combined, 0.0)


@jit
def compute_diag_weighted_square_distance(x1, x2, lengthscales):

    z1 = x1 / jnp.expand_dims(lengthscales, axis=0)
    z2 = x2 / jnp.expand_dims(lengthscales, axis=0)

    max_len = min(int(z1.shape[0]), int(z2.shape[0]))

    z1 = z1[:max_len]
    z2 = z2[:max_len]

    cross_terms = -2 * jnp.sum(z1 * z2, axis=1)
    norms = jnp.sum(z1 ** 2, axis=1) + jnp.sum(z2 ** 2, axis=1)

    return jnp.maximum(norms + cross_terms, 0.0)


def ard_rbf_kernel(x1, x2, lengthscales, alpha, diag_only=False, jitter=DEFAULT_JITTER):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    kernel = alpha ** 2 * jnp.exp(-0.5 * r_sq)

    kernel = add_jitter(kernel, jitter, diag_only)

    return kernel


def matern_kernel_32(
    x1, x2, lengthscales, alpha, diag_only=False, jitter=DEFAULT_JITTER
):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    r = jnp.sqrt(r_sq + EPS)

    kernel = alpha ** 2 * (1 + jnp.sqrt(3.0) * r) * jnp.exp(-jnp.sqrt(3.0) * r)
    kernel = add_jitter(kernel, jitter, diag_only)

    return kernel


def matern_kernel_12(x1, x2, alpha, lengthscales, diag_only=False, jitter=1e-5):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    r = jnp.sqrt(r_sq + EPS)

    kernel = alpha ** 2 * jnp.exp(-r)
    kernel = add_jitter(kernel, jitter)

    return kernel


def add_jitter(kern, jitter=DEFAULT_JITTER, diag_only=False):

    if diag_only:

        kern = kern + jitter

    else:

        kern = index_add(
            kern, jnp.diag_indices(min(kern.shape[0], kern.shape[1])), jitter
        )

    return kern


def bias_kernel(x1, x2, sd, diag_only=False, jitter=DEFAULT_JITTER):

    output_rows = x1.shape[0]
    output_cols = x2.shape[0]

    # shape = jnp.stack([output_rows, output_cols])

    if diag_only:
        kern = jnp.repeat(sd ** 2, min(output_cols, output_rows)) + jitter
    else:
        # kern = jnp.tile(sd ** 2, shape)
        kern = jnp.ones((output_rows, output_cols)) * sd ** 2
        kern = add_jitter(kern, jitter, diag_only)

    return kern


def additive_kernel(
    x1,
    x2,
    lengthscales,
    additive_alphas,
    kernel_alphas,
    base_kernel_fun,
    diag_only,
    jitter=DEFAULT_JITTER,
):

    N = additive_alphas.shape[0]

    # TODO: Could make more general to support other kernels
    to_vmap = lambda x1, x2, lengthscale, alpha: base_kernel_fun(
        x1.reshape(-1, 1),
        x2.reshape(-1, 1),
        lengthscale.reshape(
            -1,
        ),
        alpha,
        diag_only,
        jitter,
    )

    map_res = vmap(to_vmap)(x1.T, x2.T, lengthscales, kernel_alphas)

    girard_res = newton_girard_combination(map_res, N)

    kernel_res = jnp.tensordot(additive_alphas, girard_res, axes=(0, 0))

    return kernel_res


def linear_kernel_ard(x1, x2, prior_variances, jitter=DEFAULT_JITTER, diag_only=False):

    if diag_only:

        min_size = min(x1.shape[0], x2.shape[0])
        x1, x2 = x1[:min_size], x2[:min_size]
        diag_result = jnp.einsum("ik,k,ik->i", x1, prior_variances, x2)

        return diag_result + jitter

    else:

        result = jnp.einsum("ik,k,jk->ij", x1, prior_variances, x2)
        result = add_jitter(result, jitter)

        return result


def raneff_kernel(z1, z2, raneff_var, diag_only=False, jitter=DEFAULT_JITTER):

    if diag_only:
        # Often (though we won't use it here), it can be useful to compute only the diagonal entries.
        smaller = jnp.minimum(z1.shape[0], z2.shape[0])
        kern = jnp.einsum("ik,ik->i", z1[:smaller], z2[:smaller])

    else:
        # This is what we really need.
        kern = raneff_var * z1 @ z2.T

    # Typically we add a little bit of noise to the diagonal for numerical stability.
    return add_jitter(kern, jitter=DEFAULT_JITTER, diag_only=diag_only)
