from jax.scipy.stats import multivariate_normal
from jax import vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
from jax import jit


def compute_responsibilities(mu, sigma, log_w, sigma_bars, ys):

    N = ys.shape[0]

    tiled_mu = jnp.tile(mu, (N, 1))

    result = vmap(multivariate_normal.logpdf)(ys, tiled_mu, sigma + sigma_bars)

    result = result + log_w

    return result


def compute_single_w(cur_sigma_bar, cur_sigma):

    return cur_sigma @ jnp.linalg.inv(cur_sigma + cur_sigma_bar)


def compute_w_i(sigma_bars, cur_sigma):

    cur_rel = partial(compute_single_w, cur_sigma=cur_sigma)

    W_in = vmap(cur_rel)(sigma_bars)

    return W_in


def compute_sufficient(cur_w_i, cur_mu, cur_sigma, ys):

    dim = cur_sigma.shape[0]

    x_in = jnp.einsum("ijk,ik->ij", cur_w_i, ys - cur_mu) + cur_mu

    # Compute Rhat
    first_term = vmap(jnp.outer)(x_in, x_in)

    second_term_first = jnp.eye(dim).reshape(1, dim, dim) - cur_w_i
    # This is unclear!
    second_term_second = cur_sigma

    second_term = second_term_first @ second_term_second

    Rxx = first_term + second_term

    return x_in, Rxx


@jit
def e_step(cur_mu, cur_sigma, cur_log_w, sigma_bars, ys):

    # E step
    curried_resp = partial(compute_responsibilities, sigma_bars=sigma_bars, ys=ys)
    all_resp = vmap(curried_resp)(cur_mu, cur_sigma, cur_log_w)
    log_gammas = all_resp - logsumexp(all_resp, axis=0)
    all_w_i = vmap(lambda cur_sigma: compute_w_i(sigma_bars, cur_sigma))(cur_sigma)
    x_in, Rxx = vmap(partial(compute_sufficient, ys=ys))(all_w_i, cur_mu, cur_sigma)

    # Compute the log lik
    gamma_summed = logsumexp(all_resp, axis=0)
    log_lik = jnp.sum(gamma_summed)

    return all_w_i, log_gammas, x_in, Rxx, log_lik


@jit
def m_step(log_gammas, x_in, Rxx):

    N = log_gammas.shape[1]
    K = log_gammas.shape[0]

    summed_log_gammas = logsumexp(log_gammas, axis=1)
    log_w_i = summed_log_gammas - jnp.log(N)

    mu_i_log_summands = jnp.exp(jnp.expand_dims(log_gammas, axis=-1)) * x_in
    mu_i = jnp.exp(-summed_log_gammas).reshape(K, 1) * jnp.sum(
        mu_i_log_summands, axis=1
    )

    sigma_i_summands = (
        jnp.exp(jnp.expand_dims(jnp.expand_dims(log_gammas, axis=-1), axis=-1)) * Rxx
    )
    sigma_i_sum = jnp.sum(sigma_i_summands, axis=1)

    sigma_i_first_term = jnp.exp(-summed_log_gammas).reshape(K, 1, 1) * sigma_i_sum

    sigma_i = sigma_i_first_term - vmap(jnp.outer)(mu_i, mu_i)

    return log_w_i, mu_i, sigma_i


def fit_em(sigma_bars, ys, init_mu, init_sigma, init_log_w, steps):
    """
    Implements the approach in Ozerov, Lagrange, Vincent (2011) to fit a mixture
    of Gaussians to data measured with noise using an EM algorithm.

    Args:
    sigma_bars: The errors associated with each measurement.
    ys: The observed values.
    init_mu: A KxD array of initial values for the means of the MVNs.
    init_sigma: A KxDxD array of initial values for the covariances of the MVNs.
    init_log_w: A K array of the initial cluster weights.
    steps: The steps to run EM for.

    Returns:
    A dictionary containing the optimal mu, sigma and log_w values, as well as
    the sequence of observed data log likelihoods, which should be strictly
    increasing (up to numerical errors).
    """

    cur_mu, cur_sigma, cur_log_w = init_mu, init_sigma, init_log_w

    log_liks = list()

    for i in range(steps):

        all_w_i, log_gammas, x_in, Rxx, log_lik = e_step(
            cur_mu, cur_sigma, cur_log_w, sigma_bars, ys
        )

        # Update
        cur_log_w, cur_mu, cur_sigma = m_step(log_gammas, x_in, Rxx)

        log_liks.append(log_lik)

    return {
        "log_w": cur_log_w,
        "mu": cur_mu,
        "sigma": cur_sigma,
        "log_lik": jnp.array(log_liks),
    }
