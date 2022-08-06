import jax.numpy as jnp
from .jax_kernels import ard_rbf_kernel
from scipy.optimize import minimize
from jax import value_and_grad
from functools import partial
import jax.scipy.linalg as spl
from jax import jit


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = spl.solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = spl.solve_triangular(k_chol.T, s)

    return b


def fit_gp_regression(X, y, X_predict, kernel_fun, obs_var, pred_kernel_fun=None):

    if pred_kernel_fun is None:
        pred_kernel_fun = kernel_fun

    k_xstar_x = pred_kernel_fun(X_predict, X)
    k_xx = kernel_fun(X, X)
    obs_mat = jnp.diag(obs_var * jnp.ones(X.shape[0]))
    k_xstar_xstar = pred_kernel_fun(X_predict, X_predict)

    k_chol = jnp.linalg.cholesky(k_xx + obs_mat)

    pred_mean = k_xstar_x @ solve_via_cholesky(k_chol, y)
    pred_cov = k_xstar_xstar - k_xstar_x @ solve_via_cholesky(k_chol, k_xstar_x.T)

    return pred_mean, pred_cov


def gp_regression_marginal_likelihood(
    X, y, kernel_fun, obs_var, solve_fun=jnp.linalg.solve
):
    # I am omitting the n term. Should I include it? It's constant.

    obs_var_full = jnp.diag(obs_var * jnp.ones(X.shape[0]))
    k_xx = kernel_fun(X, X)

    k_chol = jnp.linalg.cholesky(k_xx + obs_var_full)

    det_term = -0.5 * jnp.linalg.slogdet(k_xx + obs_var_full)[1]
    data_term = -0.5 * y @ solve_via_cholesky(k_chol, y)

    return det_term + data_term


def map_smooth_data_1d(
    X,
    y,
    X_pred,
    kernel_fun=ard_rbf_kernel,
    standardise_y=True,
    prior_k=3,
    prior_theta=1 / 3,
    jitter=1e-5,
):

    y_mean = y.mean()
    y_std = y.std()

    if standardise_y:

        y = (y - y_mean) / y_std

    def to_optimize(theta):

        alpha, lscale, obs_var = theta ** 2

        cur_k = lambda x1, x2: kernel_fun(
            x1, x2, jnp.array([lscale]), alpha, False, jitter
        )

        marg_lik = gp_regression_marginal_likelihood(X, y, cur_k, obs_var)

        prior_contrib = (prior_k - 1) * jnp.log(lscale) - lscale / prior_theta

        return -marg_lik - prior_contrib

    to_opt_with_grad = jit(value_and_grad(to_optimize))

    opt_result = minimize(
        to_opt_with_grad, [1.0, 1.0, 1.0], jac=True, method="BFGS", tol=1e-3
    )

    alpha, lscale, obs_var = opt_result.x ** 2

    final_k = lambda x1, x2: kernel_fun(
        x1, x2, jnp.array([lscale]), alpha, False, jitter
    )

    # Predict
    pred_mean, pred_cov = fit_gp_regression(X, y, X_pred, final_k, obs_var)
    pred_var = jnp.diag(pred_cov)

    if standardise_y:

        pred_mean = (pred_mean * y_std) + y_mean
        pred_var = pred_var * y_std ** 2

    return pred_mean, pred_var, opt_result
