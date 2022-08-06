import numpy as np
from scipy.stats import t
import pandas as pd


def estimate_mean_variance_normal_inverse_gamma(mu_0, nu, alpha, beta, x):

    x_bar = x.mean()
    n = x.shape[0]

    posterior_mu = (nu * mu_0 + n * x_bar) / (nu + n)
    posterior_nu = nu + n

    posterior_alpha = alpha + n / 2
    posterior_beta = (
        beta
        + 0.5 * np.sum(x - x_bar) ** 2
        + 0.5 * (n * nu) / (nu + n) * (x_bar - mu_0) ** 2 / 2
    )

    return posterior_mu, posterior_nu, posterior_alpha, posterior_beta


def normal_unknown_mean_variance_noninformative(y_bar, s_sq, n):
    # y_bar: Sample mean
    # s_sq: Sample variance
    # n: number of observations

    # Posterior for mean is student t with:
    df, mean, variance = (n - 1, y_bar, s_sq / n)

    return t(df, loc=mean, scale=np.sqrt(variance))


def normal_unknown_mean_variance_noninformative_from_vector(vec):

    y_bar = vec.mean()
    s_sq = vec.std() ** 2
    n = vec.shape[0]

    return normal_unknown_mean_variance_noninformative(y_bar, s_sq, n)


def summarise_distribution(scipy_dist):

    return pd.Series(
        {
            "median": scipy_dist.ppf(0.5),
            "2.5%": scipy_dist.ppf(0.025),
            "97.5%": scipy_dist.ppf(0.975),
            "mean": scipy_dist.mean(),
        }
    )
