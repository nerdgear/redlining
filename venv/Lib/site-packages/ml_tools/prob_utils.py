import numpy as np
from scipy.special import comb, logsumexp
from scipy.stats import poisson
from scipy.special import gammaln, xlog1py, xlogy


def prob_m_missing(m, r, n):
    # m is number of empty cells
    # r is number of balls distributed across ...
    # ... n cells.
    pre_factor = comb(n, m)

    # Compute the sum
    nu = np.arange(0, n - m + 1)

    summand = (-1) ** nu * comb(n - m, nu) * (1 - (m + nu) / n) ** r

    return pre_factor * np.sum(summand)


def log_n_choose_k(n, k):

    log_factorial_n = gammaln(n + 1)
    log_factorial_k = gammaln(k + 1)
    log_factorial_nmk = gammaln(n - k + 1)

    return log_factorial_n - log_factorial_k - log_factorial_nmk


def log_prob_m_missing(m, r, n):

    first_term = log_n_choose_k(n, m)
    sum_range = np.arange(0, n - m + 1)
    sum_term_1 = log_n_choose_k(n - m, sum_range)
    sum_term_2 = r * np.log(1 - (m + sum_range) / n)
    signs = (-1) ** sum_range
    log_result = first_term + logsumexp(sum_term_1 + sum_term_2, b=signs)

    return log_result


def prob_m_missing_poisson_approx(m, r, n, log=False):

    l = n * np.exp(-r / n)

    if log:
        return poisson.logpmf(m, l)
    else:
        return poisson.pmf(m, l)


def prob_observe_nobs(n_obs, t, N, use_poisson_approx=True, log=True):
    # Probability of observing
    # n_obs different species when
    # t is the total observations made and
    # N is the total number of species at a site.

    # Convert to match the missing terminology:
    n = N
    r = t
    m = N - n_obs

    if use_poisson_approx:
        return prob_m_missing_poisson_approx(m, r, n, log=log)
    else:
        if log:
            return log_prob_m_missing(m, r, n)
        else:
            return prob_m_missing(m, r, n)


# def compute_posterior_over_counts(
#     t, n_obs, n_max, use_poisson_approx=True, prior_fn=None
# ):
#
#     if prior_fn is None:
#         prior_fn = lambda _: 1.0
#
#     N = np.arange(1, n_max + 1)
#
#     # assert use_poisson_approx, "Exact version not yet implemented!"
#
#     priors = prior_fn(N)
#     priors = priors / np.sum(priors)
#
#     if use_poisson_approx:
#         likelihoods = prob_observe_nobs(n_obs, t, N, use_poisson_approx=True)
#     else:
#         likelihoods = np.array(
#             [
#                 prob_observe_nobs(n_obs, t, cur_N, use_poisson_approx=False)
#                 for cur_N in N
#             ]
#         )
#
#         # TODO: Is there a way to avoid this?
#         likelihoods[np.isnan(likelihoods)] = 0.0
#
#     # Check whether this is OK:
#     posteriors = (priors * likelihoods) / np.sum(priors * likelihoods)
#
#     if np.any(np.isnan(posteriors)):
#         import ipdb
#
#         ipdb.set_trace()
#
#     return posteriors


def compute_log_posterior_over_counts(
    t, n_obs, n_max, use_poisson_approx=False, log_prior_fn=None
):

    N = np.arange(n_obs, n_max + 1)

    # TODO: The Poisson approx can be computed more quickly.
    if use_poisson_approx:
        log_probs = prob_observe_nobs(n_obs, t, N, log=True, use_poisson_approx=True)
    else:
        log_probs = np.array(
            [
                prob_observe_nobs(n_obs, t, cur_N, log=True, use_poisson_approx=False)
                for cur_N in N
            ]
        )

    # normalised_lik = log_probs - logsumexp(log_probs)

    if log_prior_fn is None:
        normalised_prior = np.zeros_like(log_probs)
    else:
        prior_probs_raw = log_prior_fn(N)
        normalised_prior = prior_probs_raw - logsumexp(prior_probs_raw)

    summed = log_probs + normalised_prior
    log_posterior = summed - logsumexp(summed)

    return log_posterior


def compute_expected_prob_and_count(
    t, n_obs, n_max, use_poisson_approx=False, log_prior_fn=None
):

    N = np.arange(n_obs, n_max + 1)

    log_posterior = compute_log_posterior_over_counts(
        t, n_obs, n_max, use_poisson_approx, log_prior_fn
    )

    expected_prob = np.sum((1 / N) * np.exp(log_posterior))
    expected_count = np.sum(N * np.exp(log_posterior))

    return expected_prob, expected_count
