import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def e_step(em_mus, em_sigmas, em_log_pi, data):

    K = em_mus.shape[0]
    N = data.shape[0]

    gamma_unnormalised = np.zeros((K, N))

    for k in range(K):

        cur_mvn = multivariate_normal(em_mus[k], em_sigmas[k])
        cur_lik = cur_mvn.logpdf(data)
        cur_log_pi = em_log_pi[k]

        gamma_unnormalised[k] = cur_log_pi + cur_lik

    # Now, use the log_sum_exp to normalise.
    summed = logsumexp(gamma_unnormalised, axis=0)

    log_gamma = gamma_unnormalised - summed

    return log_gamma


def m_step(log_gamma, data):

    N = data.shape[0]
    M = data.shape[1]
    K = log_gamma.shape[0]

    N_k = logsumexp(log_gamma, axis=1)

    gamma_subtracted = np.exp(log_gamma - N_k.reshape(-1, 1))
    new_mu = np.einsum("ki,ij->kj", gamma_subtracted, data)

    diffs = data.reshape(1, N, M) - new_mu.reshape(K, 1, M)

    new_sigmas = np.einsum("ki,kij,kil->kjl", gamma_subtracted, diffs, diffs)

    # Finally, the new pis
    log_pi_new = N_k - np.log(N)

    return new_mu, new_sigmas, log_pi_new


def log_likelihood(log_pi, mus, sigmas, data):

    K = mus.shape[0]
    N = data.shape[0]

    gammas_unnormalised = np.zeros((K, N))

    for k in range(K):

        gammas_unnormalised[k] = log_pi[k] + multivariate_normal(
            mus[k], sigmas[k]
        ).logpdf(data)

    gammas_summed = logsumexp(gammas_unnormalised, axis=0)

    return np.sum(gammas_summed)


def fit_em(init_mus, init_sigmas, init_log_pi, data, callback=None, tol=1e-8):
    """
    Fits a mixture of multivariate normals using expectation maximisation [EM].

    In the following, K is the number of clusters, M is the dimensionality
    of the observations, and N is the number of data points.

    Args:
        init_mus: The initial means. This is an [K x M] numpy array.
        init_sigmas: The initial covariances. This is an [K x M x M] numpy
            array. Each has to be positive definite.
        init_log_pi: The initial prior probability for each class label, on
            the log scale; a numpy array of shape [K,].
        data: The observations to fit. This should be an array of size [N, M].
        callback: If provided, this function will be called at each step with
            the current iteration, the current means, covariances, log prior
            probabilities, and log likelihoods up to that point. This can
            be used e.g. for plotting or printing.
        tol: The tolerance for assessing convergence. If the absolute
            difference from one step to the next is less than this value,
            the estimates are returned.

    Returns:
        A tuple of four elements: the optimal means, covariances, log prior
        class probabilities, and a list of the observed data log likelihoods.
    """

    em_mus, em_sigmas, em_log_pi = init_mus, init_sigmas, init_log_pi

    prev_log_lik = -1
    cur_log_lik = 0
    iteration = 0

    log_liks = list()

    while iteration <= 1 or np.abs(cur_log_lik - prev_log_lik) > tol:

        prev_log_lik = cur_log_lik

        # E step
        log_gamma = e_step(em_mus, em_sigmas, em_log_pi, data)

        # M step
        em_mus, em_sigmas, em_log_pi = m_step(log_gamma, data)

        # Likelihood calculation
        cur_log_lik = log_likelihood(em_log_pi, em_mus, em_sigmas, data)

        log_liks.append(cur_log_lik)

        if callback is not None:
            callback(iteration, em_mus, em_sigmas, em_log_pi, log_liks)

        iteration += 1

    return em_mus, em_sigmas, em_log_pi, log_liks


def plot_mvn(mu, sigma, xlims, ylims):
    # Plots a multivariate normal in 2D.

    to_eval = np.meshgrid(
        np.linspace(xlims[0], xlims[1], 100), np.linspace(ylims[0], ylims[1], 100)
    )

    stacked = np.stack([to_eval[0].reshape(-1), to_eval[1].reshape(-1)], axis=1)

    sample_mvn = multivariate_normal(mean=mu, cov=sigma)

    prob = sample_mvn.pdf(stacked)
    reshaped_prob = np.reshape(prob, (100, 100))

    return to_eval[0], to_eval[1], reshaped_prob
