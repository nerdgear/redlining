# TODO: Maybe add an ARD prior.
import autograd.numpy as np
from autograd.scipy.stats import norm
from autograd import hessian, jacobian
from scipy.optimize import minimize


def independent_normal_prior_fun(w, mean=0., sd=1.):

    return np.sum(norm.logpdf(w, loc=mean, scale=sd))


def calculate_log_likelihood(x, w, y):

    latent_pred = x @ w

    log_lik_elementwise = (y * norm.logcdf(latent_pred) + (1 - y) *
                           norm.logcdf(-latent_pred))

    return np.sum(log_lik_elementwise)


def fit(x, y, prior_fun=None, add_intercept=True):

    def to_minimize(w):

        if prior_fun is not None:
            prior_contribution = prior_fun(w)
        else:
            prior_contribution = 0.

        neg_log_posterior = -(calculate_log_likelihood(x, w, y) +
                              prior_contribution)

        return neg_log_posterior

    if add_intercept:
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

    n_cov = x.shape[1]
    start_w = np.zeros(n_cov)

    jac = jacobian(to_minimize)
    hess = hessian(to_minimize)

    result = minimize(to_minimize, start_w, jac=jac, hess=hess,
                      method='Newton-CG')

    assert(result.success)

    return result.x, np.linalg.inv(hess(result.x))
