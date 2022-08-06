import jax.numpy as np
import numpy as np_classic
from jax import random
from jax.scipy.stats import norm


def dual_averaging_step(t, xbart, h, mu, log_r, gamma=0.05, t0=10, kappa=0.75,
                        delta=0.65):

    # This is the deviation from the target
    cur_h = delta - np.exp(np.minimum(0, log_r))
    h = np.append(h, cur_h)

    # Calculate xtp1
    xtp1 = mu - ((np.sqrt(t) / gamma) * (1 / (t + t0)) * np.sum(h))
    eta = t**(-kappa)

    xbartp1 = eta * xtp1 + (1 - eta) * xbart
    eps = np.exp(xbartp1)

    return eps, xbartp1, h


def acceptance_step(log_r, theta, start_theta, key):

    was_accepted = False

    if log_r > 0:
        # Accept -- probability is greater than 1
        new_theta = theta
        was_accepted = True
        # We don't need to update the momentum
    else:
        accept_prob = np.exp(log_r)
        key, subkey = random.split(key)
        outcome = random.bernoulli(key, mean=accept_prob)

        print(f'Accept prob is {accept_prob}. Outcome was {outcome}.')

        if outcome:
            new_theta = theta
            was_accepted = True
        else:
            new_theta = start_theta

    return was_accepted, new_theta


def run_hmc_steps(theta, eps, Lmax, key, log_posterior,
                  log_posterior_grad_theta, diagonal_mass_matrix):
    # Diagonal mass matrix: diagonal entries of M (a vector)

    inverse_diag_mass = 1. / diagonal_mass_matrix

    key, subkey = random.split(key)

    # Location-scale transform to get the right variance
    # TODO: Check!
    phi = random.normal(
        subkey, shape=(theta.shape[0],)) * np.sqrt(diagonal_mass_matrix)

    start_theta = theta
    start_phi = phi

    cur_grad = log_posterior_grad_theta(theta)

    key, subkey = random.split(key)

    L = np_classic.random.randint(1, Lmax)

    for cur_l in range(L):
        phi = phi + 0.5 * eps * cur_grad
        theta = theta + eps * inverse_diag_mass * phi
        cur_grad = log_posterior_grad_theta(theta)
        phi = phi + 0.5 * eps * cur_grad

    # Compute (log) acceptance probability
    proposed_log_post = log_posterior(theta)
    previous_log_post = log_posterior(start_theta)

    proposed_log_phi = np.sum(norm.logpdf(
        phi, scale=np.sqrt(diagonal_mass_matrix)))
    previous_log_phi = np.sum(norm.logpdf(
        start_phi, scale=np.sqrt(diagonal_mass_matrix)))

    print(f'Proposed log posterior is: {proposed_log_post}.'
          f'Previous was {previous_log_post}.')

    if (np.isinf(proposed_log_post) or np.isnan(proposed_log_post) or
            np.isneginf(proposed_log_post)):
        # Reject
        was_accepted = False
        new_theta = start_theta
        # FIXME: What number to put here?
        log_r = -10
        return was_accepted, log_r, new_theta

    log_r = (proposed_log_post + proposed_log_phi -
             previous_log_post - previous_log_phi)

    was_accepted, new_theta = acceptance_step(log_r, theta, start_theta, key)

    return was_accepted, log_r, new_theta
