import numpy as np

EPS = 1e-12


def rbf_kernel_1d(x1, x2, alpha, lengthscales, jitter=1e-5):

    differences = np.subtract.outer(x1, x2)
    sq_diff = differences**2

    result = alpha**2 * np.exp(-sq_diff / (2 * lengthscales**2))

    result = add_jitter(result, jitter)

    return result


def ard_rbf_kernel(x1, x2, lengthscales, alpha, jitter=1e-5):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)

    # Must have same number of dimensions
    assert(x1.shape[1] == x2.shape[1])

    # Also must match lengthscales
    assert(lengthscales.shape[0] == x1.shape[1])

    # Use broadcasting
    # X1 will be (N1, 1, D)
    x1_expanded = np.expand_dims(x1, axis=1)
    # X2 will be (1, N2, D)
    x2_expanded = np.expand_dims(x2, axis=0)

    # These will be N1 x N2 x D
    sq_differences = (x1_expanded - x2_expanded)**2
    inv_sq_lengthscales = 1. / lengthscales**2

    # Use broadcasting to do a dot product
    exponent = np.sum(sq_differences * inv_sq_lengthscales, axis=2)
    exponentiated = np.exp(-0.5 * exponent)

    kern = alpha**2 * exponentiated
    diag_indices = np.diag_indices(np.min(kern.shape[:2]))
    kern[diag_indices] = kern[diag_indices] + jitter

    # Find gradients
    # Gradient with respect to alpha:
    alpha_grad = 2 * alpha * exponentiated

    # Gradient with respect to lengthscales
    # Square differences should be [N1 x N2 x D]
    lengthscale_grads = (alpha**2 * np.expand_dims(exponentiated, axis=2) *
                         sq_differences / (lengthscales**3))

    return kern, lengthscale_grads, alpha_grad


def compute_weighted_square_distances(x1, x2, lengthscales):

    z1 = x1 / np.expand_dims(lengthscales, axis=0)
    z2 = x2 / np.expand_dims(lengthscales, axis=0)

    # Matrix part
    cross_contrib = -2 * np.dot(z1, z2.T)

    # Other bits
    z1_sq = np.sum(z1**2, axis=1)
    z2_sq = np.sum(z2**2, axis=1)

    # Sum it all up
    combined = (np.expand_dims(z1_sq, axis=1) + cross_contrib +
                np.expand_dims(z2_sq, axis=0))

    return combined


def ard_rbf_kernel_efficient(x1, x2, alpha, lengthscales, jitter=1e-5):

    combined = compute_weighted_square_distances(x1, x2, lengthscales)
    kernel = alpha**2 * np.exp(-0.5 * combined)
    kernel = add_jitter(kernel, jitter)

    return kernel


def add_jitter(kernel, jitter=1e-5):

    # Add the jitter
    diag_indices = np.diag_indices(np.min(kernel.shape[:2]))
    to_add = np.zeros_like(kernel)
    to_add[diag_indices] += jitter
    kernel = kernel + to_add

    return kernel


def matern_kernel_32(x1, x2, alpha, lengthscales, jitter=1e-5):

    r_sq = compute_weighted_square_distances(x1, x2, lengthscales)
    r = np.sqrt(r_sq + EPS)

    kernel = alpha ** 2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    kernel = add_jitter(kernel, jitter)

    return kernel


def matern_kernel_12(x1, x2, alpha, lengthscales, jitter=1e-5):

    r_sq = compute_weighted_square_distances(x1, x2, lengthscales)
    r = np.sqrt(r_sq)

    kernel = alpha**2 * np.exp(-r)
    kernel = add_jitter(kernel, jitter)

    return kernel


def brownian_kernel_1d(x1, x2, alpha, jitter=1e-5):

    assert(x1.shape[1] == 1 and x2.shape[1] == 1)

    variance = alpha ** 2

    kernel = variance * np.where(np.sign(x1) == np.sign(x2.T),
                                 np.fmin(np.abs(x1), np.abs(x2.T)), 0.)

    kernel = add_jitter(kernel, jitter)

    return kernel


def mlp_kernel(x1, x2, variance, weight_variance, bias_variance, jitter=1e-5):

    four_over_tau = 2. / np.pi

    def comp_prod(x1, x2=None):

        if x2 is None:
            return ((np.square(x1) * weight_variance).sum(axis=1) +
                    bias_variance)
        else:
            return (x1 * weight_variance).dot(x2.T) + bias_variance

    x1_denom = np.sqrt(comp_prod(x1) + 1.)
    x2_denom = np.sqrt(comp_prod(x2) + 1.)
    xtx = comp_prod(x1, x2) / x1_denom[:, None] / x2_denom[None, :]
    kern = variance * four_over_tau * np.arcsin(xtx)
    kern = add_jitter(kern, jitter)

    return kern


def rq_kernel(x1, x2, variance, lscales, alpha, jitter=1e-5):

    dists = compute_weighted_square_distances(x1, x2, lscales)

    # Divide by alpha
    divided = dists / alpha
    result = (1 + divided)**(-alpha)
    result = variance * result
    result = add_jitter(result, jitter)

    return result


def additive_rbf_kernel(x1, x2, lengthscales, alpha, jitter=1e-5,
                        diag_only=False):

    n_c = x1.shape[1]

    if diag_only:
        min_size = min(x1.shape[0], x2.shape[0])
        x1 = x1[:min_size]
        x2 = x2[:min_size]
        sq_dists = (x1 - x2)**2
    else:
        x1_exp = np.expand_dims(x1, axis=1)
        x2_exp = np.expand_dims(x2, axis=0)
        sq_dists = (x1_exp - x2_exp)**2

    # Weight them
    weighted_sq_dists = sq_dists / lengthscales**2

    # Exponentiate them
    exp_versions = np.exp(-weighted_sq_dists / 2)

    # Sum these
    summed = np.sum(exp_versions, axis=-1)

    version = (alpha**2 / n_c) * summed

    if diag_only:
        version += jitter
    else:
        # Add a bit of jitter
        version = add_jitter(version, jitter)

    return version
