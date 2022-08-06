import autograd.numpy as np


def ard_rbf_kernel_efficient(x1, x2, alpha, lengthscales, jitter=1e-5):

    combined = compute_weighted_square_distances(x1, x2, lengthscales)
    kernel = alpha**2 * np.exp(-0.5 * combined)
    kernel = add_jitter(kernel, jitter)

    return kernel


def matern_kernel_32(x1, x2, alpha, lengthscales, jitter=1e-5):

    r_sq = compute_weighted_square_distances(x1, x2, lengthscales)
    r = np.sqrt(r_sq)

    kernel = alpha ** 2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    kernel = add_jitter(kernel, jitter)

    return kernel


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


def add_jitter(kernel, jitter=1e-5):

    # Add the jitter
    diag_indices = np.diag_indices(np.min(kernel.shape[:2]))
    to_add = np.zeros_like(kernel)
    to_add[diag_indices] += jitter
    kernel = kernel + to_add

    return kernel
