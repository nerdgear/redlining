import numpy as np
import tensorflow as tf


EPS = tf.constant(1e-12, dtype=tf.float32)
DEFAULT_JITTER = tf.constant(1e-5, dtype=tf.float32)


@tf.function
def compute_weighted_square_distances(x1, x2, lengthscales):

    assert x1.shape[1] == x2.shape[1]

    z1 = x1 / tf.expand_dims(lengthscales, axis=0)
    z2 = x2 / tf.expand_dims(lengthscales, axis=0)

    cross_contrib = -2 * tf.matmul(z1, tf.transpose(z2))

    z1_sq = tf.reduce_sum(z1 ** 2, axis=1)
    z2_sq = tf.reduce_sum(z2 ** 2, axis=1)

    combined = (
        tf.expand_dims(z1_sq, axis=1) + cross_contrib + tf.expand_dims(z2_sq, axis=0)
    )

    # This seems required as some elements can become smaller than zero.
    # Would be nice to fix this another way & make sure all OK.
    return tf.maximum(combined, 0.0)


@tf.function
def compute_diag_weighted_square_distance(x1, x2, lengthscales):

    z1 = x1 / tf.expand_dims(lengthscales, axis=0)
    z2 = x2 / tf.expand_dims(lengthscales, axis=0)

    max_len = min(int(z1.shape[0]), int(z2.shape[0]))

    z1 = z1[:max_len]
    z2 = z2[:max_len]

    cross_terms = -2 * tf.reduce_sum(z1 * z2, axis=1)
    norms = tf.reduce_sum(z1 ** 2, axis=1) + tf.reduce_sum(z2 ** 2, axis=1)

    return tf.maximum(norms + cross_terms, 0.0)


@tf.function
def ard_rbf_kernel(x1, x2, lengthscales, alpha, jitter=DEFAULT_JITTER, diag_only=False):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    kernel = alpha ** 2 * tf.exp(-0.5 * r_sq)

    kernel = add_jitter(kernel, jitter, diag_only)

    return kernel


@tf.function
def matern_kernel_32(
    x1, x2, alpha, lengthscales, jitter=DEFAULT_JITTER, diag_only=False
):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    r = tf.sqrt(r_sq + EPS)

    kernel = alpha ** 2 * (1 + tf.sqrt(3.0) * r) * tf.exp(-tf.sqrt(3.0) * r)
    kernel = add_jitter(kernel, jitter, diag_only)

    return kernel


@tf.function
def matern_kernel_12(
    x1, x2, alpha, lengthscales, jitter=DEFAULT_JITTER, diag_only=False
):

    if diag_only:

        r_sq = compute_diag_weighted_square_distance(x1, x2, lengthscales)

    else:

        r_sq = compute_weighted_square_distances(x1, x2, lengthscales)

    r = np.sqrt(r_sq + EPS)

    kernel = alpha ** 2 * np.exp(-r)
    kernel = add_jitter(kernel, jitter, diag_only)

    return kernel


@tf.function
def add_jitter(kern, jitter=DEFAULT_JITTER, diag_only=False):

    if diag_only:

        kern = kern + jitter

    else:

        kern = tf.linalg.set_diag(kern, tf.linalg.diag_part(kern) + jitter)

    return kern


def ard_rbf_kernel_old(x1, x2, lengthscales, alpha, jitter=DEFAULT_JITTER):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)

    # Must have same number of dimensions
    assert x1.get_shape()[1] == x2.get_shape()[1]

    # Also must match lengthscales
    assert lengthscales.get_shape()[0] == x1.get_shape()[1]

    # Use broadcasting
    # X1 will be (N1, 1, D)
    x1_expanded = tf.expand_dims(x1, axis=1)
    # X2 will be (1, N2, D)
    x2_expanded = tf.expand_dims(x2, axis=0)

    # These will be N1 x N2 x D
    scaled_diffs = ((x1_expanded - x2_expanded) / lengthscales) ** 2

    # Use broadcasting to do a dot product
    exponent = tf.reduce_sum(scaled_diffs, axis=2)

    kern = alpha ** 2 * tf.exp(-0.5 * exponent)

    # Jitter this a little bit
    kern = tf.linalg.set_diag(kern, tf.linalg.diag_part(kern) + jitter)

    return kern


@tf.function
def bias_kernel(x1, x2, sd, jitter=DEFAULT_JITTER, diag_only=False):

    output_rows = int(x1.shape[0])
    output_cols = int(x2.shape[0])

    shape = tf.stack([output_rows, output_cols])

    if diag_only:
        kern = tf.fill((min(output_cols, output_rows),), sd ** 2) + jitter
    else:
        kern = tf.fill(shape, sd ** 2)
        kern = tf.linalg.set_diag(kern, tf.linalg.diag_part(kern) + jitter)

    return kern


def ard_rbf_kernel_batch(x1, x2, lengthscales, alpha, jitter=DEFAULT_JITTER):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)
    # lengthscales is B x D [B is batch dim]
    # alpha is B,

    # Must have same number of dimensions
    assert x1.get_shape()[1] == x2.get_shape()[1]

    # Also must match lengthscales
    assert lengthscales.get_shape()[1] == x1.get_shape()[1]

    l_expanded = tf.expand_dims(lengthscales, axis=1)

    # Divide x1 by lengthscales
    # Gives (B x N x D)
    x1 = x1 / l_expanded
    x2 = x2 / l_expanded

    # This will be (B x N)
    x1s = tf.reduce_sum(tf.square(x1), axis=2)
    x2s = tf.reduce_sum(tf.square(x2), axis=2)

    # This produces an (N x N) matrix
    cross_prods = -2 * tf.matmul(x1, x2, transpose_b=True)

    # This should produce a (B x N x N) distance mat
    dist = cross_prods + tf.expand_dims(x1s, axis=2) + tf.expand_dims(x2s, axis=1)

    # Multiply all of these
    kern = tf.expand_dims(tf.expand_dims(alpha ** 2, axis=1), axis=1) * tf.exp(
        -0.5 * dist
    )

    # Jitter this a little bit
    kern = tf.matrix_set_diag(kern, tf.matrix_diag_part(kern) + jitter)

    return kern


def random_intercept_kernel(x1, x2, sd, jitter=DEFAULT_JITTER, diag_only=False):
    # x1, x2 must be vectors
    # returns sd^2 if they are the same
    # 0 otherwise.

    assert len(x1.shape) == 1

    output_rows = int(x1.shape[0])
    output_cols = int(x2.shape[0])
    diag_length = min(output_rows, output_cols)

    if diag_only:
        return tf.fill((diag_length,), sd ** 2) + jitter
    else:

        # Perhaps there is a more efficient way but I'll use broadcasting for
        # now
        x1 = tf.expand_dims(x1, axis=1)
        x2 = tf.expand_dims(x2, axis=0)

        is_equal = tf.equal(x1, x2)
        float_version = tf.cast(is_equal, sd.dtype)

        float_version = tf.linalg.set_diag(
            float_version, tf.linalg.diag_part(float_version) + jitter
        )

        return float_version * sd ** 2


@tf.function
def additive_rbf_kernel(
    x1, x2, lengthscales, alpha, jitter=DEFAULT_JITTER, diag_only=False
):

    n_c = x1.shape[1]

    if diag_only:
        min_size = min(x1.shape[0], x2.shape[0])
        x1 = x1[:min_size]
        x2 = x2[:min_size]
        sq_dists = (x1 - x2) ** 2
    else:
        x1_exp = tf.expand_dims(x1, axis=1)
        x2_exp = tf.expand_dims(x2, axis=0)
        sq_dists = (x1_exp - x2_exp) ** 2

    # Weight them
    weighted_sq_dists = sq_dists / lengthscales ** 2

    # Exponentiate them
    exp_versions = tf.exp(-weighted_sq_dists / 2)

    # Sum these
    summed = tf.reduce_sum(exp_versions, axis=-1)

    version = (alpha ** 2 / n_c) * summed

    if diag_only:
        version += jitter
    else:
        # Add a bit of jitter
        version = add_jitter(version, jitter)

    return version


def linear_kernel_ard(x1, x2, prior_variances, jitter=DEFAULT_JITTER, diag_only=False):

    if diag_only:

        min_size = min(x1.shape[0], x2.shape[0])
        x1, x2 = x1[:min_size], x2[:min_size]
        diag_result = tf.einsum("ik,k,ik->i", x1, prior_variances, x2)

        return diag_result + jitter

    else:

        result = tf.einsum("ik,k,jk->ij", x1, prior_variances, x2)
        result = add_jitter(result, jitter)

        return result
