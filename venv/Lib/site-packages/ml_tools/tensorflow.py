import numpy as np
import tensorflow as tf


def newton_optimize(start_f, fun, jac, hess, solve_fun=tf.linalg.solve,
                    tolerance=1e-5, debug=False, float_dtype=tf.float64):

    # TODO: Consider adding a maxiter
    # FIXME: The float casts are egregious.
    # TODO: Also, we need to make sure that the shapes are as expected.
    def body(f, difference):

        cur_hess = hess(f)

        # Ensure jac is a (column) vector
        cur_jac = tf.reshape(jac(f), (-1, 1))

        sol = tf.squeeze(solve_fun(cur_hess, cur_jac))

        new_f = f - sol

        if debug:
            # TODO: Not the neatest -- is there another way?
            print_tensors = [tf.print(cur_hess), tf.print(f)]
            hess_evals, _ = tf.linalg.eigh(cur_hess)
            print_tensors += [tf.print(tf.reduce_max(hess_evals) /
                                       tf.reduce_min(hess_evals))]
            with tf.control_dependencies(print_tensors):
                difference = tf.linalg.norm(f - new_f)
        else:
            difference = tf.linalg.norm(f - new_f)

        return (new_f, difference)

    init_val = (start_f, tf.constant(1., dtype=float_dtype))

    result = tf.while_loop(lambda f, difference: difference > tolerance, body,
                           init_val)

    return result[0]


def lo_tri_from_elements(elements, n):
    # Elements are the elements to include
    # n is the size of the lower triangular matrix

    indices = np.array(np.tril_indices(n)).T
    L = tf.scatter_nd(indices, elements, (n, n))

    return L


def rep_vector(vector, n_rep):

    # Replicates vector and stacks along first axis.
    # For example, inputs vector = [1, 2, 3] and n_rep = 2
    # should return [[1, 2, 3], [1, 2, 3]], a matrix of size (2, 3).

    assert len(vector.shape) == 1, 'rep_vector only defined for 1D inputs!'

    return tf.reshape(tf.tile(vector, [n_rep]), (n_rep, -1))


def rep_matrix(matrix, n_rep):

    # Replicates matrix and stacks along first axis.

    assert len(matrix.shape) == 2, 'rep_matrix only defined for 2D inputs!'

    return tf.reshape(tf.tile(matrix, [n_rep, 1]),
                      (n_rep, matrix.shape[0], -1))


def solve_via_cholesky(k_chol, y):

    s = tf.linalg.triangular_solve(k_chol, y, lower=True)
    b = tf.linalg.triangular_solve(tf.transpose(k_chol), s, lower=False)

    return b


def weighted_sum(mean, cov, weights):
    """
    Computes mean and variance of a weighted sum of the mvn r.v.
    Args:
        mean (np.array): The mean of the MVN.
        cov (np.array): The covariance of the MVN.
        weights (np.array): A vector of weights to give the elements.
    Returns:
        Tuple[float, float]: The mean and variance of the weighted sum.
    """

    mean_summed_theta = tf.einsum('i,i->', mean,  weights)

    outer_x = tf.einsum('i,j->ij', weights, weights)
    multiplied = cov * outer_x
    weighted_sum = tf.reduce_sum(multiplied)

    return mean_summed_theta, weighted_sum


def logistic_normal_integral_approx(mu, var):
    """
    Approximates the logistic normal integral, E[logit^{-1}(X)], where
    X ~ N(mu, var).
    """

    gamma = tf.sqrt(1 + (np.pi * (var / 8)))

    return tf.sigmoid(mu / gamma)


# From
# https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def covar_to_corr_and_scales(covar):

    scales = tf.sqrt(tf.linalg.diag_part(covar))
    diag_inv_scales = tf.linalg.diag(1 / scales)
    corr = (diag_inv_scales) @ covar @ (diag_inv_scales)

    return corr, scales
