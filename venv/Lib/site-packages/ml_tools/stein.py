# A collection of useful functions for Stein Variational Gradient Descent
import tensorflow as tf


def distance_matrix(A):
    # Calculates the distance matrix between all entries of A.
    # Returns matrix D whose entries Dij are the distance from row i to row j.
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    return D


def rbf_kernel(X, h=None):
    # Calculates the RBF kernel matrix for X
    # Also returns the gradients of the RBF kernel as defined in the Stein
    # paper, and the length scale it uses.

    distances = distance_matrix(X)

    if h is None:
        h = tf.contrib.distributions.percentile(distances, 50.)**2 / tf.log(
                tf.cast(tf.shape(X)[0], tf.float32))

    kernel = tf.exp(-distances / h)

    # Calculate the kernel derivative
    first_part = X * tf.expand_dims(tf.reduce_sum(kernel, axis=0), axis=1)
    second_part = tf.matmul(kernel, X)

    gradient = 2. / h * (first_part - second_part)

    return kernel, gradient


def calculate_log_prob_gradients(log_prob, x):
    # Calculates the gradients of each element of `log_prob` with respect to
    # each element of x.
    # Note: This uses a TF while loop; maybe there is a way to optimise it.

    def cond(_, i):

        return i < tf.shape(log_prob)[0]

    def body(output, i):

        cur_log_prob = log_prob[i]
        cur_gradient = tf.gradients(cur_log_prob, x)[0][i]
        output = output.write(i, cur_gradient)
        i = tf.add(i, 1)

        return output, i

    output = tf.TensorArray(tf.float32, x.get_shape()[0])
    i = tf.zeros((), tf.int32)
    final_output, _ = tf.while_loop(cond, body, (output, i))
    return final_output.stack()


def calculate_update(x, log_prob, kernel_fun=rbf_kernel):
    # Calculates the update for particles x according to SVGD.
    # Note: This calculates the quantity phi_hat_star from the paper. It needs
    # to be pre-multiplied by the step size before being used in an update.

    # Define the kernel
    kernel, kernel_gradients = kernel_fun(x)

    n_particles = tf.cast(tf.shape(log_prob)[0], tf.float32)

    log_prob_gradients = calculate_log_prob_gradients(log_prob, x)

    first_term = tf.matmul(kernel, log_prob_gradients)
    second_term = kernel_gradients

    combined = 1. / n_particles * (first_term + second_term)

    return combined
