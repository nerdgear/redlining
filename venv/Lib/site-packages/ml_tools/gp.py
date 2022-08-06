import scipy.linalg as spl
from sklearn.cluster import MiniBatchKMeans, KMeans


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = spl.solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = spl.solve_triangular(k_chol.T, s)

    return b


def find_starting_z(X, num_inducing, use_minibatching=False):
    # Find starting locations for inducing points

    if use_minibatching:
        k_means = MiniBatchKMeans(n_clusters=num_inducing)
    else:
        k_means = KMeans(n_clusters=num_inducing)

    k_means.fit(X)
    Z = k_means.cluster_centers_

    return Z
