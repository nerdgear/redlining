import numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import cho_solve, cho_factor, solve_triangular
import jax.numpy as jnp
from scipy.linalg import eigh_tridiagonal


def cholesky_inverse(matrix):

    return cho_solve(cho_factor(matrix, lower=True), np.eye(matrix.shape[0]))


def compute_sparse_inverse(sparse_matrix):

    lu = spl.splu(sparse_matrix)
    inverse = lu.solve(np.eye(sparse_matrix.shape[0]))

    return inverse


def pos_def_mat_from_vector(vec, target_size, jitter=0):

    L = np.zeros((target_size, target_size))
    L[np.tril_indices(target_size)] = vec

    return np.matmul(L, L.T) + np.eye(target_size) * jitter


def vector_from_pos_def_mat(pos_def_mat, jitter=0):

    # Subtract jitter
    pos_def_mat -= np.eye(pos_def_mat.shape[0]) * jitter
    L = np.linalg.cholesky(pos_def_mat)
    elts = np.tril_indices_from(L)

    return L[elts]


def num_triangular_elts(mat_size, include_diagonal=True):

    if include_diagonal:
        return int(mat_size * (mat_size + 1) / 2)
    else:
        return int(mat_size * (mat_size - 1) / 2)


def solve_via_cholesky(k_chol, y):
    """Solves a positive definite linear system via a Cholesky decomposition.

    Args:
        k_chol: The Cholesky factor of the matrix to solve. A lower triangular
            matrix, perhaps more commonly known as L.
        y: The vector to solve.
    """

    # Solve Ls = y
    s = solve_triangular(k_chol, y, lower=True)

    # Solve Lt b = s
    b = solve_triangular(k_chol.T, s)

    return b


def generate_random_pos_def(n, jitter=10 ** -4):

    elements = np.random.randn(n, n)
    cov = elements @ elements.T + np.eye(n) * jitter

    return cov


def triple_matmul_with_diagonal_mat(A, B_elts, C):
    # B is diagonal.
    return jnp.einsum("ik,k,kj->ij", A, B_elts, C)


def log_determinant_diag_plus_low_rank_eigendecomp(A_elts, U, W_elts):
    # Computes the log of the determinant
    # |A + U W U^T|,
    # where A is diagonal, W is diagonal, and W's dimension is smaller than A's dimension.

    A_logdet = jnp.sum(jnp.log(A_elts))
    W_logdet = jnp.sum(jnp.log(W_elts))

    A_inv_elts = 1 / A_elts
    W_inv_elts = 1 / W_elts

    second_term_in_slogdet = triple_matmul_with_diagonal_mat(U.T, A_inv_elts, U)

    sign, magnitude = jnp.linalg.slogdet(jnp.diag(W_inv_elts) + second_term_in_slogdet)
    first_term = sign * magnitude

    return first_term + W_logdet + A_logdet


def cg(A_dot_x, b, log_fun=None, tol=1e-5, maxiter=200):
    # DOES NOT AGREE WITH SCIPY -- DO NOT USE FOR ANYTHING SERIOUS!
    # Straightforward implementation of conjugate gradients following Trefethen
    # & Bau, p.294.
    # This takes a function A_dot_x computing the matrix-vector product of A
    # with x, and the vector b in Ax = b.
    # The first element returned is the solution found. The second is a
    # dictionary containing information about the iterations.
    # log_fun is a function taking xnm1, rnm1, pnm1 [current solution, current
    # residual, and current search directions, respectively]. If provided, the
    # info dictionary will contain a list of the function results, recorded at
    # each iteration.

    print(
        "Warning: This function does not give the same result as scipy.linalg.sparse.cg"
        " and is thus likely not working as intended yet. Do not use for serious work!"
    )

    # Initialise
    xnm1 = np.zeros_like(b)
    rnm1 = b.copy()
    pnm1 = rnm1.copy()

    log_results = list()

    for n in range(1, b.shape[0] + 1):

        if log_fun is not None:
            log_results.append(log_fun(xnm1, rnm1, pnm1))

        cur_Apnm1 = A_dot_x(pnm1)

        # Step length
        alpha_n = (rnm1 @ rnm1) / (pnm1 @ cur_Apnm1)

        # Approximate solution
        xn = xnm1 + alpha_n * pnm1

        # Residual
        rn = rnm1 - alpha_n * cur_Apnm1

        # Improvement this step
        betan = (rn @ rn) / (rnm1 @ rnm1)

        # Search direction
        pn = rn + betan * pnm1

        # TODO: This is probably not efficient -- unless it makes
        # passes by reference?
        xnm1 = xn
        rnm1 = rn
        pnm1 = pn

        if np.linalg.norm(rn) < tol or n >= maxiter:
            break

    return xn, {"n_iterations": n, "log_results": log_results, "success": n < maxiter}


def lanczos(A_dot_x, b, m=30, return_eigenvalues=True):

    # FIXME: I'm pretty sure this implementation is suboptimal, in that it does
    # too much work. Fix.
    # FIXME: Does this return Q or Q_T? The quspin library seems to call it Q_T?

    N = b.shape[0]

    alphas = np.zeros(m)
    betas = np.zeros(m + 1)
    qs = np.zeros((m + 2, N))
    qs[1] = b / np.linalg.norm(b)

    for i in range(m):

        cur_n = i + 1
        v = A_dot_x(qs[cur_n])
        alphas[cur_n - 1] = qs[cur_n] @ v
        v = v - betas[cur_n - 1] * qs[cur_n - 1] - alphas[cur_n - 1] * qs[cur_n]
        betas[cur_n] = np.linalg.norm(v)
        qs[cur_n + 1] = v / betas[cur_n]

    res_qs = qs[1:-1]
    res_betas = betas[1:-1]
    res_alphas = alphas

    results = {"Q": res_qs.T, "beta": res_betas, "alpha": res_alphas}

    if return_eigenvalues:
        results["evs"], results["evecs"] = eigh_tridiagonal(
            results["alpha"], results["beta"]
        )

    return results


def solve_inv_sqrt_approx_lanczos(E, V, Q, z):

    m = E.shape[0]

    e1 = (np.arange(m) == 0).astype(float)

    sqrt_term = np.linalg.solve(V.T, (1 / np.sqrt(E)) * np.linalg.solve(V, e1))
    z_norm = np.linalg.norm(z)
    result = z_norm * np.matmul(Q, sqrt_term)

    return result


def get_rot_mat(theta):

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
