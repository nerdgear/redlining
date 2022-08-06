import jax.numpy as jnp
import numpy as np
from typing import Callable, Dict, Tuple, Any
from .flattening import flatten_and_summarise, reconstruct
from scipy.optimize import minimize
from jax import jit, value_and_grad
from .jax import convert_decorator, hvp, print_decorator
from functools import partial
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

# Function for easy maximum likelihood / a posteriori estimation.
def find_map_estimate(
    init_theta: Dict[str, np.ndarray],
    likelihood_fun: Callable[[Dict[str, np.ndarray]], float],
    transform_function: Callable[
        [Dict[str, np.ndarray]], Dict[str, np.ndarray]
    ] = lambda x: x,
    prior_fun: Callable[[Dict[str, np.ndarray]], float] = lambda _: 0.0,
    verbose: bool = False,
    callback_fn: Callable[
        [Dict[str, np.ndarray], float, float], None
    ] = lambda theta, likelihood, prior: None,
    opt_method="trust-krylov",
    gtol=1e-3,
) -> Tuple[Dict[str, np.ndarray], Any]:
    """
    Computes the map estimate using JAX + scipy.optimize.minimize [L-BFGS-B].

    Args:
        init_theta: The initial values for the optimization as a dictionary of numpy
            arrays [JAX or base numpy, either is fine].
        likelihood_fun: A function that takes in the current value of the
            parameters and returns their likelihood. Must be written using JAX.
        transform_function: A function that takes in the current parameters and
            transforms them. This is used e.g. to ensure certain parameters
            remain positive. Recommended for use with
            ml_tools.constrain.apply_transformation. Defaults to no transform.
        prior_fun: A function taking in the parameter settings after
            transformation and returning the value of their prior. Defaults to
            returning zero [i.e. maximum likelihood].
        verbose: If True, prints the current value of the [negative] objective
            as well as the current norm of the gradient used for optimising.
        callback_fn: An optional function to call at every step of the
            optimisation, taking in the current parameters as well as the
            current likelihood and prior and not returning anything.

    Returns:
    A Tuple whose first element is the [transformed] optimal set of values, and
    whose second element is the result from scipy.optimize.minimize which can be
    used to check the optimization worked OK.
    """

    assert opt_method in ["trust-ncg", "trust-krylov", "L-BFGS-B"]

    flat_theta, summary = flatten_and_summarise(**init_theta)

    def to_minimize(flat_theta):

        theta = reconstruct(flat_theta, summary, jnp.reshape)
        theta = transform_function(theta)
        log_lik = likelihood_fun(theta)
        prior = prior_fun(theta)

        callback_fn(theta, log_lik, prior)

        return -log_lik - prior

    with_grad = jit(value_and_grad(to_minimize))

    if opt_method == "L-BFGS-B":
        with_grad = partial(convert_decorator, verbose=verbose)(with_grad)
        opt_result = minimize(
            with_grad, flat_theta, method=opt_method, jac=True, options=dict(gtol=gtol)
        )
    elif opt_method in ["trust-ncg", "trust-krylov"]:
        with_grad = partial(print_decorator, verbose=verbose)(with_grad)
        hvp_fun = jit(lambda x, y: hvp(to_minimize, x, y))
        opt_result = minimize(
            with_grad,
            flat_theta,
            method=opt_method,
            jac=True,
            hessp=hvp_fun,
            options=dict(gtol=gtol),
        )
        # print("Computing min ev...")
        # matvec_fun = jit(lambda x: hvp_fun(opt_result.x, x))
        # N = flat_theta.shape[0]
        # op = LinearOperator((N, N), matvec=matvec_fun, rmatvec=matvec_fun)
        # eig_res = eigsh(op, which="SA", k=1, maxiter=10000)
        # print(eig_res[0])

    final_theta = reconstruct(opt_result.x, summary, jnp.reshape)
    final_theta = transform_function(final_theta)

    return final_theta, opt_result
