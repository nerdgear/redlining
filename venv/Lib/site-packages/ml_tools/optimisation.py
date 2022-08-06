import numpy
from jax import jit, value_and_grad
from scipy.optimize import minimize
from .decorators import count_decorator, print_decorator
from functools import partial
from .jax import hvp


def newton_optimise(
    start_f, fun, jac, hess, solve_fun=numpy.linalg.solve, tolerance=1e-5
):

    f = start_f

    n_iter = 0
    # Write a Newton routine
    difference = 1.0

    while difference > tolerance:

        cur_hess = hess(f)
        cur_jac = jac(f)

        sol = solve_fun(cur_hess, cur_jac)

        new_f = f - sol
        difference = numpy.linalg.norm(f - new_f)

        f = new_f

        n_iter = n_iter + 1

        if n_iter % 10 == 0:
            print(f"On iteration {n_iter}. Difference is {difference}")

    print(f"Converged after {n_iter} iterations.")

    return f


def optimize_with_hvp(
    to_minimize,
    start_params,
    method_name="trust-ncg",
    verbose=False,
    minimize_kwargs={},
):

    val_grad_fun = jit(value_and_grad(to_minimize))

    decorated = count_decorator(partial(print_decorator, verbose=verbose)(val_grad_fun))

    hvp_fun = lambda x, p: hvp(to_minimize, x, p)
    hvp_fun = count_decorator(jit(hvp_fun))

    result = minimize(
        decorated,
        start_params,
        method=method_name,
        hessp=hvp_fun,
        jac=True,
        **minimize_kwargs,
    )

    n_hvp_calls = hvp_fun.calls
    n_val_and_grad_calls = decorated.calls

    return (
        result,
        hvp_fun,
        val_grad_fun,
        {"n_hvp_calls": n_hvp_calls, "n_val_and_grad_calls": n_val_and_grad_calls},
    )
