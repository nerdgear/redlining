import numpy as np
from functools import wraps


def count_decorator(function):
    # If wrapped around a function, the number of calls of the function can be
    # accessed by calling function.calls on the decorated result.
    @wraps(function)
    def new_fun(*args, **kwargs):
        new_fun.calls += 1
        return function(*args, **kwargs)

    new_fun.calls = 0
    return new_fun


def print_decorator(fun, verbose=True):
    def result(x):

        value, grad = fun(x)

        if verbose:
            print(f"'f': {value}, ||grad(f)||: {np.linalg.norm(grad)}", flush=True)

        return value, grad

    return result
