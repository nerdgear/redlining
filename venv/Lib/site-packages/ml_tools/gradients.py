import numpy as np
from scipy.optimize import approx_fprime


def approx_fprime_default(xk, f, eps=np.sqrt(np.finfo(float).eps)):

    return approx_fprime(xk, f, epsilon=eps)
