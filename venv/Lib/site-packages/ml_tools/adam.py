import numpy as np
from typing import NamedTuple, Tuple, Callable


class AdamState(NamedTuple):

    # Estimator of mean
    m: np.ndarray

    # Estimator of second moment
    v: np.ndarray

    # Time step
    t: int


def initialise_state(n_params) -> AdamState:

    return AdamState(m=np.zeros(n_params), v=np.zeros(n_params), t=0)


def adam_step(cur_state: AdamState,
              theta: np.ndarray,
              grad: np.ndarray,
              step_size_fun: Callable[[int], float],
              beta_1: float = 0.9,
              beta_2: float = 0.999,
              eps: float = 1e-8) -> Tuple[np.ndarray, AdamState]:
    """
    Updates the parameter estimates theta using the gradient information in
    grad and the Adam method for stochastic optimisation.

    Args:
        cur_state: Previous estimates of first and second moments.
        theta: Current parameter setting.
        grad: Gradient of the objective with respect to theta.
        step_size_fun: The step size to use as a function of the current time
            step; also known as alpha in the original paper.
        beta_1: Exponential decay rate for the estimates of the first moment
        beta_2: Exponential decay rate for the estimates of the second moment
        eps: Small constant to help numerical stability [?].

    Returns:
        The next setting for theta as well as the updated state.
    """

    # Add 1 here so that we start at 1 rather than zero.
    step_size = step_size_fun(cur_state.t + 1)

    m_t = beta_1 * cur_state.m + (1 - beta_1) * grad
    v_t = beta_2 * cur_state.v + (1 - beta_2) * grad**2

    # TODO: Note that the Adam paper writes that these two lines can be turned
    # into 2. However, I think this way is easier to understand and I don't
    # think efficiency will be very different. But can reconsider in the
    # future.
    m_hat_t = m_t / (1 - beta_1**(cur_state.t + 1))
    v_hat_t = v_t / (1 - beta_2**(cur_state.t + 1))
    theta_t = theta - step_size * m_hat_t / (np.sqrt(v_hat_t) + eps)

    # Update the state tuple
    new_state = AdamState(m=m_t, v=v_t, t=cur_state.t+1)

    return theta_t, new_state
