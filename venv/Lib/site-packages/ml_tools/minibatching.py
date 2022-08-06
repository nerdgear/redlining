import os
import numpy as np
from functools import partial
from tqdm import tqdm
from typing import Dict, Tuple, Callable, Any, IO
from ml_tools.flattening import reconstruct_np


def get_batch_indices(indices: np.ndarray, batch_size: int, cur_start: int) \
        -> Tuple[np.ndarray, int]:
    """
    Moves through a sequence of indices, extracting a batch each time and
    looping back at the end.

    Args:
        indices: The array of indices
        batch_size: The number of elements to pick
        cur_start: The current position in the array of indices

    Example:
        get_batch_indices(np.array([1, 2, 3, 4]), 3, 2) should return
        np.array([3, 4, 1]) and 1 [we're looping back to the start].
    """

    array_length = len(indices)

    cur_end = cur_start + batch_size

    if cur_end < array_length:

        picked = indices[cur_start:cur_end]

        new_start = cur_end

    else:

        # First pick whatever we can
        picked = indices[cur_start:]

        still_to_pick = batch_size - len(picked)

        picked = np.concatenate([picked, indices[:still_to_pick]])

        new_start = still_to_pick

    return picked, new_start


def next_batch(array_dict: Dict[str, np.ndarray],
               shuffled_indices: np.ndarray,
               cur_position: int,
               batch_size: int) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Returns the next batch of arrays.

    Args:
        array_dict: A dictionary of arrays.
        shuffled_indices: The order of indices to traverse.
        cur_position: Where in the indices we currently are.
        batch_size: The size of batches to return.

    Returns:
        A tuple with the current batch of arrays and the next position.
    """

    indices, next_position = get_batch_indices(
        shuffled_indices, batch_size, cur_position)

    subset_arrays = {x: y[indices] for x, y in array_dict.items()}

    return subset_arrays, next_position


def optimise_minibatching(
        data_dict: Dict[str, np.ndarray],
        to_optimise: Callable[[np.ndarray, Any],
                              Tuple[np.ndarray, np.ndarray]],
        opt_step_fun: Callable[[Any, np.ndarray, np.ndarray],
                               Tuple[Any, np.ndarray]],
        opt_state: Any,
        theta: np.ndarray,
        batch_size: int,
        n_steps: int,
        n_data: int,
        callback: Callable[[int, float, np.ndarray, np.ndarray, Any], None]):
    """
    Optimises a function using minibatching.

    Args:
        data_dict: The full dataset in the form of a dictionary of arrays.
        to_optimise: The function to optimise. It takes the current setting
            of parameters, theta, as its first arguments, and the elements
            of the data dictionary as its others. It returns the objective
            and its gradient.
        opt_step_fun: Takes the current state of the optimiser, the current
            parameter setting, and the gradient of the objective. It produces
            an updated state of the optimiser and the new parameter settings.
        theta: The initial parameter setting.
        batch_size: The batch size to use.
        n_steps: How many steps to run the optimisation for
        n_data: How many data points there are in total
        log_file: If given, writes the sequence of losses to the log file.
        append_to_log_file: Appends to log file if true, otherwise creates a
            new one.
        opt_state: The initial state of the optimiser.

    Returns:
        A tuple containing the final setting of parameters theta and the
        list of objective values during optimisation.
    """

    # Pick shuffled indices
    indices = np.random.permutation(n_data)
    cur_position = 0

    loss_log = list()

    for i in tqdm(range(n_steps)):

        # Get the array subset
        cur_arrays, cur_position = next_batch(
            data_dict, indices, cur_position, batch_size)

        cur_opt_fun = partial(to_optimise, **cur_arrays)
        obj, grad = cur_opt_fun(theta)

        # Callback
        callback(i, obj, theta, grad, opt_state)

        theta, opt_state = opt_step_fun(opt_state, theta, grad)
        loss_log.append(obj)

    return theta, loss_log, opt_state


def loss_log_callback(step: int, loss: float, theta: np.ndarray,
                      grad: np.ndarray, opt_state: Any, file_handle: IO):

    file_handle.write(f'{step},{loss}\n')
    file_handle.flush()


def save_opt_state_callback(step: int, loss: float, theta: np.ndarray,
                            grad: np.ndarray, opt_state: Any, target_dir: str,
                            save_every: int):

    os.makedirs(target_dir, exist_ok=True)

    cur_target_file = os.path.join(target_dir, f'opt_state_{step}.npz')
    state_dict = opt_state._asdict()
    np.savez(cur_target_file, **state_dict)


def save_theta_and_grad_callback(step: int, loss: float, theta: np.ndarray,
                                 grad: np.ndarray, opt_state: Any,
                                 target_dir: str, summary: Any,
                                 save_every: int,
                                 additional_vars: Dict[str, np.ndarray] = {}):

    os.makedirs(target_dir, exist_ok=True)

    if step % save_every == 0:

        # Reconstruct
        theta_dict = reconstruct_np(theta, summary)
        grad_dict = reconstruct_np(grad, summary)

        theta_dict['loss'] = loss
        theta_dict['step'] = step
        grad_dict['step'] = step

        # Add any additional variables passed in
        theta_dict.update(additional_vars)

        # Save
        theta_target = os.path.join(target_dir, f'theta_{step}.npz')
        grad_target = os.path.join(target_dir, f'grad_{step}.npz')

        np.savez(theta_target, **theta_dict)
        np.savez(grad_target, **grad_dict)
