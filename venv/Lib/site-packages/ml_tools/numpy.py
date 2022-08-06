import numpy as np


def matrix_argsort(mat):
    # Returns indices of the sorted elements in mat, starting with the smallest
    # Return shape is N x 2, where N is the total number of elements in mat

    return np.dstack(np.unravel_index(np.argsort(mat.ravel()), mat.shape))[0]


def interleave_arrays(arr_1, arr_2):
    # Returns array which alternates elements from array 1 and 2, starting
    # with array 1. Assumes they are the same shape.
    assert arr_1.shape[0] == arr_2.shape[0]

    result = np.empty(arr_1.shape[0] * 2)

    result[::2] = arr_1
    result[1::2] = arr_2

    return result


def load_to_dict(npz_file, *args, **kwargs):

    with open(npz_file, "rb") as f:
        loaded = np.load(f, *args, **kwargs)
        return dict(loaded)
