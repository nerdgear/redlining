import numpy as np


def newton_girard_combination(kerns, N):
    """Uses equation (6) in Duvenaud et al.'s additive GP paper to compute N kernels,
       one for each order of interaction."""
    # TODO: Make a JAX version. Try it on BBS.

    es = np.zeros((N + 1, *kerns[0].shape))
    es[0] = 1.0
    zs = kerns

    for cur_n in range(1, N + 1):

        cur_total = np.zeros_like(es[0])

        for k in range(1, cur_n + 1):

            cur_sign = (-1) ** (k - 1)
            cur_sk = np.sum(zs ** k, axis=0)

            cur_n_minus_k = cur_n - k
            cur_e = es[cur_n_minus_k]

            cur_total += cur_e * cur_sk * cur_sign

        es[cur_n] = cur_total / cur_n

    return es[1:]
