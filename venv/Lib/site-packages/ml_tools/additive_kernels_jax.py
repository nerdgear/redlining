import jax.numpy as jnp
from jax.lax import fori_loop
from jax.ops import index_update
from jax import jit
from functools import partial


@jit
def _inner_fun(k, vals):

    cur_sign = (-1) ** (k - 1)
    cur_sk = jnp.sum(vals["zs"] ** k, axis=0)
    cur_n_minus_k = vals["cur_n"] - k

    cur_e = vals["es"][cur_n_minus_k]

    cur_result = cur_e * cur_sk * cur_sign

    vals["cur_total"] = vals["cur_total"] + cur_result

    return vals


@jit
def _outer_fun(n, vals):

    vals["cur_total"] = jnp.zeros_like(vals["es"][0])
    vals["cur_n"] = n

    result = fori_loop(1, n + 1, _inner_fun, vals)

    vals["es"] = index_update(vals["es"], n, result["cur_total"] / n)

    return vals


@jit
def _run_newton_girard_loop(init_vals, N):

    final_result = fori_loop(1, N + 1, _outer_fun, init_vals)["es"][1:]

    return final_result


def newton_girard_combination_for_loop(kerns, N):

    init_es = jnp.zeros((N + 1, *kerns[0].shape))
    init_es = index_update(init_es, 0, 1.0)

    init_vals = {
        "zs": jnp.array(kerns),
        "N": N,
        "cur_n": 0,
        "es": init_es,
        "cur_total": jnp.zeros_like(kerns[0]),
    }

    return _run_newton_girard_loop(init_vals, N)


@partial(jit, static_argnums=2)
def newton_girard_inner_loop_python(es, zs, N):

    for cur_n in range(1, N + 1):

        cur_total = jnp.zeros_like(es[0])

        for k in range(1, cur_n + 1):

            cur_sign = (-1) ** (k - 1)
            cur_sk = jnp.sum(zs ** k, axis=0)

            cur_n_minus_k = cur_n - k
            cur_e = es[cur_n_minus_k]

            cur_total = cur_total + cur_e * cur_sk * cur_sign

        es = index_update(es, cur_n, cur_total / cur_n)

    return es[1:]


def newton_girard_combination(kerns, N):
    """Uses equation (6) in Duvenaud et al.'s additive GP paper to compute N kernels,
       one for each order of interaction."""

    es = jnp.zeros((N + 1, *kerns[0].shape))
    es = index_update(es, 0, 1.0)
    zs = kerns

    return newton_girard_inner_loop_python(es, zs, N)
