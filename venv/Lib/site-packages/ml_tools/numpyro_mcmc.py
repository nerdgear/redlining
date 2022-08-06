from numpyro.infer import MCMC, NUTS
from jax import jit
import jax.numpy as jnp
import jax
import numpy as np
from jax import vmap, jit
import arviz as az

# TODO: Would be better not to have a dependency on jax_advi here!
from jax_advi.utils.flattening import flatten_and_summarise, reconstruct
from jax_advi.constraints import apply_constraints
from jax_advi.advi import _calculate_log_posterior
from functools import partial


def initialise_from_shapes(param_shape_dict, sd=0.1, n_chains=4):

    # Make placeholder
    init_theta = {x: np.empty(y) for x, y in param_shape_dict.items()}

    flat_placeholder, summary = flatten_and_summarise(**init_theta)

    # Make flat draws
    flat_init = np.random.randn(n_chains, flat_placeholder.shape[0])

    return flat_init, summary


def sample_numpyro_nuts(
    log_posterior_fun,
    flat_init_params,
    parameter_summary,
    constrain_fun_dict={},
    target_accept=0.8,
    draws=1000,
    tune=1000,
    chains=4,
    progress_bar=True,
    random_seed=10,
    chain_method="parallel",
    thinning=1,
):
    # Strongly inspired by:
    # https://github.com/pymc-devs/pymc3/blob/master/pymc3/sampling_jax.py#L116
    def _sample(current_state, seed):

        step_size = jnp.ones_like(flat_init_params)

        nuts_kernel = NUTS(
            potential_fn=lambda x: -log_posterior_fun(x),
            target_accept_prob=target_accept,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        numpyro = MCMC(
            nuts_kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            postprocess_fn=None,
            progress_bar=progress_bar,
            chain_method=chain_method,
            thinning=thinning,
        )

        numpyro.run(seed, init_params=current_state)
        samples = numpyro.get_samples(group_by_chain=True)
        return samples

    seed = jax.random.PRNGKey(random_seed)
    samples = _sample(flat_init_params, seed)

    # Reshape this into a dict
    def reshape_single_chain(theta):
        fun_to_map = lambda x: apply_constraints(
            reconstruct(x, parameter_summary, jnp.reshape), constrain_fun_dict
        )[0]
        return vmap(fun_to_map)(theta)

    samples = vmap(reshape_single_chain)(samples)

    return az.from_dict(posterior=samples)


def sample_nuts(
    parameter_shape_dict,
    log_prior_fun,
    log_lik_fun,
    constrain_fun_dict,
    target_accept=0.8,
    draws=1000,
    tune=1000,
    chains=4,
    progress_bar=True,
    random_seed=10,
    chain_method="vectorized",
    thinning=1,
    use_tfp=False,
):

    flat_theta, summary = initialise_from_shapes(parameter_shape_dict, n_chains=chains)

    log_post_fun = jit(
        partial(
            _calculate_log_posterior,
            log_lik_fun=log_lik_fun,
            log_prior_fun=log_prior_fun,
            constrain_fun_dict=constrain_fun_dict,
            summary=summary,
        )
    )

    if use_tfp:

        # TODO: Currently ignores thinning
        sampling_result = _sample_tfp_nuts(
            log_post_fun,
            flat_theta,
            summary,
            constrain_fun_dict,
            chains=chains,
            draws=draws,
            tune=tune,
        )

    else:

        sampling_result = sample_numpyro_nuts(
            log_post_fun,
            flat_theta,
            summary,
            constrain_fun_dict,
            chains=chains,
            draws=draws,
            tune=tune,
            chain_method=chain_method,
            thinning=thinning,
        )

    return sampling_result


def _sample_tfp_nuts(
    log_posterior_fun,
    flat_init_params,
    parameter_summary,
    constrain_fun_dict={},
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    num_tuning_epoch=2,
    num_compute_step_size=500,
):
    import jax

    from tensorflow_probability.substrates import jax as tfp

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    @jax.pmap
    def _sample(init_state, seed):
        def gen_kernel(step_size):
            hmc = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=log_posterior_fun, step_size=step_size
            )
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                hmc, tune // num_tuning_epoch, target_accept_prob=target_accept
            )

        def trace_fn(_, pkr):
            return pkr.new_step_size

        def get_tuned_stepsize(samples, step_size):
            return step_size[-1] * jax.numpy.std(samples[-num_compute_step_size:])

        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        for i in range(num_tuning_epoch - 1):
            tuning_hmc = gen_kernel(step_size)
            init_samples, tuning_result, kernel_results = tfp.mcmc.sample_chain(
                num_results=tune // num_tuning_epoch,
                current_state=init_state,
                kernel=tuning_hmc,
                trace_fn=trace_fn,
                return_final_kernel_results=True,
                seed=seed,
            )

            step_size = jax.tree_multimap(
                get_tuned_stepsize, list(init_samples), tuning_result
            )
            init_state = [x[-1] for x in init_samples]

        # Run inference
        sample_kernel = gen_kernel(step_size)
        mcmc_samples, _ = tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // num_tuning_epoch,
            current_state=init_state,
            kernel=sample_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.leapfrogs_taken,
            seed=seed,
        )

        return mcmc_samples

    map_seed = jax.random.split(seed, chains)

    mcmc_samples = _sample(flat_init_params, map_seed)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    # Reshape this into a dict
    def reshape_single_chain(theta):
        fun_to_map = lambda x: apply_constraints(
            reconstruct(x, parameter_summary, jnp.reshape), constrain_fun_dict
        )[0]
        return vmap(fun_to_map)(theta)

    samples = vmap(reshape_single_chain)(samples)

    az_trace = az.from_dict(posterior=posterior)

    return az_trace
