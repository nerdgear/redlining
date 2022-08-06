import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from typing import Callable, Tuple, Dict
from tqdm import tqdm


def multi_class_eval(y_p_df: pd.DataFrame, y_t_df: pd.DataFrame,
                     metric_fn: Callable[[np.ndarray, np.ndarray], float] =
                     log_loss,
                     metric_name: str = 'metric') -> pd.Series:
    """Computes per-class evaluation metrics for binary outcomes.

    Args:
        y_p_df: Predicted probabilities for each class.
        y_t_df: Binary outcomes for each class.
        metric_fn: The function to use to compute the metric. Must take
            y_t and y_p arguments.
        metric_name: The name to give to the Series returned.

    Returns:
        A Series whose index is the class and whose values are the metric
        evaluated for that class.
    """

    # Make sure all the prediction columns are in the true df
    assert(len(set(y_p_df.columns) & set(y_t_df.columns)) ==
           len(set(y_p_df.columns)))

    assert(y_p_df.shape[0] == y_t_df.shape[0])

    results = dict()

    for cur_class in y_p_df.columns:

        cur_y_p = y_p_df[cur_class].values
        cur_y_t = y_t_df[cur_class].values

        cur_metric = metric_fn(cur_y_t, cur_y_p)
        results[cur_class] = cur_metric

    return pd.Series(results, name=metric_name)


def bootstrap_metric(metric, y_t, y_p, n_samples=1000):
    # TODO: I could just rewrite this using the new "bootstrap_fun" function.

    boot_metrics = np.empty(n_samples)

    for i in range(n_samples):

        # Get a bootstrap sample
        sample = np.random.choice(y_t.shape[0], size=y_t.shape[0],
                                  replace=True)

        boot_y_t = y_t[sample]
        boot_y_p = y_p[sample]

        cur_metric = metric(boot_y_t, boot_y_p)

        boot_metrics[i] = cur_metric

    return boot_metrics


def richness_rmse(y_t, y_p):

    # Each is expected to be N x S, where S is the number of outcomes.
    expected_sum = y_p.sum(axis=1)
    actual_sum = y_t.sum(axis=1)

    return np.sqrt(np.mean((expected_sum - actual_sum)**2))


def bootstrap_multi_class_eval(metric, y_t_df, y_p_df, n_samples=1000):
    # TODO: I could just rewrite this using the new "bootstrap_fun" function.

    results = dict()

    for cur_outcome in y_t_df.columns:

        cur_y_t = y_t_df[cur_outcome].values
        cur_y_p = y_p_df[cur_outcome].values

        try:
            cur_metrics = bootstrap_metric(metric, cur_y_t, cur_y_p,
                                           n_samples=n_samples)
        except ValueError:
            cur_metrics = np.zeros(n_samples) * np.nan

        results[cur_outcome] = cur_metrics

    return pd.DataFrame(results)


def bootstrap_multi_class_eval_and_summarise(metric, y_t_df, y_p_df,
                                             n_samples=1000):

    bootstrap_df = bootstrap_multi_class_eval(
        metric, y_t_df, y_p_df, n_samples)

    # Compute mean across species for each of the bootstrap replications
    means = bootstrap_df.mean(axis=1)

    # Compute their standard deviation to get bootstrap error
    sd_of_mean = means.std()

    # Compute the overall mean
    bootstrap_mean = means.mean()

    return pd.Series({'mean': bootstrap_mean, 'sd': sd_of_mean})


def calculate_mean_difference_to_reference(
        y_p_df: pd.DataFrame, y_t_df: pd.DataFrame, ref_y_p_df: pd.DataFrame,
        metric_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Predicts mean difference in metrics across classes for a model and a
    reference.

    Args:
        y_p_df: Predicted probabilities from model.
        y_t_df: Actual outcomes [binary].
        ref_y_p_df: Reference model predictions.
        metric_fn: The metric to use.

    Returns:
        The mean difference in the metric across outcomes.
    """

    metric_model = multi_class_eval(y_p_df, y_t_df, metric_fn=metric_fn)
    metric_ref = multi_class_eval(ref_y_p_df, y_t_df, metric_fn=metric_fn)

    diffs = metric_model - metric_ref

    mean_diff = np.mean(diffs)

    return mean_diff


def bootstrap_mean_difference_to_reference(y_p_df, y_t_df, ref_y_p_df,
                                           metric_fn, n_bootstrap):
    # TODO: Add documentation.
    # TODO: I could just rewrite this using the new "bootstrap_fun" function.

    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        choice = np.random.choice(ref_y_p_df.shape[0],
                                  size=ref_y_p_df.shape[0])
        cur_y_p, cur_y_t, cur_ref_y_p = map(
            lambda x: x.iloc[choice, :], [y_p_df, y_t_df, ref_y_p_df])
        diffs[i] = calculate_mean_difference_to_reference(
            cur_y_p, cur_y_t, cur_ref_y_p, metric_fn)

    return diffs


def auc_with_nan(y_t, y_p):

    if len(np.unique(y_t)) > 1:
        return roc_auc_score(y_t, y_p)
    else:
        return np.nan


def log_loss_with_labels(y_t, y_p):

    return log_loss(y_t, y_p, labels=[0, 1])


def neg_log_loss_with_labels(y_t, y_p):

    return -log_loss_with_labels(y_t, y_p)


def compute_model_differences(
        model_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
        reference_name: str, n_bootstrap: int = 1000,
        verbose: bool = False) -> pd.DataFrame:

    if reference_name not in model_results:
        raise Exception('Model not present')

    ref_y_p, ref_y_t = model_results[reference_name]

    results = list()

    model_results = {x: y for x, y in model_results.items() if x !=
                     reference_name}

    to_it = tqdm(model_results.items()) if verbose else model_results.items()

    for cur_model, (cur_y_p, _) in to_it:

        for cur_metric_name, cur_metric_fn in metrics.items():

            cur_bootstrap_diffs = bootstrap_mean_difference_to_reference(
                cur_y_p, ref_y_t, ref_y_p, cur_metric_fn, n_bootstrap)

            diff_mean, diff_sd = (np.mean(cur_bootstrap_diffs),
                                  np.std(cur_bootstrap_diffs))

            results.append({
                'model': cur_model,
                'metric': cur_metric_name,
                'diff_mean': diff_mean,
                'diff_sd': diff_sd
            })

    return pd.DataFrame(results)


def bootstrap_fun(fun, *args, is_df=True, n_bootstraps=1000,
                  show_progress=True):
    # *args is assumed to be a sequence of dataframes or np.arrays
    # fun is assumed to take subsetted versions of these *args and return
    # something.
    first_shape = [x.shape[0] for x in args]

    assert len(set(first_shape)) == 1

    results = list()

    if show_progress:
        iterator = tqdm(range(n_bootstraps))
    else:
        iterator = range(n_bootstraps)

    for _ in iterator:

        # Bootstrap choice
        chosen = np.random.choice(first_shape[0], size=first_shape[0],
                                  replace=True)

        if is_df:
            subsetted = [x.iloc[chosen] for x in args]
        else:
            subsetted = [x[chosen] for x in args]

        cur_result = fun(*subsetted)

        results.append(cur_result)

    return results


def sample_mean_distribution_clt(x):
    # Given a vector of observations x, computes the sample mean and its
    # standard deviation using the CLT.

    mean = np.mean(x)
    sd = np.std(x)
    sd_of_mean = sd / np.sqrt(x.shape[0])

    return mean, sd_of_mean
