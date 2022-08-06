import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def plot_with_error_bars(
    x, lower, mean, upper, ax=None, fill_alpha=0.5, fill_kwargs={}, **plot_kwargs
):

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Plot the median
    ax.plot(x, mean, **plot_kwargs)
    ax.fill_between(x, lower, upper, alpha=fill_alpha, **fill_kwargs)

    return ax


def plot_with_error_bars_sd(
    x, mean, sd, ax=None, fill_alpha=0.5, fill_kwargs={}, **kwargs
):

    return plot_with_error_bars(
        x,
        mean - 2 * sd,
        mean,
        mean + 2 * sd,
        ax=ax,
        fill_alpha=fill_alpha,
        fill_kwargs=fill_kwargs,
        **kwargs
    )


def add_legend_on_right(ax):

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    return ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def conditional_plot_2d(
    pred_fun,
    dims_to_vary,
    n_covs,
    n_points=100,
    other_feat_vals=0.0,
    lower_lim=-2,
    upper_lim=2,
    scaler=None,
):

    # Returns arrays for plotting contour plots / surface plots.
    # Predicts by varying x and y between lower_lim and upper_lim while
    # keeping all other covariates fixed at other_feat_vals.

    x = np.linspace(lower_lim, upper_lim, n_points)
    y = np.linspace(lower_lim, upper_lim, n_points)

    grid = np.meshgrid(x, y)
    to_predict = np.stack([grid[0].reshape(-1), grid[1].reshape(-1)], axis=1)
    base_vals = np.tile(other_feat_vals, (n_points * n_points, n_covs))
    base_vals[:, dims_to_vary] = to_predict
    predicted = pred_fun(base_vals)
    reshaped = np.reshape(predicted, (n_points, n_points))

    if scaler is not None:
        transformed_vals = scaler.inverse_transform(base_vals)
        relevant = transformed_vals[:, dims_to_vary]
        grid[0] = relevant[:, 0].reshape(n_points, n_points)
        grid[1] = relevant[:, 1].reshape(n_points, n_points)

    return grid[0], grid[1], reshaped


def conditional_plot_1d_all(
    pred_fun, n_covs, n_points=100, other_feat_vals=0.0, lower_lim=-2, upper_lim=2
):

    # Returns arrays for plotting contour plots / surface plots.
    # Predicts by varying x and y between lower_lim and upper_lim while
    # keeping all other covariates fixed at other_feat_vals.
    x = np.linspace(lower_lim, upper_lim, n_points)

    predictions = np.zeros((n_points, n_covs))

    for cur_cov in range(n_covs):

        base_vals = np.tile(other_feat_vals, (n_points, n_covs))
        base_vals[:, cur_cov] = x.copy()

        predicted = pred_fun(base_vals)

        predictions[:, cur_cov] = predicted

    return x, predictions


def plot_annotated_scatter(x, y, text, ax, text_alpha=0.5, **point_kwargs):

    ax.scatter(x, y, **point_kwargs)
    texts = [
        ax.text(
            x[i], y[i], "{}".format(text[i]), ha="center", va="center", alpha=text_alpha
        )
        for i in range(len(x))
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="red"))

    return ax


def add_equality_line(ax):

    return ax.plot([0, 1], [0, 1], transform=ax.transAxes)
