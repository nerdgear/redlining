import numpy as np


def bootstrap_statistic(statistic, data, n_samples=1000):

    resampled_statistics = np.zeros(n_samples)

    for i in range(n_samples):

        cur_sample = np.random.choice(data.shape[0], size=data.shape[0],
                                      replace=True)
        resampled_statistics[i] = statistic(data[cur_sample])

    return resampled_statistics
