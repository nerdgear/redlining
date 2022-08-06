import scipy
import numpy as np
from scipy.stats import beta as beta_dist


class BetaBinomial:

    def __init__(self, alpha, beta):

        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def alpha_beta_from_mean_and_sample_size(mean, sample_size):

        return mean * sample_size, (1 - mean) * sample_size

    @classmethod
    def from_mean_and_sample_size(cls, mean, sample_size):

        alpha, beta = cls.alpha_beta_from_mean_and_sample_size(
            mean, sample_size)

        return cls(alpha, beta)

    def update(self, n_successes, n_failures):

        self.alpha += n_successes
        self.beta += n_failures

    def update_single(self, success):

        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def predict_prob_success(self):

        return (scipy.special.beta(1 + self.alpha, self.beta) /
                scipy.special.beta(self.alpha, self.beta))

    def get_arrays_for_plotting(self):

        to_plot = np.linspace(0, 1, 100)
        y_plot = beta_dist.pdf(to_plot, self.alpha, self.beta)

        return to_plot, y_plot

    def sample(self, n_samples):

        return beta_dist.rvs(self.alpha, self.beta, size=n_samples)
