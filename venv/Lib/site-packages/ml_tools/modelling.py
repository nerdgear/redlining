import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def remove_correlated_variables(covariate_df, corr_threshold=0.95):

    # Returns names of columns to keep so that those with an absolute
    # correlation above the threshold are dropped.
    covariate_names = covariate_df.columns

    to_keep = remove_correlated_variables_array(covariate_df.values)

    to_keep_names = np.array(covariate_names)[to_keep]

    return to_keep_names


def remove_correlated_variables_array(design_matrix, corr_threshold=0.95):

    corrs = np.corrcoef(design_matrix.T)
    corrs = np.tril(corrs, -1)

    abs_corr = np.abs(corrs)
    highly_correlated = np.where(abs_corr > corr_threshold)
    to_drop = highly_correlated[0]

    to_keep = np.setdiff1d(np.arange(design_matrix.shape[1]), to_drop)

    return to_keep


def add_quadratic_terms_and_interactions(X):

    features = PolynomialFeatures(degree=2, include_bias=False)

    return features.fit_transform(X)


def add_quadratic_terms(X):

    return np.concatenate([X, X ** 2], axis=1)


def add_cubic_terms(X):

    return np.concatenate([X, X ** 2, X ** 3], axis=1)


def add_cubic_terms_and_simple_interactions(X):

    cubic_additions = X ** 3

    quadratic_terms = add_quadratic_terms_and_interactions(X)

    return np.concatenate([quadratic_terms, cubic_additions], axis=1)
