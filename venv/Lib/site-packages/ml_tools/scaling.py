def standardise(df, col_names=None):

    if col_names is None:
        col_names = df.columns

    vals_to_scale = df[col_names]

    # Standardise
    means = vals_to_scale.mean()
    sds = vals_to_scale.std()
    scaled_vals = (vals_to_scale - means) / sds

    standardised_df = df.copy()
    standardised_df[col_names] = scaled_vals

    scaling_dict = {
        'means': means,
        'sds': sds
    }

    return standardised_df, scaling_dict


def apply_standardisation(df, scaling_dict):

    means = scaling_dict['means']
    sds = scaling_dict['sds']
    scaled_df = df.copy()

    names_to_scale = means.index
    scaled_df[names_to_scale] = (scaled_df[names_to_scale] - means) / sds

    return scaled_df


def invert_standardisation(scaled_df, scaling_dict):

    means = scaling_dict['means']
    sds = scaling_dict['sds']
    unscaled_df = scaled_df.copy()

    names_to_scale = means.index
    unscaled_df[names_to_scale] = (unscaled_df[names_to_scale] * sds) + means

    return unscaled_df
