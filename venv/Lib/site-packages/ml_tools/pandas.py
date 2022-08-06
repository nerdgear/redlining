import pandas as pd


def compute_mean_and_sd(df):

    mean = df.mean(axis=0)
    sd = df.std(axis=0)

    return pd.concat([mean, sd], axis=1, keys=['mean', 'sd'])
