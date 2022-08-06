import pymc3 as pm


def log_sigmoid(logit_p):

    return -pm.math.log1pexp(-logit_p)
