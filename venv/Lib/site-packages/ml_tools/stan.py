import os
import pystan
from ml_tools.utils import load_pickle_safely, save_pickle_safely


def load_stan_model_cached(model_path):

    model_last_modified = os.path.getmtime(model_path)

    assert(os.path.isfile(model_path))
    cache_path = os.path.splitext(model_path)[0] + '.pkl'

    if (os.path.isfile(cache_path) and os.path.getmtime(cache_path) >
            model_last_modified):

        return load_pickle_safely(cache_path)

    else:

        model = pystan.StanModel(model_path)
        save_pickle_safely(model, cache_path)
        return model
