import pickle
from patsy import dmatrix


def create_quadratic_effect_formula(cov_names):

    quad_terms = "+".join([f"I({x}**2)" for x in cov_names])
    return quad_terms


def create_main_effect_formula(cov_names):

    main_effects = "+".join(cov_names)
    return main_effects


def create_interaction_formula(cov_names):

    main_effects = create_main_effect_formula(cov_names)
    inter_terms = "(" + main_effects + ")"
    inter_terms = inter_terms + ":" + inter_terms

    return inter_terms


def create_formula(
    cov_names,
    main_effects=True,
    quadratic_effects=True,
    interactions=False,
    intercept=True,
):

    model_str = ""

    if main_effects:
        model_str = model_str + create_main_effect_formula(cov_names)

    if quadratic_effects:
        addition = "+" if len(model_str) > 0 else ""
        model_str = model_str + addition + create_quadratic_effect_formula(cov_names)

    if interactions:
        addition = "+" if len(model_str) > 0 else ""
        model_str = model_str + addition + create_interaction_formula(cov_names)

    if not intercept:
        model_str = model_str + "- 1"

    return model_str


def remove_intercept_column(array, design_info):
    # This function is to sidestep the issue in patsy where it is essentially
    # impossible to remove the intercept term:
    # https://github.com/pydata/patsy/issues/80
    # The workaround is to build the design matrix with the intercept column and
    # then remove it with this function.
    column_name_lookup = {x: i for i, x in enumerate(design_info.column_names)}
    intercept_column = column_name_lookup["Intercept"]
    remaining_indices = [i for i in range(array.shape[1]) if i != intercept_column]

    return array[:, remaining_indices]


def save_design_info(X, formula, design_info, target_file):

    # Try to shrink the necessary data to save.
    # We need to keep the factor levels, but beyond that it's OK.
    factor_names = [
        x.name() for x, y in design_info.factor_infos.items() if y.type == "categorical"
    ]

    if len(factor_names) > 0:
        shrunk_X = X.drop_duplicates(subset=factor_names)
    else:
        # Just keep a few rows
        shrunk_X = X.iloc[:10]

    rel_X = shrunk_X

    with open(target_file, "wb") as f:
        pickle.dump({"formula": formula, "data": rel_X}, f)


def restore_design_info(pickled_info):

    with open(pickled_info, "rb") as f:
        loaded = pickle.load(f)

    design_mat = dmatrix(loaded["formula"], loaded["data"])

    return design_mat.design_info
