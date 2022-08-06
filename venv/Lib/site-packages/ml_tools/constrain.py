import tensorflow as tf


def apply_transformation(parameter_dict, search_key, transformation,
                         replace_with='_'):
    # Applies the transformation function "transformation" to each element
    # in the dictionary matching "search_key".
    # Returns a new dictionary where the search keys are replaced with
    # "replace_with" containing the result.
    # Example usage:
    # parameter_dict = {'prior_log_var': 0.}
    # apply_transformation(parameter_dict, '_log_', tf.exp)
    # Should return: {'prior_var': 1.}

    keys_to_apply_to = [x for x in parameter_dict.keys() if search_key in x]

    new_dict = {x: y for x, y in parameter_dict.items() if x not in
                keys_to_apply_to}

    for cur_key in keys_to_apply_to:

        cur_value = parameter_dict[cur_key]
        new_value = transformation(cur_value)

        new_key = cur_key.replace(search_key, replace_with)

        new_dict[new_key] = new_value

    return new_dict


def apply_exp_transformation_tf(parameter_dict, search_key='_log_',
                                transformation=tf.exp, replace_with='_'):

    return apply_transformation(parameter_dict, search_key, transformation,
                                replace_with)
