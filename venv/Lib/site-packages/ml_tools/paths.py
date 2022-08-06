import os


def base_name_from_path(path):

    return os.path.split(os.path.splitext(path)[0])[1]


def get_cur_script_path(file_handle):
    # Use by passing the __file__ variable
    # Returns the path to the current script, including the script name itself.
    exec_dir = os.path.abspath(file_handle)
    return exec_dir
