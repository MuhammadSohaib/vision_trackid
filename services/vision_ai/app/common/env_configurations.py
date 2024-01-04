import os


def get_env_string(name, default):

    try:

        env_var = os.environ.get(name)

        if not env_var:  # docker env sets to ''
            env_var = default


    except Exception:
        env_var = default

    return env_var


def get_env_int(name, default):

    try:

        env_var = os.environ.get(name)
        if not env_var:  # docker env sets to ''
            env_var = int(default)
        else:
            env_var = int(env_var)

    except Exception:
        env_var = default

    return env_var


def get_env_float(name, default):

    try:

        env_var = os.environ.get(name)
        if not env_var:  # docker env sets to ''
            env_var = float(default)
        else:
            env_var = float(env_var)

    except Exception:
        env_var = default

    return env_var


def get_env_bool(name, default):

    try:

        env_var = os.environ.get(name)
        if not env_var:  # docker env sets to ''
            env_var = default
        elif isinstance(env_var, str) and env_var.lower() == "true":
            env_var = True

        elif isinstance(env_var, str) and env_var.lower() == "false":
            env_var = False

    except Exception:
        env_var = default

    return env_var
