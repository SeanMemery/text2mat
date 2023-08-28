import os


def find_project_full_path(project_dir="lang2mat"):
    r"""Find the full path of `project_dir`.

    Parameters
    ----------
    project_dir : str
        Top directory (lang2mat).

    Returns
    -------
    str
        Full path to top directory, equivalent to running pwd inside `project_dir`.

    Examples
    --------
    >>> import os
    >>> full_path = find_project_full_path()
    >>> pwd_path = os.getcwd()
    >>> assert full_path == pwd_path
    """
    full_path = os.getcwd()
    assert (
        project_dir in full_path
    ), f"It needs to be called from within {project_dir} directory."
    return os.path.join(full_path[: full_path.find(project_dir)], project_dir)


# Project path
PROJECT_PATH = find_project_full_path()
