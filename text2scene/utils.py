import os

from typing import Dict


LOCAL_PATH = os.getcwd()


def get_info_from_object(obj) -> None :
    position, orientation = obj.get_world_pose()
    linear_velocity = obj.get_linear_velocity()
    print("Object position is : " + str(position))
    print("Object's orientation is : " + str(orientation))
    print("Object's linear velocity is : " + str(linear_velocity))


def remove_prefix(path: str, prefix: str) -> str:
    """
    Remove the *prefix* of the *path*

    Args:
        path (str): Path to folder
        prefix (str): Prefix to remove from the path

    Returns:
        (str): Path without prefix
    """
    return path[len(prefix) :] if path.startswith(prefix) else path


def get_list_assets_local(path: str) -> Dict:
    """
    Get the list all files and sub-folders at *path*

    Args:
        path (str): Path to local folder

    Returns:
        dirs_and_files (typing.Dict): A dictionary containing folders
        and files under path
    """
    dirs_and_files = {}
    if not os.path.isdir(path):
        return dirs_and_files

    os.chdir(path)
    new_pwd = os.getcwd()

    for root, _, files in os.walk(new_pwd + "/"):
        files = [f for f in files if not f.startswith(".")]
        dirs_and_files["." + remove_prefix(root, new_pwd)] = files

    os.chdir(LOCAL_PATH)
    return dirs_and_files
