""""
Examples:

1. Synchronize local folder "{$PWD}/lang2mat/data/" and its files
   to a particular omniverse folder "omnifile://localhost/persistent/"
>>> python omniverse_sync.py --sync "../../data/" "persistent/"

2. Save metadata (number of files and number of folders) of url 
   "omnifile://localhost/persistent/"
>>> python omniverse_sync.py --metadata "persistent/"

3. Verify that the cached metadata (number of folders and files) 
   matches those found locally in "{$PWD}/lang2mat/data/"
>>> python omniverse_sync.py --verify "../../data/"

NOTE: The inputs should always end with a slash, so as to be 
identified as directories. 
"""

import carb
import asyncio
import os
import json
import argparse

from pathlib import Path
from os.path import exists, join
from utils import remove_prefix, get_list_assets_local
from typing import Union, Callable, List, Tuple, Dict
from variables import REMOTE_FOLDER_PREFIX, HOSTNAME, CACHE_OMNIVERSE

from omni.client import (
    list_async,
    combine_urls,
    set_hang_detection_time_ms,
    stat,
    create_folder,
    copy,
    Result,
)

OMNIVERSE_PREFIX = join(REMOTE_FOLDER_PREFIX, HOSTNAME)


async def create_gather_async(func: Callable, input: Union[List, str]) -> List:
    """
    Instantiate *func* with *input* asynchronously and gather
    the results obtained

    Args:
        func (func): Function to run asynchronously
        input (str|list): Input to the function

    Returns:
        results (typing.List): List of results parameters
    """
    if type(input) == list:
        tasks = [asyncio.create_task(func(x)) for x in input]
    else:
        tasks = [asyncio.create_task(func(input))]
    results = await asyncio.gather(*tasks)
    return results


async def is_dir_async(path: str) -> bool:
    """
    Check if *path* is a folder

    Args:
        path (str): Path to folder

    Returns:
        (bool): True if path is a folder
    """
    result, folder = await asyncio.wait_for(list_async(path), timeout=100)
    return result == Result.OK and len(folder) > 0


async def list_folder(path: str) -> Tuple[List, List]:
    """
    List files and sub-folders from *path*

    Args:
        path (str): Path to root folder

    Raises:
        Exception: When unable to find files under the path.

    Returns:
        files (typing.List): List of path to each file
        dirs (typing.List): List of path to each sub-folder
    """
    import omni.client

    files, dirs = [], []

    carb.log_info(f"Collecting files for {path}")
    result, entries = await asyncio.wait_for(list_async(path), timeout=100)

    if result != Result.OK:
        # raise Exception(f"Failed to list entries for {path}: {result}")
        return [], []

    for entry in entries:
        set_hang_detection_time_ms(10000)
        full_path = combine_urls(path, entry.relative_path)
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN > 0:
            dirs.append(full_path + "/")
        else:
            carb.log_info(f"Enqueuing {full_path} for processing")
            files.append(full_path)

    return files, dirs


async def recursive_list_folder(path: str) -> List:
    """
    Recursively list all files and sub-folders from *path*

    Args:
        path (str): Path to folder

    Returns:
        dirs_and_files (typing.List): A dictionary containing folders
        and files under path
    """
    dirs_and_files = {}
    is_dir = await create_gather_async(is_dir_async, path)
    if not is_dir:
        return dirs_and_files

    files, dirs = await list_folder(path)
    dirs_and_files[path] = files

    results = await create_gather_async(recursive_list_folder, dirs)
    for result in results:
        dirs_and_files.update(result)

    return dirs_and_files


def get_list_assets_omniverse(path: str) -> Dict:
    """
    Get the list all files and sub-folders at *path*

    Args:
        path (str): Path to omniverse folder

    Returns:
        dirs_and_files (typing.Dict): A dictionary containing folders
        and files under path
    """
    full_path = join(OMNIVERSE_PREFIX, path)
    results = asyncio.get_event_loop().run_until_complete(
        recursive_list_folder(full_path)
    )

    dirs_and_files = {}
    for dir, files in results.items():
        dir_cleaned = remove_prefix(dir, full_path)
        dir_standarized = "./" + dir_cleaned[:-1]
        files_with_no_prefix = remove_prefix_list(files, join(full_path, dir_cleaned))
        dirs_and_files[dir_standarized] = files_with_no_prefix

    return dirs_and_files


def remove_prefix_list(paths: List, prefix: str) -> List:
    """
    Remove the *prefix* of the list of *paths*

    Args:
        paths (typing.List): List of paths
        prefix (str): Prefix to remove from the path

    Returns:
        (typing.List): List of paths without prefix
    """
    return [remove_prefix(path, prefix) for path in paths]


def get_nb_folders_and_nb_files(folders_and_files: Dict) -> Union[int, int]:
    """
    Get the total number of folders and files present
    in *folders_and_files*

    Args:
        folders_and_files (typing.Dict): Dictionary {folder: files}

    Returns:
        folders_counter (int): Number of folders
        files_counter (int): Number of files
    """
    folders_counter = len(folders_and_files)
    files_counter = sum([len(files) for _, files in folders_and_files.items()])

    return folders_counter, files_counter


def diff_of_folders_between_dicts(a: Dict, b: Dict) -> List:
    """
    Difference of folders between two dictionaries i.e. *a* - *b*

    Args:
        a (typing.Dict): Dictionary {folder: files}
        b (typing.Dict): Dictionary {folder: files}

    Returns:
        (typing.List): Folders only present in *a* and
        not *b*
    """
    return a.keys() - b


def diff_of_files_between_lists(a: List, b: List) -> List:
    """
    Difference of files between two list of files i.e. *a* - *b*

    Args:
        a (typing.List): List of files
        b (typing.List): List of files

    Returns:
        (typing.List): Files only present in *a* and
        not *b*
    """
    return [element for element in a if element not in b]


def diff_between_dicts_of_folders_and_files(dict_a: Dict, dict_b: Dict) -> Dict:
    """
    Get the difference between dicts (*dict_a*, *dict_b*) of
    folders and files, i.e. local vs Omniverse, to obtain
    the files and folders that will require uploading

    Args:
        dict_a (typing.Dict): Dictionary {folder: files}
        dict_b (typing.Dict): Dictionary {folder: files}

    Returns:
        (typing.Dict): Folder and files only present in *dict_a*
        and not *dict_b*
    """
    diff_dict = {}
    folder_diff = diff_of_folders_between_dicts(dict_a, dict_b)
    for folder in folder_diff:
        diff_dict[folder] = dict_a[folder]

    folder_intersection = {f for f in dict_a.keys() & dict_b}
    for folder in folder_intersection:
        diff_dict[folder] = diff_of_files_between_lists(dict_a[folder], dict_b[folder])

    # remove empty folders
    diff_dict_clean = {k: v for k, v in diff_dict.items() if v}

    return diff_dict_clean


def add_prefix(path: str, type: str) -> str:
    """
    Remove the *prefix* from *path*, e.g. omniverse or local

    Args:
        path (str): Path to folder
        type (str): Type of prefix

    Returns:
        (str): Path to folder with a prefix included
    """
    if type == "omniverse":
        return join(OMNIVERSE_PREFIX, path)
    else:
        return Path(path).resolve(strict=True)


def upload_files_to_omniverse(
    folders_and_files: Dict, src_path: str, dst_path: str
) -> bool:
    """
    Upload *folders_and_files* from local (*src_path*) to the
    omniverse (*dst_path*)

    Args:
        folders_and_files (typing.Dict): Folders and files to
        upload to omniverse
        src_path (str): Local path
        dst_path (str): Omniverse path

    Returns:
        uploading_performed (bool): True if any uploading was performed
    """
    uploading_performed = False
    for dir, files in folders_and_files.items():
        prefix_dst_path = add_prefix(dst_path, "omniverse")
        prefix_src_path = add_prefix(src_path, "local")
        dst_path_dir = join(prefix_dst_path, dir[2:])
        src_path_dir = join(prefix_src_path, dir[2:])
        (result, _) = stat(dst_path_dir)

        if not result == Result.OK:
            print(f"Creating directory {join(dst_path_dir, dir[2:])}")

            create_folder(dst_path_dir)
            uploading_performed = True
        for file in files:
            print(f"Copying file {join(dst_path_dir, file)}")
            full_dst_path_file = join(dst_path_dir, file)
            full_src_path_file = join(src_path_dir, file)

            copy(full_src_path_file, full_dst_path_file)
            uploading_performed = True
    return uploading_performed


def sync_files_from_local_to_omniverse(src_path: str, dst_path: str) -> bool:
    """
    Synchronize folders and files from local (*src_path*)
    to omniverse path (*dst_path*)

    Args:
        src_path (str): Path to source folder
        dst_path (str): Path to destination folder

    Returns:
        (bool): True if the synchronization was successful
    """
    src_path_folders_and_files = get_list_assets_local(src_path)
    dst_path_folders_and_files = get_list_assets_omniverse(dst_path)

    diff = diff_between_dicts_of_folders_and_files(
        src_path_folders_and_files, dst_path_folders_and_files
    )

    return upload_files_to_omniverse(diff, src_path, dst_path)


def simple_verification_of_assets(path: str) -> bool:
    """
    Verify that the number of folders and files in both local
    and omniverse *path* is similar, we do not care about
    folder structure nor files positioning

    Args:
        path (str): Path to folder for both local and omniverse

    Returns:
        (bool): True if both local and omniverse paths have
        the same sub-folders and files
    """
    print(f'Verifying against "{Path(path).resolve()}"....')
    folders_and_files = get_list_assets_local(path)
    nb_folders, nb_files = get_nb_folders_and_nb_files(folders_and_files)
    nb_folders_omni, nb_files_omni = load_cached_omniverse_metadata()

    is_different = False
    diff_nb_folders = abs(nb_folders_omni - nb_folders)
    diff_nb_files = abs(nb_files_omni - nb_files)
    if diff_nb_folders:
        print(f"There are {diff_nb_folders} folders that are not synced!")
    if diff_nb_files:
        print(f"There are {diff_nb_files} files that are not synced!")

    is_different = diff_nb_folders or diff_nb_files
    if not is_different:
        print("Flawless, all is in sync!")
    return is_different


def load_cached_omniverse_metadata() -> Union[int, int]:
    """
    Load omniverse metadata from *filename* (cached) regarding
    number of folders and files present in omniverse

    Args:
        filename (str): Path to file that contains the metadata

    Returns:
        (typing.Union[int, int]): Number of folders and files
    """
    filename = CACHE_OMNIVERSE
    exists_metadata = exists(filename)
    if not exists_metadata:
        raise ValueError("No folder '%s' was found" % filename)

    with open(filename, "r") as f:
        if not os.stat(filename).st_size:
            return (0, 0)
        metadata = json.load(f)
        return (metadata["folder_counter"], metadata["files_counter"])


def save_omniverse_files_and_folders_metadata(path: str) -> bool:
    """
    Save omniverse metadata under *path* (i.e. number of files
    and number of folders), just for fast verification when
    working with assets in omniverse

    Args:
        path (str): Path to file that contains the metadata

    Returns:
        (bool): True if the file was saved
    """
    folders_and_files = get_list_assets_omniverse(path)
    folder_nb, files_nb = get_nb_folders_and_nb_files(folders_and_files)

    metadata = {"folder_counter": folder_nb, "files_counter": files_nb}

    filename = CACHE_OMNIVERSE
    create_new_file = not exists(filename)

    saved_metadata = False
    with open(filename, "w+") as f:
        if not create_new_file:
            file_content = f.read()
            if len(file_content):
                file_content_json = json.load(file_content)
                for k, v in metadata.items():
                    file_content_json[k] = v
                json.dump(file_content_json, f)
            else:
                json.dump(metadata, f)
            saved_metadata = True
        else:
            json.dump(metadata, f)
            saved_metadata = True
        f.write("\n")
    return saved_metadata


parser = argparse.ArgumentParser(
    prog="Omniverse file sync",
    description="A script for uploading local files into the omniverse",
)
parser.add_argument(
    "--metadata",
    help="Create metadata about existing omniverse files under folder <path>",
)
parser.add_argument(
    "--sync",
    nargs="+",
    help="Synchronize local files <path1> to omniverse <path2>",
)
parser.add_argument(
    "--verify",
    help="Verify that number of local folders and files at <path> is similar to those in omniverse (cache version)",
)
args = parser.parse_args()


from minimal_simulation import App

with App() as app:
    if args.metadata:
        #TODO: There seems to be a bug while counting
        save_omniverse_files_and_folders_metadata(args.metadata)
    elif args.sync:
        sync_files_from_local_to_omniverse(*args.sync)
    elif args.verify:
        #TODO: As it depends on save_omniverse... it is also not entirely working
        simple_verification_of_assets(args.verify)
