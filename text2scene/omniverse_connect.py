""""
Examples:

1. Login to omniverse
>>> python omniverse_connect.py -c

When the default browser pops up the login, fill out the blanks:
user: "admin"
password: "admin"

2. Sign out of omniverse
>>> python omniverse_connect.py -s
"""

import argparse

from os.path import join
from variables import REMOTE_FOLDER_PREFIX, HOSTNAME

from omni.client import (
    sign_out, 
    get_server_info, 
    Result
)


OMNIVERSE_PREFIX = join(REMOTE_FOLDER_PREFIX, HOSTNAME)


def sign_out_omniverse() -> None:
    """
    Log out the current omniverse session

    Returns:
        (None): The current session will end
    """
    sign_out(OMNIVERSE_PREFIX)
    print("Log out of omniverse")


def connect_omniverse() -> bool:
    """
    Log into omniverse (via browser)

    Returns:
        (bool): True if it created a client session that
        will allow to use the omniverse API
    """
    result, _ = get_server_info(OMNIVERSE_PREFIX)
    if result != Result.OK:
        print("The credentials are not valid! (default: admin/admin)")
        return False
    else:
        print("Connection established!")
        return True


parser = argparse.ArgumentParser(
    prog="Omniverse connect and sign out",
    description="A script for connecting and signing out of omniverse",
)
parser.add_argument(
    "-c",
    "--connect",
    action="store_true",
    help="Connect to omniverse (needed for uploading and modifying assets)",
)
parser.add_argument(
    "-s",
    "--sign-out", action="store_true", help="Sign out from omniverse"
)
args = parser.parse_args()

if args.connect:
    connect_omniverse()
elif args.sign_out:
    sign_out_omniverse()
