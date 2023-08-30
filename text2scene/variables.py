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

# Launcher parameters
WIDTH_LAUNCHER_APP = "/app/renderer/resolution/width"
HEIGHT_LAUNCHER_APP = "/app/renderer/resolution/height"

# Replicator
WRITER_FOLDER = f"{PROJECT_PATH}/data/processed/writer/"
WRITER_VIDEO_FOLDER = f"{PROJECT_PATH}/data/processed/writer_video/"

# Connect
REMOTE_FOLDER_PREFIX = "omniverse://"
HOSTNAME = "localhost"
CACHE_OMNIVERSE = ".cache_omniverse"

# Materials and models
LOCAL_MODELS_PATH = f"{PROJECT_PATH}/data/processed/3dmodels"
LOCAL_MATERIALS_PATH = f"{PROJECT_PATH}/data/raw/materials"
LOCAL_OMNIPBR_PATH = f"{PROJECT_PATH}/data/OmniPBR.mdl"
LOCAL_OMNIPBR_CLEARCOAT_PATH = f"{PROJECT_PATH}/data/OmniPBR_ClearCoat.mdl"
LOCAL_OMNI_SURFACE_PATH = f"{PROJECT_PATH}/data/OmniSurface.mdl"
LOCAL_OMNI_PREVIEW_PATH = f"{PROJECT_PATH}/data/UsdPreviewSurface.mdl"

OMNIVERSE_MODELS_PATH = "persistent/processed/3dmodels"
OMNIVERSE_MATERIALS_PATH = "persistent/raw/materials"

# Minimal launcher config
PATH_APP_CONFIG = "/home/ocedron/Research/lang2mat/config/omni.isaac.sim.python.kit"

# Stage (SimulationApp defaults)
DEFAULT_GROUND_PLANE = "/World/defaultGroundPlane"
DEFAULT_GROUND_PLANE_VISUAL_MATERIAL = "/World/defaultGroundPlane/Looks/theGrid"
DEFAULT_GROUND_PLANE_LIGHT = "/World/defaultGroundPlane/SphereLight"
DEFAULT_GROUND_PLANE_GEOMETRY = "/World/defaultGroundPlane/Environment/Geometry"
DEFAULT_GROUND_PLANE_COLLISION_PLANE = (
    "/World/defaultGroundPlane/GroundPlane/CollissionPlane"
)

PHYSICS_MATERIAL_PATH = "/World/Physics_Materials"
WORLD_PATH = "/World"
MATERIALS_PATH = "/Looks"
