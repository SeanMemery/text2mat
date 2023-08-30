import random
import numpy as np
import glob

from os.path import join, basename
from variables import REMOTE_FOLDER_PREFIX, HOSTNAME, OMNIVERSE_MODELS_PATH, \
    LOCAL_MODELS_PATH, WORLD_PATH, PHYSICS_MATERIAL_PATH
from typing import List, Tuple, Union, Callable, Dict

import omni 

from pxr import UsdLux, Gf, Sdf, UsdGeom
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.materials import PhysicsMaterial, OmniPBR, OmniGlass
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name


OMNIVERSE_FULL_MODELS_PATH = join(REMOTE_FOLDER_PREFIX, HOSTNAME, \
    OMNIVERSE_MODELS_PATH)


def generate_valid_prim_path(prim_path: str):
    return find_unique_string_name(
        initial_name=prim_path,
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )


def get_3d_obj_path(object_dir: str, object_file: str="") -> str:
    """
    It returns the local asset path directory of *object*

    Args:
        object_dir (str): Directory containing the object
        object_file (typing.Optional[str]): If there is many usd files in
            *object_dir* you can specify which one you want to get the path to

    NOTE: Be careful about upper and lower case letters, as the directory name should
    match the asset directory
    """
    object_full_path = f"{join(LOCAL_MODELS_PATH, object_dir)}"
    if not object_file:
        object_full_path = f"{join(object_full_path, '*.usd')}"
    else:
        object_full_path = f"{join(object_full_path, object_file + '.usd')}"

    usd_path = glob.glob(object_full_path)
    assert len(usd_path) == 1, f"There is more than one USD file in \
        {object_dir}, please provide the object_file parameter"

    return join(OMNIVERSE_FULL_MODELS_PATH, object_dir, basename(usd_path[0]))


def get_3d_obj_parameters(objects: List[str]) -> List[Tuple]:
    """
    Get parameters of list of *objects*, so far is hard-coded

    NOTE: The object directories must match the omniverse directories (asset_path)
    """
    obj_and_parameters = []
    for object in objects:
        object_l = object.lower()
        prim_path = f"{join(WORLD_PATH, object_l.capitalize())}"
        if object_l == "person":
            obj_and_parameters.append(
                (XFormPrimView, 
                {
                    "name": object_l, 
                    "asset_path": get_3d_obj_path("male_body"),
                    "prim_paths_expr": prim_path,
                    "orientations": np.array([[0.0, 0.0, 1.0, 1.0]]),
                    "positions": np.array([[0.0, 3.0, 1.0]]),
                    "scales": np.array([[0.05, 0.05, 0.05]]),
                 }, 
                 {}, 
                 {}, False)
            )
        elif object_l == "car":
            obj_and_parameters.append(
                (XFormPrimView, {
                    "name": object, 
                    "asset_path": get_3d_obj_path(object),
                    "prim_paths_expr": prim_path,
                    "orientations": np.array([[0.0, 0.0, 0.5, 0.5]]),
                    "positions": np.array([[0.0, 3.0, 1.0]]),
                    "scales": np.array([[0.1, 0.1, 0.1]]),
                    }, {} , {}, False))
        elif object_l == "slope" or object_l == "hill" or object_l == "incline":
            obj_and_parameters.append(
                (XFormPrimView, {
                    "name": object, 
                    "asset_path": get_3d_obj_path(object, 'slope_20'),
                    "prim_paths_expr": prim_path,
                    "orientations": np.array([[0.0, 0.0, 0.5, 0.5]]),
                    "positions": np.array([[0.0, 7.0, 1.0]]),
                    "scales": np.array([[1.0, 1.0, 1.0]]),
                    }, {} , {}, False))
        elif object_l == "road":
            obj_and_parameters.append(
                (XFormPrimView, {
                    "name": object, 
                    "asset_path": get_3d_obj_path(object),
                    "prim_paths_expr": prim_path,
                    "orientations": np.array([[0.0, 0.0, 0.5, 0.5]]),
                    "positions": np.array([[0.0, 15.0, 1.0]]),
                    "scales": np.array([[1.0, 1.0, 1.0]]),
                    }, {} , {}, False))
        elif object_l == "trampoline":
            obj_and_parameters.append(
                (XFormPrimView, {
                    "name": object, 
                    "asset_path": get_3d_obj_path(object),
                    "prim_paths_expr": prim_path,
                    "orientations": np.array([[0.0, 0.0, 1, 1]]),
                    "positions": np.array([[0.0, 3.0, 1.0]]),
                    "scales": np.array([[0.05, 0.05, 0.05]]),
                    }, {} , {}, False))
        elif object_l == "ball":
            obj_and_parameters.append(
                (DynamicSphere, {
                    "name": object, 
                    "prim_path": prim_path,
                    "position": np.array([0, 1.0, 2]),
                    "scale": np.array([[10.0, 10.0, 10.0]]),
                }, 
                {
                    'prim_path': f"{join(PHYSICS_MATERIAL_PATH, 'physics_material')}",
                    'dynamic_friction': 0.1,
                    'static_friction': 0.1,
                    'restitution': 0.0,
                },
                {
                    # 'type': OmniGlass,
                    # 'params': {
                    #     'prim_path': f"{join(WORLD_PATH, 'OmniGlass')}",
                    #     'ior': 1.5,
                    #     'depth': 0.1,
                    #     'thin_walled': True,
                    #     'color': np.array([random.random(), random.random(), random.random()]),
                    # },
                }, True)
            )
    return obj_and_parameters


def create_camera(prim_path: str, translation: Tuple, rotation: Tuple):
    """
    Set up the camera at *prim_path* and then apply *translations*
    and *rotations*

    Args:
        prim_path (str): Primitive path
        translation (typing.Tuple): Translation
        rotation (typing.Tuple): Rotation XYZ
    """
    stage = omni.usd.get_context().get_stage()

    camera = stage.GetPrimAtPath(prim_path)
    if not camera.IsValid():
        camera = stage.DefinePrim(prim_path, "Camera")
        UsdGeom.Xformable(camera).AddTranslateOp().Set(translation)
        UsdGeom.Xformable(camera).AddRotateXYZOp().Set(rotation)

    return camera.GetPrimPath()


def create_object(app, objects: List[str]) -> None:
    """
    Create *objects* and load them using *app*

    Args:
        app (SimulationApp): Main app
        objects (typing.List[typing.Union[Object, typing.Dict, 
                 typing.Dict, typing.Dict, bool]]): All objects and params 
                 needed for loading primitives, including materials and
                 physics:
                 [
                    object, object_params, physics_params, visual_params, 
                    wheter_object_is_prefabricated_or_not
                 ]
    """
    prims_to_process = get_3d_obj_parameters(objects)
    for obj, o_params, p_params, v_params, default_obj in prims_to_process:
        if v_params and 'type' in v_params:
            if 'params' in v_params:
                visual_material = v_params['type'](**v_params['params'])
            else:
                visual_material = v_params['type']()
            o_params['visual_material'] = visual_material
        if p_params:
            physics_material = PhysicsMaterial(**p_params)
            o_params['physics_material'] = physics_material

        app.load_prim(obj, o_params, default_obj)


def init_transformation_operation(op: UsdGeom.XformOp, values: Tuple) -> UsdGeom.XformOp:
    """
    Initialize the *values* with the correct function, depending on *op*

    Args:
        op (UsdGeom.XformOp): Type of function that needs to be initialized
        values (typing.Tuple): Values used to initialize the correct function
    """
    init_func = Gf.Vec3d if op == UsdGeom.XformOp.TypeTranslate else Gf.Quatd
    return init_func(*values)


def init_property(func, kwargs: Dict, property: str) -> None:
    """
    Initialize an *property* of an Xform object using its own 
    constructor *func* and parameters *kwargs*, then delete
    the value from *kwargs*

    Args:
        func (UsdGeom.XformOp.Create...): Property constructor
        kwargs (Dict): Parameters of the constructor
        property (str): Name of the property
    """
    if property in kwargs:
        func().Set(kwargs.pop(property))


def init_common_light_properties(obj, kwargs: Dict) -> None:
    """
    Initialize common light properties of an *obj* using *kwargs*

    Args:
        obj (UsdGeom.XformOp): Existing object to modify
        kwargs (Dict): Common light parameters used to update the object
    """
    u_kwargs = kwargs.copy()
    ip = init_property
    for attribute in kwargs.keys():
        if attribute == 'color':
            ip(obj.CreateColorAttr, u_kwargs, attribute)
        elif attribute == 'enable_color_temperature':
            ip(obj.CreateEnableColorTemperatureAttr, u_kwargs, attribute)
        elif attribute == 'color_temperature':
            ip(obj.CreateColorTemperatureAttr, u_kwargs, attribute)
        elif attribute == 'intensity':
            ip(obj.CreateIntensityAttr, u_kwargs, attribute)
        elif attribute == 'exposure':
            ip(obj.CreateExposureAttr, u_kwargs, attribute)
        elif attribute == 'normalize_power':
            ip(obj.CreateNormalizeAttr, u_kwargs, attribute)
        elif attribute == 'diffuse_multiplier':
            ip(obj.CreateDiffuseAttr, u_kwargs, attribute)
        elif attribute == 'specular_multiplier':
            ip(obj.CreateSpecularAttr, u_kwargs, attribute)
    kwargs = u_kwargs


def set_light_pose_from_op(prim, func_and_values: Dict) -> None:
    """
    Set up the *prim* (light) pose (translation and orientation) from a list of
    operations and its parameters (*func_and_values*)

    Args:
        prim (UsdGeom.XformOp): Light object
        func_and_values (Dict): Common light parameters used to update the object
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    for func, value in func_and_values.items():
        if func == 't':
            func = UsdGeom.XformOp.TypeTranslate
        else:
            func = UsdGeom.XformOp.TypeOrient
        xform_op = xform.AddXformOp(func, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(init_transformation_operation(func, value))


def set_light_pose_from_transform(prim, transform) -> None:
    """
    Set up the *prim* (light) pose (translation and orientation) from a 
    transformation operation

    Args:
        prim (UsdGeom.XformOp): Light object
        transform (Set): Parameters of the transform matrix
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    transform = Gf.Matrix4d(transform)

    xform_op_t = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, \
        UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_t.Set(transform.ExtractTranslation())

    xform_op_r = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, \
        UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_r.Set(transform.ExtractRotationQuat())


def init_light_pose(prim, pose) -> Callable:
    """
    Choose which light pose function to use depending on the
    type of *pose*

    Args:
        prim (UsdGeom.XformOp): Light object
        pose (typing.Union[Set, List]): Set or list of funcs and values
    """
    if pose:
        if isinstance(pose, Dict):
            return set_light_pose_from_op(prim, pose)
        else:
            return set_light_pose_from_transform(prim, pose)


def create_light(stage, kwargs: Dict) -> None:
    """
    Create light in *stage* using *kwargs*

    Args:
        stage: World stage where all objects reside
        kwargs (typing.Union[Set, List]): Functions and values that
            initializer light, placement and other attributes of our light object
    """
    ip = init_property
    prim_path = generate_valid_prim_path(join(WORLD_PATH, kwargs['type']))

    obj = None
    if kwargs['type'] == 'SphereLight':
        obj = UsdLux.SphereLight.Define(stage, prim_path)
        ip(obj.CreateRadiusAttr, kwargs, "radius")
        ip(obj.CreateTreatAsPointAttr, kwargs, "treat_as_point")
    elif kwargs['type'] == 'DomeLight':
        obj = UsdLux.DomeLight.Define(stage, prim_path)
        ip(obj.CreateTextureFileAttr, kwargs, "texture_file")
        ip(obj.CreateTextureFormatAttr, kwargs, "texture_format")
    elif kwargs['type'] == 'DistantLight':
        obj = UsdLux.DistantLight.Define(stage, prim_path)
        ip(obj.CreateAngleAttr, kwargs, "angle")

    init_common_light_properties(obj, kwargs['common_properties'])
    init_light_pose(obj, kwargs['transforms'])
