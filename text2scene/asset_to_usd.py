""""
Examples:

1. Convert all the 3d models found in "{$PWD}/lang2mat/data/raw/3dmodels/" and
   save all the converted 3d models (USD) in "{$PWD}/lang2mat/data/processed/3dmodels/"
   and force to create the files if they exist already
>>> python asset_to_usd.py --input-path "../../data/raw/3dmodels/" --output-path ."./../data/processed/3dmodels/" --force
"""

import carb
import os
import asyncio
import argparse

from distutils.util import strtobool


def progress_callback(current_step: int, total: int) -> None:
    """
    Progress callback of this task

    Args:
        current_step (int): Current progress
        total (int): Total steps
    """
    print(f"{current_step} of {total}")


async def convert_asset_to_usd(
    input_obj: str,
    output_usd: str,
    ignore_material: bool = True,
    ignore_animation: bool = False,
    single_mesh: bool = True,
    smooth_normals: bool = False,
    ignore_cameras: bool = False,
    preview_surface: bool = False,
    use_meter_as_world_unit: bool = False,
    create_world_as_default_root_prim: bool = True,
    ignore_lights: bool = False,
    embed_textures: bool = False,
    convert_fbx_to_y_up: bool = False,
    convert_fbx_to_z_up: bool = False,
    keep_all_materials: bool = False,
    merge_all_meshes: bool = False,
    use_double_precision_to_usd_transform_op: bool = False,
    ignore_pivots: bool = False,
    disable_instancing: bool = False,
    export_hidden_props: bool = False,
    baking_scales: bool = False,
) -> None:
    """
    Convert asset (OBJ/FBX/glTF) to USD

    NOTE: It uses FBX SDK for FBX convert and Assimp as fallback backend,
    so it should support all assets that Assimp supports. But only obj/glTF
    are fully verified

    Args:
        input_obj (str): Path to the object to convert
        output_usd (str): Path to the USD output
        ignore_animation (bool): Whether to export animation
        ignore_material (bool): Whether to export materials
        single_mesh (bool): True, it will export separate USD files for instancing assets
        smooth_normals (bool): Generate smooth normals for every mesh
        ignore_cameras (bool): Whether to export camera
        preview_surface (bool): Whether to export preview surface of USD
        use_meter_as_world_unit (bool): Uses meter as world unit
        create_world_as_default_root_prim (bool): Whether to create /World as default root prim
        embed_textures (bool): Whether to embed textures for export
        material_loader (Callable[OmniConveterFuture,
                         OmniConverterMaterialDescription): Material  loader for this task
        convert_fbx_to_y_up (bool): Whether to convert imported fbx stage to Maya Y-Up
        convert_fbx_to_z_up (bool): Whether to convert imported fbx stage to Maya Z-Up
        keep_all_materials (bool): Whether to keep all materials including those ones that
                                   are not referenced by any meshes
        merge_all_meshes (bool): Whether to merge all meshes as a single one
        use_double_precision_to_usd_transform_op (bool): Whether to use double precision for
                                                         all USD transform operations
        ignore_pivots (bool): Don't import pivots from assets
        disable_instancing (bool): Disables scene instancing for USD export.
        export_hidden_props (bool): Export props that are hidden or not
        baking_scales (bool): Baking scales into mesh for fbx import
    """
    import omni.kit.asset_converter as converter

    context = converter.AssetConverterContext()
    context.ignore_animation = ignore_animation
    context.ignore_material = ignore_material
    context.single_mesh = single_mesh
    context.smooth_normals = smooth_normals
    context.ignore_cameras = ignore_cameras
    context.preview_surface = preview_surface
    context.use_meter_as_world_unit = use_meter_as_world_unit
    context.create_world_as_default_root_prim = create_world_as_default_root_prim
    context.ignore_lights = ignore_lights
    context.embed_textures = embed_textures
    context.convert_fbx_to_y_up = convert_fbx_to_y_up
    context.convert_fbx_to_z_up = convert_fbx_to_z_up
    context.keep_all_materials = keep_all_materials
    context.merge_all_meshes = merge_all_meshes
    context.use_double_precision_to_usd_transform_op = (
        use_double_precision_to_usd_transform_op
    )
    context.ignore_pivots = ignore_pivots
    context.disable_instancing = disable_instancing
    context.export_hidden_props = export_hidden_props
    context.baking_scales = baking_scales

    instance = converter.get_instance()
    task = instance.create_converter_task(
        input_obj, output_usd, progress_callback, context
    )
    success = await task.wait_until_finished()
    if not success:
        carb.log_error(task.get_status(), task.get_detailed_error())
    else:
        print(f'Conversion from "{input_obj}" to "{output_usd}" done')
    return success


def asset_convert(input_path: str, output_path: str, force: bool, **kwargs) -> None:
    """
    Walk through the models existing in *input_path* and output
    their converted models at *output_usd*, using the *kwargs*
    settings

    Args:
        input_path (str): Path to the folders containing the models
        output_path (str): Path to the USD models
        force (bool): True if the files exist then it will create it again
        kwargs (typing.Dict): All the flags used for converting to USD
    """
    from utils import get_list_assets_local

    supported_file_formats = ["stl", "obj", "fbx"]
    input_dir = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_path)
    os.makedirs(output_dir, exist_ok=True)

    folders_and_files = get_list_assets_local(input_path)

    for dir, files in folders_and_files.items():
        for file in files:
            model_name = os.path.splitext(file)[0]
            model_format = (os.path.splitext(file)[1])[1:]
            if model_format in supported_file_formats:
                model_path = os.path.join(input_dir, dir[2:], file)
                converted_path = os.path.join(output_dir, dir[2:], model_name + ".usd")
                if not os.path.exists(converted_path) or force:
                    status = asyncio.get_event_loop().run_until_complete(
                        convert_asset_to_usd(model_path, converted_path, **kwargs)
                    )
                    assert status, f"ERROR status is {status}"
                else:
                    print(f'The file "{model_path}" already exists!')


default_type = {"default": None, "type": lambda x: bool(strtobool(x))}

parser = argparse.ArgumentParser(
    prog="Convert asset to USD",
    description="A script that walks through a folder and converts found models into USD",
)
parser.add_argument("--input-path")
parser.add_argument("--output-path")
parser.add_argument("--force", action="store_true")
parser.add_argument("--ignore-material", **default_type)
parser.add_argument("--ignore-animation", **default_type)
parser.add_argument("--single-mesh", **default_type)
parser.add_argument("--smooth-normals", **default_type)
parser.add_argument("--ignore-cameras", **default_type)
parser.add_argument("--preview-surface", **default_type)
parser.add_argument("--use-meter-as-world-unit", **default_type)
parser.add_argument("--create-world-as-default-root-prim", **default_type)
parser.add_argument("--ignore-lights", **default_type)
parser.add_argument("--embed-textures", **default_type)
parser.add_argument("--convert-fbx-to-y-up", **default_type)
parser.add_argument("--convert-fbx-to-z-up", **default_type)
parser.add_argument("--keep-all-materials", **default_type)
parser.add_argument("--merge-all-meshes", **default_type)
parser.add_argument("--use-double-precision-to-usd-transform-op", **default_type)
parser.add_argument("--ignore-pivots", **default_type)
parser.add_argument("--disable-instancing", **default_type)
parser.add_argument("--export-hidden-props", **default_type)
parser.add_argument("--baking-scales", **default_type)
args = parser.parse_args()

args_dict = vars(args).copy()
args_dict = {k: v for k, v in args_dict.items() if v is not None}
for key in ["input_path", "output_path", "force"]:
    args_dict.pop(key)


from minimal_simulation import App

with App() as app:
    asset_convert(args.input_path, args.output_path, \
        args.force, **args_dict)
