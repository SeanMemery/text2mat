from typing import List, Dict, Optional, Union, Callable

import omni
import omni.isaac.core.utils as utils

from variables import *

from pxr import Usd, UsdShade, Sdf, Gf
from omni.kit.material.library.material_utils import MaterialUtils


def remove_materials_when(_, func_condition: Callable, number_mtl_to_delete: int=1000,
                          func_filter: Callable=lambda x: '/Replicator/Looks/' in x):
    """
    Remove *number_mtl_to_delete* materials from stage using *func_condition* 
    after having filtered the materials satisfying *func_filter*

    Args:
        func_condition (typing.Callable): Condition to trigger the removal of
            materials, e.g. when the number of materials is larger than 1000,
        func_filter (typing.Callable): Filter materials, e.g. only delete
            materials that start with '/Replicator/Looks/',
        number_mtl_to_delete (int): How many materials will be deleted,
    """
    if func_condition():
        list_materials = list(filter(func_filter, get_list_materials()))[:number_mtl_to_delete]
        for material in list_materials:
            remove_prim(material)


async def wait_for_update(usd_context=omni.usd.get_context(), wait_frames: int=10) -> None:
    """
    *wait_frames* for the *usd_context* to update

    Args:
        usd_context (str): Context bound to running stage (default),
        wait_frames (int): Number of frames waiting for the changes to take effect,

    NOTE: There is no need to explicitly call it as the main loop takes care
    of it. However, for more complex USD updates (articulations) it might be useful
    """
    for _ in range(wait_frames):
        _, files_loaded, total_files = usd_context.get_stage_loading_status()
        await omni.kit.app.get_app().next_update_async()
        if files_loaded or total_files:
            continue


def create_and_bind_material(prim: Usd.Prim, params: Dict, material_path: str, 
                             material_name: str) -> UsdShade.Material:
    """
    Create *material_name* at *material_path* (stage) and bind it to *prim*
    using *params*

    Args:
        prims (typing.List): List of prim objects to create,
        params (typinh.Dict): Parameters used to create material (each value contains
            a list of values),
        material_path (str): Path to material inside USD stage e.g. /World/Looks/matCube,
        material_name (str): Name of the material e.g. matCube,
    """
    material, _ = create_unitialized_material(material_path, material_name)
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    with Sdf.ChangeBlock():
        mat_params = {}
        if 'diffuse_color_constant' in params:
            mat_params['diffuse_color_constant'] = Gf.Vec3f(*params['diffuse_color_constant'])
        if 'diffuse_texture' in params:
            mat_params['diffuse_texture'] = params['diffuse_texture']
        if 'reflection_roughness_constant' in params:
            mat_params['reflection_roughness_constant'] = params['reflection_roughness_constant']
        if 'reflectionroughness_texture' in params:
            mat_params['reflectionroughness_texture'] = params['reflectionroughness_texture']
        if 'metallic_constant' in params:
            mat_params['metallic_constant'] = params['metallic_constant']
        if 'metallic_texture' in params:
            mat_params['metallic_texture'] = params['metallic_texture']
        if 'specular_level' in params:
            mat_params['specular_level'] = params['specular_level']
        if 'emissive_color' in params:
            mat_params['emissive_color'] = Gf.Vec3f(*params['emissive_color'])
        if 'emissive_color_texture' in params:
            mat_params['emissive_color_texture'] = params['emissive_color_texture']
        if 'emissive_intensity' in params:
            mat_params['emissive_intensity'] = params['emissive_intensity']
        if 'project_uvw' in params:
            mat_params['project_uvw'] = params['project_uvw']

        modify_existing_material(prim, mat_params)

    return material


def create_unitialized_material(material_path: str, material_name: str, 
                                stage=omni.usd.get_context().get_stage()) -> UsdShade.Material:
    """
    Create a template USD material with *material_name* at *material_path*

    Args:
        material_path (str): Path to material inside USD stage e.g. /World/Looks/matCube,
        material_name (str): Name of the material e.g. matCube,
        stage: Default running stage,
    """
    omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand", mtl_url=OMNI_SURFACE_PATH, mtl_name=material_name, 
        mtl_path=material_path
    )
    #material_prim = stage.GetPrimAtPath(material_path)
    #shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, False))

    mtl_path = Sdf.Path(material_path)
    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(OMNI_SURFACE_PATH, "mdl")
    #shader.SetSourceAssetSubIdentifier("UsdPreviewSurface", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    ############ UsdPreviewSurface Inputs ############
    # shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)                       # 0.18, 0.18, 0.18
    # shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f)                      # 0.0, 0.0, 0.0
    # shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)                             # 0.0
    # shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)                            # 0.5
    # shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool)                        # False
    # shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)                              # 0.0
    # shader.CreateInput("ior", Sdf.ValueTypeNames.Float)                                  # 1.5
    # shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float)                            # 1.0
    # shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int)                # 0

    # ############ OmniSurface Inputs ############

    # shader.CreateInput("diffuse_reflection_weight", Sdf.ValueTypeNames.Float)                       # 0.8
    # shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f)                   # 1.0, 1.0, 1.0
    # shader.CreateInput("diffuse_reflection_roughness", Sdf.ValueTypeNames.Float)                    # 0.0
    # shader.CreateInput("metalness", Sdf.ValueTypeNames.Float)                                       # 0.0

    # ### Specular
    # shader.CreateInput("specular_reflection_weight", Sdf.ValueTypeNames.Float)                      # 1.0
    # shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float)                   # 0.2
    # shader.CreateInput("specular_reflection_anisotropy", Sdf.ValueTypeNames.Float)                  # 0.0

    ### Base
    shader.CreateInput("diffuse_reflection_weight", Sdf.ValueTypeNames.Float)                       # 0.8
    shader.CreateInput("diffuse_reflection_color", Sdf.ValueTypeNames.Color3f)                      # 1.0, 1.0, 1.0
    shader.CreateInput("diffuse_reflection_roughness", Sdf.ValueTypeNames.Float)                    # 0.0
    shader.CreateInput("metalness", Sdf.ValueTypeNames.Float)                                       # 0.0

    ### Specular
    shader.CreateInput("specular_reflection_weight", Sdf.ValueTypeNames.Float)                      # 1.0
    shader.CreateInput("specular_reflection_color", Sdf.ValueTypeNames.Color3f)                     # 1.0, 1.0, 1.0
    shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float)                   # 0.2
    #shader.CreateInput("specular_reflection_ior_preset", Sdf.ValueTypeNames.Asset)                  # "ior_custom"
    shader.CreateInput("specular_reflection_ior", Sdf.ValueTypeNames.Float)                         # 1.5
    shader.CreateInput("specular_reflection_anisotropy", Sdf.ValueTypeNames.Float)                  # 0.0
    #shader.CreateInput("specular_reflection_anisotropy_rotation", Sdf.ValueTypeNames.Float)         # 0.0

    ### Transmission
    shader.CreateInput("enable_specular_transmission", Sdf.ValueTypeNames.Bool)                     # false
    shader.CreateInput("specular_transmission_weight", Sdf.ValueTypeNames.Float)                    # 0.0
    #shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f)                   # 1.0, 1.0, 1.0
    #shader.CreateInput("specular_transmission_scattering_depth", Sdf.ValueTypeNames.Float)          # 0.0
    #shader.CreateInput("specular_transmission_scattering_color", Sdf.ValueTypeNames.Color3f)        # 0.0, 0.0, 0.0
    #shader.CreateInput("specular_transmission_scatter_anisotropy", Sdf.ValueTypeNames.Float)        # 0.0
    #shader.CreateInput("specular_transmission_dispersion_abbe", Sdf.ValueTypeNames.Float)           # 0.0

    shader.CreateInput("diffuse_reflection_color_image", Sdf.ValueTypeNames.Asset)                      # false
    shader.CreateInput("geometry_normal_image", Sdf.ValueTypeNames.Asset)                      # false

    



    # ### Subsurface
    # shader.CreateInput("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool)                      # false
    # shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float)                               # 0.0
    # shader.CreateInput("subsurface_scattering_colors_preset", Sdf.ValueTypeNames.Asset)             # "scattering_colors_custom"
    # shader.CreateInput("subsurface_transmission_color", Sdf.ValueTypeNames.Color3f)                 # 1.0, 1.0, 1.0
    # shader.CreateInput("subsurface_scattering_color", Sdf.ValueTypeNames.Color3f)                   # 1.0, 1.0, 1.0
    # shader.CreateInput("subsurface_scale", Sdf.ValueTypeNames.Float)                                # 1.0
    # shader.CreateInput("subsurface_anisotropy", Sdf.ValueTypeNames.Float)                           # 0.0

    # ### Coat
    # shader.CreateInput("coat_weight", Sdf.ValueTypeNames.Float)                                     # 0.0
    # shader.CreateInput("coat_color", Sdf.ValueTypeNames.Color3f)                                    # 1.0, 1.0, 1.0
    # shader.CreateInput("coat_roughness", Sdf.ValueTypeNames.Float)                                  # 0.1
    # shader.CreateInput("coat_ior_preset", Sdf.ValueTypeNames.Asset)                                 # "ior_custom"
    # shader.CreateInput("coat_ior", Sdf.ValueTypeNames.Float)                                        # 1.5
    # shader.CreateInput("coat_anisotropy", Sdf.ValueTypeNames.Float)                                 # 0.0
    # shader.CreateInput("coat_anisotropy_rotation", Sdf.ValueTypeNames.Float)                        # 0.0
    # shader.CreateInput("coat_affect_color", Sdf.ValueTypeNames.Float3)                               # 0.0, 0.0, 0.0
    # shader.CreateInput("coat_affect_roughness", Sdf.ValueTypeNames.Float)                           # 0.0
    # shader.CreateInput("coat_normal", Sdf.ValueTypeNames.Float3)                                    # state::normal()

    # ### Emission
    # shader.CreateInput("emission_weight", Sdf.ValueTypeNames.Float)                                # 0.0
    # shader.CreateInput("emission_mode", Sdf.ValueTypeNames.Asset)                                  # "emission_lx"
    # shader.CreateInput("emission_intensity", Sdf.ValueTypeNames.Float)                             # 1.0
    # shader.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f)                               # 1.0, 1.0, 1.0
    # shader.CreateInput("emission_use_temperature", Sdf.ValueTypeNames.Bool)                        # false
    # shader.CreateInput("emission_temperature", Sdf.ValueTypeNames.Float)                           # 6500.0

    # ### Thin Film
    # shader.CreateInput("enable_thin_film", Sdf.ValueTypeNames.Bool)                                # 0.0
    # shader.CreateInput("thin_film_thickness", Sdf.ValueTypeNames.Float)                            # 400.0
    # shader.CreateInput("thin_film_ior_preset", Sdf.ValueTypeNames.Asset)                           # "ior_custom"
    # shader.CreateInput("thin_film_ior", Sdf.ValueTypeNames.Color3f)                                # 1.52

    # ### Geometry Section
    # shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool)                                     # false
    # shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool)                                  # false
    # shader.CreateInput("geometry_opacity", Sdf.ValueTypeNames.Float)                               # 1.0
    # shader.CreateInput("geometry_opacity_threshold", Sdf.ValueTypeNames.Float)                     # 0.0
    # shader.CreateInput("geometry_normal", Sdf.ValueTypeNames.Float3)                               # state::normal()
    # shader.CreateInput("geometry_displacement", Sdf.ValueTypeNames.Float3)                         # 0.0, 0.0, 0.0
    
    # shader.CreateInput("geometry_normal_image", Sdf.ValueTypeNames.Asset)                          # 
    # shader.CreateInput("geometry_normal_strength", Sdf.ValueTypeNames.Float)                       # 1.0
    # shader.CreateInput("geometry_displacement_image", Sdf.ValueTypeNames.Asset)                          # 
    # shader.CreateInput("geometry_displacement_scale", Sdf.ValueTypeNames.Float)                       # 1.0

    return (mtl, shader)


def create_prim(kwargs: Dict) -> Usd.Prim:
    """
    Create a primitive using *kwargs*

    kwargs (Dict): Parameters used to create *prim*:
        prim_path: str,
        prim_type: str = "Xform",
        position: typing.Optional[typing.Sequence[float]] = None,
        translation: typing.Optional[typing.Sequence[float]] = None,
        orientation: typing.Optional[typing.Sequence[float]] = None,
        scale: typing.Optional[typing.Sequence[float]] = None,
        usd_path: typing.Optional[str] = None,
        semantic_label: typing.Optional[str] = None,
        semantic_type: str = "class",
        attributes: typing.Optional[dict] = None,
    """
    return utils.prims.create_prim(**kwargs)


def get_list_materials():
    """ Get list of materials """
    return MaterialUtils().get_materials_from_stage()


def modify_existing_material(prim: Usd.Prim, kwargs: Dict) -> bool:
    """
    Modify existing material

    kwargs (Dict): Parameters used to modify *prim*:
        diffuse: typing.Tuple[float] = None,
        diffuse_texture: str = None,
        roughness: float = None,
        roughness_texture: str = None,
        metallic: float = None,
        metallic_texture: str = None,
        specular: float = None,
        emissive_color: typing.Tuple[float] = None,
        emissive_texture: str = None,
        emissive_intensity: float = 0.0,
        project_uvw: bool = False,
        semantics: typing.List[typing.Tuple[str, str]] = None,

    Args:
        prim (Usd.Prim): Prim object,
    """
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(material.GetPrim(), True))

    for k, v in kwargs.items():
        shader.GetInput(k).Set(v)

    return True


def bind_material_to_object(material_path: str, prim_path: str, 
                            stage=omni.usd.get_context().get_stage()) -> None:
    """
    Bind material found at *material_path* to *prim_path*

    Args:
        material_path (str): Path to material inside USD stage e.g. /World/Looks/matCube,
        prim_path (str): Path to prim inside USD stage e.g. /World/Cube,
        stage: Default running stage,
    """
    prim = stage.GetPrimAtPath(prim_path)
    material_prim = stage.GetPrimAtPath(material_path)

    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), \
        UsdShade.Tokens.strongerThanDescendants)


def get_material_from_prim(prim: Usd.Prim) -> str:
    """
    Get material bound to *prim*

    Args:
        prim (Usd.Prim): Prim object,
    """
    binding_property = "material:binding"
    material = prim.GetRelationship(binding_property).GetTargets()

    return str(material[0]) if material else ""


def bind_texture_to_material(texture_path: str, material_path: str, prim_path: str,
                             stage=omni.usd.get_context().get_stage()):
    """
    Bind texture to a material already present in the scene

    Args:
        material_path (str): Path to material inside USD stage e.g. /World/Looks/matCube,
        texture_path (str): Path to texture using the Omniverse filesystem URI,
        prim_path (str): Path to prim inside USD stage e.g. /World/Cube,
        stage: Default running stage,

    NOTE: We assume that
    1. Both the material and prim exist in the stage
    2. The (existing) material does not possess any texture bound to it, otherwise 
    it might be better to use the function 'modify_existing_material'
    """
    prim = stage.GetPrimAtPath(prim_path)
    material_prim = stage.GetPrimAtPath(material_path)

    omni.usd.create_material_input(
        material_prim,
        "diffuse_texture",
        texture_path,
        Sdf.ValueTypeNames.Asset,
    )

    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(material_prim), \
        UsdShade.Tokens.strongerThanDescendants)


def unbind_material_to_object(prim_path: str) -> None:
    """
    Unbind the material bound at *prim_path*

    Args:
        prim_path (str): Path to prim inside stage e.g. /World/Cube,
    """
    omni.kit.commands.execute(
        "BindMaterial", prim_path=prim_path, material_path="", 
        strength=UsdShade.Tokens.strongerThanDescendants
    )


def remove_prim(prim_path: str, stage=omni.usd.get_context().get_stage()) -> None:
    """
    Remove any prim (material, object, light, etc..) at *prim_path* from stage,   

    Args:
        material_path (str): Path to material inside USD stage e.g. /World/Looks/matCube,
        stage: Default running stage,
    """
    stage.RemovePrim(Sdf.Path(prim_path))