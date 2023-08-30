from typing import Dict, List

import omni.graph.core as og
import omni.replicator.core as rep


def create_material_omnipbr(kwargs: Dict) -> List[og.Node]:
    """ 
    Create an omnibpr material

    kwargs:
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
        count: int = 1,

    Example:
        >>> mat1 = create_material_omnipbr(
        ...    diffuse=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        ...    roughness=rep.distribution.uniform(0, 1),
        ...    metallic=rep.distribution.choice([0, 1]),
        ...    emissive_color=rep.distribution.uniform((0, 0, 0.5), (0, 0, 1)),
        ...    emissive_intensity=rep.distribution.uniform(0, 1000),
        ... )
    """
    return rep.create.material_omnipbr(**kwargs)


def set_replicator_lights(kwargs: Dict) -> og.Node:
    """
    Set replicator lights using *kwargs* parameters

    kwargs:
        position: Union[ReplicatorItem, float, typing.Tuple[float]] = None,
        scale: Union[ReplicatorItem, float, typing.Tuple[float]] = None,
        rotation: Union[ReplicatorItem, float, typing.Tuple[float]] = None,
        look_at: Union[ReplicatorItem, str, Sdf.Path, typing.Tuple[float, float, float]] = None,
        look_at_up_axis: Union[ReplicatorItem, typing.Tuple[float]] = None,
        light_type: str = "Distant",
        color: Union[ReplicatorItem, typing.Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        intensity: Union[ReplicatorItem, float] = 1000.0,
        exposure: Union[ReplicatorItem, float] = None,
        temperature: Union[ReplicatorItem, float] = 6500,
        texture: Union[ReplicatorItem, str] = None,
        count: int = 1,
    """
    for k, v in kwargs.items():
        if k == 'temperature' or k == 'intensity':
            kwargs[k] = rep.distribution.normal(*v)
        elif k == 'position' or k == 'scale':
            kwargs[k] = rep.distribution.uniform(*v)
        elif k == 'texture':
            kwargs[k] = rep.distribution.choice(v)
    
    lights = rep.create.light(**kwargs)
    return lights.node


def set_replicator_materials(kwargs: Dict) -> og.Node:
    """
    Set replicator to change materials (if desired also textures) 
    of the input prims using *kwargs* parameters

    kwargs:
        materials: Union[ReplicatorItem, List[str]],
        seed: int = None,
    """
    mats = create_material_omnipbr(kwargs['materials'])
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        rep.randomizer.materials(mats)
    return prims.node


def set_replicator_textures(kwargs: Dict) -> og.Node:
    """
    Set replicator to change textures of the input prims using 
    *kwargs* parameters

    kwargs:
        textures: Union[ReplicatorItem, List[str]],
        texture_scale: Union[ReplicatorItem, List[Tuple[float, float]]] = None,
        texture_rotate: Union[ReplicatorItem, List[int]] = None,
        per_sub_mesh: bool = False,
        project_uvw: bool = False,
        seed: int = None,
    """
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        rep.randomizer.texture(textures=kwargs['textures'])
    return prims.node


def set_replicator_color(kwargs: Dict) -> og.Node:
    """
    Set replicator to change color of the input prims
    using *kwargs* parameters

    kwargs:
        colors: Union[ReplicatorItem, Tuple[float]],
        per_sub_mesh: bool = False,
        seed: int = None,
    """
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        rep.randomizer.color(colors=rep.distribution.uniform(*kwargs['colors']))
    return prims.node


def set_replicator_randomize_pose(kwargs: Dict) -> og.Node:
    """
    Set replicator to randomize pose of the input prims
    using *kwargs* parameters

    kwargs:
        position: Union[ReplicatorItem, float, Tuple[float]] = None,
        rotation: Union[ReplicatorItem, float, Tuple[float]] = None,
        rotation_order: str = "XYZ",
        scale: Union[ReplicatorItem, float, Tuple[float]] = None,
        size: Union[ReplicatorItem, float, Tuple[float]] = None,
        pivot: Union[ReplicatorItem, Tuple[float]] = None,
        look_at: Union[ReplicatorItem, List[Union[str, Sdf.Path]]] = None,
        look_at_up_axis: Union[str, Tuple[float, float, float]] = None,
        input_prims: Union[ReplicatorItem, List[str]] = None,
    """
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        for k, v in kwargs:
            if k == 'position' or k == 'rotation':
                kwargs[k] = rep.distribution.uniform(*v)
            elif k == 'scale':
                kwargs[k] = rep.distribution.choice(*v)
        rep.modify.pose(**kwargs)
    return prims.node


def set_replicator_scatter_2d(kwargs: Dict) -> og.Node:
    """
    Set replicator to randomize 2d scatter the input prims
    using *kwargs* parameters

    kwargs:
        surface_prims: Union[ReplicatorItem, List[str]]: The prims 
            across which to scatter the input prims. These can be 
            meshes or GeomSubsets which specify a subset of a mesh's 
            polygons on which to scatter,
        seed: int = None,
    """
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        rep.randomizer.scatter_2d(**kwargs if kwargs else kwargs)
    return prims.node


def set_replicator_scatter_3d(kwargs: Dict) -> og.Node:
    """
    Set replicator to randomize 3d scatter the input prims
    using *kwargs* parameters

    kwargs:
        volume_prims: Union[ReplicatorItem, List[str]]: The prims within 
            which to scatter the input prims. Currently, only meshes are 
            supported,
        resolution_scaling=1.0,
        seed: int = None,
    """
    prims = rep.get.prims(path_pattern=kwargs['path_pattern'])
    del kwargs['path_pattern']

    with prims:
        rep.randomizer.scatter_3d(**kwargs if kwargs else kwargs)
    return prims.node


def set_replicator_instantiate_assets_randomly(kwargs: Dict) -> og.Node:
    """
    Set replicator to randomize instantiate assets from USD files
    using *kwargs* parameters

    kwargs['instances']:
        path_omniverse_folder: The folder in omniverse where all the
            USD are present
        size: Union[ReplicatorItem, int],
        weights: List[float] = None,
        mode: str = "scene_instance",
        with_replacements=True,
        seed: int = None,
        use_cache: bool = True,

    kwargs['pose']:
        position: Union[ReplicatorItem, float, Tuple[float]] = None,
        rotation: Union[ReplicatorItem, float, Tuple[float]] = None,
        rotation_order: str = "XYZ",
        scale: Union[ReplicatorItem, float, Tuple[float]] = None,
        size: Union[ReplicatorItem, float, Tuple[float]] = None,
        pivot: Union[ReplicatorItem, Tuple[float]] = None,
        look_at: Union[ReplicatorItem, List[Union[str, Sdf.Path]]] = None,
        look_at_up_axis: Union[str, Tuple[float, float, float]] = None,
        input_prims: Union[ReplicatorItem, List[str]] = None,
    """
    path_omniverse_folder = kwargs['instances']['path_omniverse_folder']
    del kwargs['instances']['path_omniverse_folder']

    path = rep.utils.get_usd_files(path_omniverse_folder, recursive=True)
    instances = rep.randomizer.instantiate(path, **kwargs['instances'])
    
    with instances:
        rep.modify.pose(**kwargs['pose'])
    return instances.node
