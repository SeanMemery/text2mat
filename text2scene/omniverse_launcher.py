################## DO NOT ADD IMPORT HERE #################
from omni.isaac.kit import SimulationApp
from variables import PATH_APP_CONFIG, WRITER_FOLDER

launcher_config = {
    "width": 1280,
    "height": 720,
    "headless": False,
    "renderer": "RayTracedLighting",
}

simulation_app = SimulationApp(launcher_config, PATH_APP_CONFIG)
##########################################################

import omni
import usd_utils
import isaac_utils
import argparse
import nlp_model
import replicator
import view_replicator

from variables import DEFAULT_GROUND_PLANE_LIGHT, REMOTE_FOLDER_PREFIX, HOSTNAME
from os.path import join

from omniverse_core import App

import omni.replicator.core as rep

from pxr import UsdLux


parser = argparse.ArgumentParser(prog="Omniverse launcher", description="Text to scene")
parser.add_argument("-t", "--text", help="Input (description)")
parser.add_argument("-o", "--open", help="USD stage")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Verbose version for debugging"
)
args = parser.parse_args()

if args.text:
    (tags, spans) = nlp_model.process_description(args.text)
    (objs, actions) = nlp_model.get_pos_from_description(tags)
else:
    objs = ["person", "ball"]

app = App(simulation_app, launcher_config)

app.create(timesteps=-1, load_usd=args.open)
app.set_default_light_off()

kwargs_distant_light = {
    "type": "DistantLight",
    "angle": 0.53,
    "common_properties": {
        # 'color':,
        # 'enable_color_temperature':,
        # 'color_temperature':,
        "intensity": 4000,
        # 'exposure':,
        # 'normalize_power':,
        # 'diffuse_multiplier':,
        # 'specular_multiplier':,
    },
    "transforms": omni.usd.utils.get_local_transform_matrix(
        app.world.stage.GetPrimAtPath(DEFAULT_GROUND_PLANE_LIGHT)
    ),
}

kwargs_sphere_light = {
    "type": "SphereLight",
    "radius": 10,
    "common_properties": {
        # 'color': ,
        "enable_color_temperature": True,
        "color_temperature": 4500,
        "intensity": 5000,
        # 'exposure':,
        # 'normalize_power':,
        # 'diffuse_multiplier':,
        # 'specular_multiplier':,
    },
    "transforms": {"r": (-0.383, 0, 0, 0.924), "t": (1.0, 3.0, 5.0)},
}

kwargs_dome_light = {
    "type": "DomeLight",
    "texture_format": UsdLux.Tokens.latlong,
    "texture_file": join(
        REMOTE_FOLDER_PREFIX, HOSTNAME, "persistent/raw", "canary_wharf_4k.hdr"
    ),
    "common_properties": {
        # 'color': ,
        # 'enable_color_temperature': True,
        # 'color_temperature': 4500,
        "intensity": 500,
        # 'exposure':,
        # 'normalize_power':,
        # 'diffuse_multiplier':,
        # 'specular_multiplier':,
    },
    "transforms": {},
}

# isaac_utils.create_light(app.world.stage, kwargs_dome_light)
initialized_objs = isaac_utils.create_object(app, objs)

app.setup_replicator(
    [
        [
            replicator.set_replicator_lights,
            {
                "light_type": "Sphere",
                "position": ((0, 0, 3), (0, 0, 4)),
                "scale": (0, 1),
                "intensity": (5000, 35000),
                "temperature": (500, 6500),
                "count": 10,
            },
        ],
        [
            replicator.set_replicator_color,
            {"path_pattern": "/World/Person", "colors": ((0, 0, 0), (1, 1, 1))},
        ],
        [
            replicator.set_replicator_textures,
            {
                "path_pattern": "/World/Ball",
                "textures": [
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
                ],
            },
        ],
        [
            replicator.set_replicator_materials,
            {
                "path_pattern": "/World/Ball",
                "materials": {
                    "diffuse": rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
                    "count": 1,
                },
            },
        ],
    ]
)

### UTILS TO CREATE MATERIALS WITHOUT REPLICATOR
# prim = usd_utils.create_prim({'prim_path': "/World/Cube10", 'prim_type': "Cube"})
# mat = usd_utils.create_unitialized_material("/World/Looks/matCube", "matCube")

# usd_utils.bind_material_to_object("/World/Looks/matCube", "/World/Cube10")
# usd_utils.bind_texture_to_material('omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg', "/World/Looks/matCube", "/World/Cube10")
# usd_utils.unbind_material_to_object("/World/Cube10")

# usd_utils.create_and_bind_material(prim, {
#     'reflection_roughness_constant': 0.5,
#     'diffuse_color_constant': (0.200000003, 0.200000003, 0.200000003),
#     'emissive_color': (1, 0.100000001, 0.100000001),
#     'metallic_constant': 0.5,
#     }, "/World/Looks/matCube2", "matCube2")
# print('prim', usd_utils.get_material_from_prim(prim))
#########

app.on_loaded_assets()

# The functions to be sent to the pipeline are
main_pipeline = [
    [
        view_replicator.set_up_view,
        (
            [
                [
                    "/World/Camera1",
                    (320, 320),
                    (0.0, 10.0, 20.0),
                    (-15.0, 0.0, 0.0),
                    False,
                ],
                [
                    "/World/Camera2",
                    (640, 640),
                    (-10.0, 15.0, 15.0),
                    (-45.0, 0.0, 45.0),
                    False,
                ],
                ["/OmniverseKit_Persp", (1024, 1024), None, None, True],
            ],
            WRITER_FOLDER,
            "annotator",
        ),
    ],
]

shutdown_pipeline = [
    [view_replicator.convert_frames_to_video, (WRITER_FOLDER, False)],
]
funcs_update, funcs_parameters = app.prepare_function_for_update(main_pipeline)

### REMOVE MATERIALS
# funcs_update.append(usd_utils.remove_materials_when)
# funcs_parameters.append({'func_condition': lambda: len(usd_utils.get_list_materials()) > 10, 'number_mtl_to_delete': 10})
############

app.update(funcs_update, funcs_parameters)

app.save_usd_stage(join(REMOTE_FOLDER_PREFIX, HOSTNAME, "persistent/world.usd"))

app.on_shutdown(shutdown_pipeline)
