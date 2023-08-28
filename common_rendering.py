import sys, torch
import numpy as np
from typing import Dict

from variables import PROJECT_PATH

text2scene_abs_path = f"{PROJECT_PATH}/src/text2scene/"

async def gen_mdl(prim, name):
    import sys, asyncio
    sys.path.append(text2scene_abs_path+"../")
    import usd_converter
    await asyncio.wait_for(usd_converter.export_to_mdl(f"./mdl/{name}.mdl", prim), timeout=5)
    with open(f"./mdl/{name}.mdl", "r") as f:
        data = f.readlines()
    data[13] = "import ::OmniSurface::OmniSurface;\n"
    data[18] = " = ::OmniSurface::OmniSurface(\n"
    with open(f"./mdl/{name}.mdl", "w") as f:
        f.writelines(data)

def set_shader_vals(mat_vec, app, v):
    from pxr import Gf

    #app.shader.GetInput("diffuseColor").Set(Gf.Vec3f( 0.75 * float(v[0]), 0.75 * float(v[1]), 0.75 * float(v[2])))                       # 0.18, 0.18, 0.18
    # if float(v[3]) < 0.9:
    #     app.shader.GetInput("enable_opacity").Set(True)      
    #     app.shader.GetInput("opacity").Set(float(v[3]))
    #     app.shader.GetInput("specularColor").Set(Gf.Vec3f( 0.75 * float(v[0]), 0.75 * float(v[1]), 0.75 * float(v[2])))                       # 0.18, 0.18, 0.18
    #     app.shader.GetInput("useSpecularWorkflow").Set(1)
    
    # app.shader.GetInput("metallic").Set(float(mat_vec[0]))                               # 0.0
    # app.shader.GetInput("roughness").Set(float(mat_vec[1]))                              # 0.5
    # app.shader.GetInput("occlusion").Set(float(mat_vec[2]))                              # 1.0

    # app.shader.GetInput("enable_opacity").Set(True)                                      # False
    # app.shader.GetInput("opacity").Set(float(mat_vec[2]))                                # 0.0
    # app.shader.GetInput("ior").Set(1.0 + float(mat_vec[3]))                              # 1.5

    ### Base
    app.shader.GetInput("diffuse_reflection_weight").Set(float(mat_vec[0]))                       # 0.8
    app.shader.GetInput("diffuse_reflection_color").Set(Gf.Vec3f( float(v[0]), float(v[1]), float(v[2])))                   # 1.0, 1.0, 1.0
    app.shader.GetInput("diffuse_reflection_roughness").Set(float(mat_vec[1]))                    # 0.0
    app.shader.GetInput("metalness").Set(float(mat_vec[2]))                                       # 0.0

    ### Specular
    app.shader.GetInput("specular_reflection_weight").Set(float(mat_vec[3]))                      # 1.0
    app.shader.GetInput("specular_reflection_roughness").Set(float(mat_vec[4]))                   # 0.2
    app.shader.GetInput("specular_reflection_anisotropy").Set(float(mat_vec[5]))                  # 0.0
    #app.shader.GetInput("specular_reflection_ior").Set(float(v[4]))         # 1.5

    ### Transmission
    #app.shader.GetInput("enable_specular_transmission").Set(True)                    # 0.0
    #app.shader.GetInput("specular_transmission_weight").Set(1.0 - float(v[3]))                                 # 0.0
   
def get_shader(mat_vec, name, kwargs: Dict, cols):
    app = kwargs["app"]

    set_shader_vals(mat_vec, app, cols)

    app.world.step()

    import asyncio
    asyncio.ensure_future(gen_mdl(name))

def reset_shader_vals(app):
    from pxr import Gf
    app.shader.GetInput("enable_opacity").Set(False)      
    app.shader.GetInput("useSpecularWorkflow").Set(0)
    app.shader.GetInput("diffuseColor").Set(Gf.Vec3f( 0.18, 0.18, 0.18))                      # 0.18, 0.18, 0.18
    app.shader.GetInput("specularColor").Set(Gf.Vec3f( 0.18, 0.18, 0.18 ))                       # 0.18, 0.18, 0.18

def get_image(mat_vec, kwargs: Dict, v):
    app = kwargs["app"]
    num_angles = kwargs["training_images"]
    res = kwargs["res"]
    num_mats = len(mat_vec)

    frames = np.ones((num_angles, num_mats, res, res, 4), dtype=np.uint8)

    remove_err = 1
    for i in range(-remove_err, num_mats):
    
        set_shader_vals(mat_vec[i], app, v[i])
        app.world.step()

        for j in range(num_angles):

            data = app.cams["rgb_data"][j].get_data()
            img = np.frombuffer(data, dtype=np.uint8).reshape(res, res, 4)
                
            ### Fix rotations
            if j==0:
                img = np.rot90(img, k=1, axes=(1,0))
            elif j==1:
                img = np.rot90(img, k=1, axes=(0,1))
            elif j==2:
                img = np.rot90(img, k=2, axes=(0,1))
                
            frames[j][i] = img

    ### NOTE: This is necessary as the rendering is one off !!!
    for j in range(num_angles):
        frames[j] = np.concatenate((frames[j][1:], np.expand_dims(frames[j][0], axis=0)))
    
    return frames

def gen_cam_pos():
    p = 130
    p_sqrt2 = p / np.sqrt(2)
    return [
        (p, 0, 0),
        (-p, 0, 0),
        (0, p, 0),
        (0, -p, 0),
        ( p_sqrt2,  p_sqrt2, 0),  
        (-p_sqrt2, -p_sqrt2, 0),   
        (-p_sqrt2,  p_sqrt2, 0),  
        ( p_sqrt2, -p_sqrt2, 0),   
    ]

def gen_cam_rot():
    r = 90
    r_2 = r / 2
    return [
        (0, r, 0),
        (0, -r, 0),
        (-r, 0, 0),
        (r, 0, 0),
        (-r, 180, -r_2),
        (r, 0, -r_2), 
        (-r, 180, r_2),
        (r, 0, r_2),  
    ]

def create_app(config=None):
    if config is None:
        res=512
        num_images=1
        samples_per_pixel_per_frame=256
    else:
        res = config["model_params"]["res"]
        num_images = config["model_params"]["training_images"]
        samples_per_pixel_per_frame = config["model_params"]["samples_per_pixel_per_frame"]
    sys.path.append(text2scene_abs_path)
    ################## DO NOT ADD IMPORT HERE #################
    from omni.isaac.kit import SimulationApp
    from variables import PATH_APP_CONFIG
    launcher_config = {
        "width": res,
        "height": res,
        "headless": True,
        "renderer": "PathTracing",
        "samples_per_pixel_per_frame": 256,
        "anti_aliasing": 4,
        "denoiser": False,
        "max_bounces": 8,
        "max_specular_transmission_bounces": 8,
        "fast_shutdown": True,
        "subdiv_refinement_level": 4,
    }
    simulation_app = SimulationApp(launcher_config, PATH_APP_CONFIG)
    ##########################################################
    import omni.replicator.core as rep
    from usd_utils import remove_prim
    from omniverse_core import App
    from pxr import UsdLux
    import isaac_utils
    import omni, view_replicator
    from variables import WRITER_FOLDER
    import omni
    from usd_utils import create_unitialized_material
    from pxr import UsdShade
    app = App(simulation_app, launcher_config)
    app.create(timesteps=1)
    kwargs_sphere_light = {
        'type': 'SphereLight',
        'radius': 75,
        'common_properties': {
            'color': (1,1,1),
            'enable_color_temperature': True,
            'color_temperature': 4500,
            'intensity': 15000,
            # 'exposure':,
            # 'normalize_power':,
            # 'diffuse_multiplier':,
            # 'specular_multiplier':,
        },
        'transforms': {
            't': (189, 327, 180)
        }
    }
    kwargs_dome_light = {
        'type': 'DomeLight',
        'texture_format': UsdLux.Tokens.latlong,
        'texture_file': f"{text2scene_abs_path}/../ML/data/env_map.hdr", #f"{text2scene_abs_path}../ML/data/evening_road_01.hdr",          #env_map.hdr",
        'common_properties': {
            # 'color': ,
            # 'enable_color_temperature': True,
            # 'color_temperature': 4500,
            'intensity': 1100,
            # 'exposure':,
            # 'normalize_power':,
            # 'diffuse_multiplier':,
            # 'specular_multiplier':,
        },
        'transforms': {
        }
    }

    
    num_cams = num_images
    cam_inputs = []
    pos = gen_cam_pos()
    rot = gen_cam_rot()
    assert num_cams <= len(pos), f"Number of cameras cannot be more than {len(pos)}"
    for i in range(num_cams):
        cam_inputs.append([f"/World/Camera{i+1}", (res, res), pos[i], rot[i], False])

    _, app.cams = view_replicator.set_up_view(cam_inputs, WRITER_FOLDER, "annotator")
    sphereprim = rep.create.sphere(semantics=[('class', 'Sphere')] , position=(0, 0 , 0), rotation=(0, 0, 0), scale=(0.5, 0.5, 0.5))
    #isaac_utils.create_light(app.world.stage, kwargs_sphere_light)
    isaac_utils.create_light(app.world.stage, kwargs_dome_light)
    remove_prim("/World/defaultGroundPlane")
    app.on_loaded_assets()
    remove_prim("/World/defaultGroundPlane")

    ### Set material
    app.prim = omni.usd.get_context().get_stage().GetPrimAtPath("/Replicator/Sphere_Xform/Sphere")
    app.material, app.shader = create_unitialized_material(f"/World/Looks/Sphere_Xform/matSphere", "OmniSurfaceBase")
    UsdShade.MaterialBindingAPI(app.prim).Bind(UsdShade.Material(app.material), UsdShade.Tokens.strongerThanDescendants)
    ###

    ### Explicitly set samples per pixel
    rtx_mode = "/rtx"
    app.app.set_setting(rtx_mode + "/pathtracing/spp", samples_per_pixel_per_frame)
    app.app.set_setting(
        rtx_mode + "/pathtracing/totalSpp", samples_per_pixel_per_frame
    )
    app.app.set_setting(
        rtx_mode + "/pathtracing/clampSpp", samples_per_pixel_per_frame
    )

    # Step to remove errors
    app.world.step()
    app.world.step()
    app.world.step()

    return app 