import omni

from typing import Callable, List, Dict, Union, Tuple
from variables import (
    WIDTH_LAUNCHER_APP,
    HEIGHT_LAUNCHER_APP,
    DEFAULT_GROUND_PLANE_LIGHT,
)

from replicator import (
    set_replicator_lights,
    set_replicator_materials,
    set_replicator_color,
    set_replicator_randomize_pose,
    set_replicator_scatter_2d,
    set_replicator_scatter_3d,
    set_replicator_instantiate_assets_randomly,
)
from isaac_utils import generate_valid_prim_path

import omni.replicator.core as rep


from pxr import UsdLux, Gf, Sdf
from omni.isaac.core.utils.stage import add_reference_to_stage


class App:
    def __init__(self, simulation_app, launcher_config: Dict) -> None:
        """
        Intialize Isaac app

        Args:
            simulation_app : Application object that manages the simulator
            launcher_config (typing.Dict): Dictionary of configs (e.g. resolution)
        """
        self.launcher_config = launcher_config
        self.app = simulation_app
        self.world = None
        self.set_settings()
        self.cams = None
        self.prim = None
        self.shader = None
        self.material = None

    def save_usd_stage(self, omniverse_path) -> None:
        """
        Save stage in USD format at *omniverse_path*
        NOTE: To call before on_shutdown function call
        """
        omni.usd.get_context().save_as_stage(omniverse_path, None)

    def set_settings(self) -> None:
        """Set launcher *launcher_config* of the app"""
        self.app.set_setting(WIDTH_LAUNCHER_APP, self.launcher_config["width"])
        self.app.set_setting(HEIGHT_LAUNCHER_APP, self.launcher_config["height"])
        self.app.update()

    def set_default_light_off(self) -> None:
        """Set default light (/World/defaultLight) off (invisible)"""
        light_prim = Sdf.Path(DEFAULT_GROUND_PLANE_LIGHT)

        prim = self.world.stage.GetPrimAtPath(light_prim)
        prim.GetAttribute("visibility").Set("invisible")

    def set_default_light_params(
        self, intensity=70000, radius=50.0, translate_mtx_=(0, 0, 2)
    ) -> None:
        """
        Set light further over the invoked objects so as to not
        juxtapose with spawned objects
        """
        light_prim = Sdf.Path(DEFAULT_GROUND_PLANE_LIGHT)
        sphereLight = UsdLux.SphereLight.Define(self.world.stage, light_prim)
        sphereLight.GetIntensityAttr().Set(intensity)
        sphereLight.GetRadiusAttr().Set(radius)

        translate_mtx = Gf.Matrix4d()
        translate_mtx.SetTranslate(Gf.Vec3d(*translate_mtx_))
        ref_light_mtx = sphereLight.GetLocalTransformation()
        ref_light_mtx = ref_light_mtx * translate_mtx

        omni.kit.commands.execute(
            "TransformPrim", path=light_prim, new_transform_matrix=ref_light_mtx
        )

    def setup_replicator(self, funcs_and_params: List = []) -> None:
        """
        Setup the replicator graph with various attributes

        Args:
            funcs_and_params (typing.List[typing.Union[Callable, Dict]): List
            of functions and parameters
        """
        with rep.new_layer():
            for func, _ in funcs_and_params:
                rep.randomizer.register(func)

            with rep.trigger.on_frame():
                for func, params in funcs_and_params:
                    randomizer = getattr(rep.randomizer, func.__name__)
                    randomizer(params)

    def load_plugins(self, plugins: List[str]) -> None:
        """
        Load *plugins* into the app

        Args:
            plugins (typing.List[str]): List of plugins (str)
        """
        from omni.isaac.core.utils.extensions import enable_extension

        loaded_plugins = [enable_extension(p) for p in plugins]
        for loaded_plugin in loaded_plugins:
            if loaded_plugin:
                print(f"Extension {loaded_plugin} was not uploaded")

        if any(loaded_plugin):
            self.app.update()

    def create(
        self,
        stage_units_in_meters: float = 1.0,
        static_friction: float = 0.2,
        dynamic_friction: float = 0.2,
        restitution: float = 0.01,
        timesteps: int = 100,
        load_usd=None,
        load_usd_prim_path="/World",
    ) -> None:
        """
        Main function called after launching the script, it does
        setup the ground plane in the scene and its corresponding stage

        Args:
            stage_units_in_meters (float): The metric units of assets, this will
                affect gravity value, etc..
            static_friction (float): Static friction of the ground plane
            dynamic_friction (float): Dynamic friction of the ground plane
            restitution (float): Restitution of the ground plane
            timesteps (int): Number of steps the app runs
        """
        from omni.isaac.core import World

        self.world = World(stage_units_in_meters=stage_units_in_meters)
        if load_usd:
            prim = self.world.stage.DefinePrim(load_usd_prim_path, "")
            prim.GetReferences().AddReference(load_usd, load_usd_prim_path)
            self.app.update()
            self.world.initialize_physics()
        else:
            self.world.scene.add_default_ground_plane(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            )
        self.timesteps = timesteps
        if self.timesteps < 0:
            self.update = self.update_while
        else:
            self.update = self.update_for

    def get_world(self) -> None:
        """Return world object"""
        return self.world

    def load_prim(self, obj, obj_params: Dict, default_object: bool = False) -> None:
        """
        Main loop for updating the app with *funcs*(*params*) for
        a number of *timesteps*

        Args:
            asset_path (str): Path to the asset to load
            prim_path (str): Path to the primitive
            params (typing.Dict): Parameters used to load the asset
        """
        prim_path_key = [k for k in obj_params.keys() if k.startswith("prim_path")][0]

        obj_params[prim_path_key] = generate_valid_prim_path(obj_params[prim_path_key])
        if not default_object:
            add_reference_to_stage(
                usd_path=obj_params["asset_path"], prim_path=obj_params[prim_path_key]
            )
            del obj_params["asset_path"]

        return self.world.scene.add(obj(**obj_params))

    def prepare_function_for_update(
        self, uninstatiated_funcs_and_params
    ) -> Union[Callable, Tuple]:
        """
        Instantiate *uninstatiated_funcs_and_params* for main update function

        Args:
            uninstatiated_funcs_and_params (typing.Union[typing.Callable, typing.Tuple]]]):
                Uninstatiated functions to execute to get them ready for main update function
        """
        funcs_update, funcs_parameters = [], []
        for funcs, params in uninstatiated_funcs_and_params:
            output_funcs, output_params = funcs(*params)

            funcs_update.append(output_funcs)
            funcs_parameters.append(output_params)

        return funcs_update, funcs_parameters

    def update_for(self, funcs: List[Callable], params: List[Dict]) -> None:
        """
        Main loop for updating the app with *funcs_and_params* per timestep

        Args:
            funcs (typing.List[typing.Callable]):
                Functions to execute for each timestep
            params (typing.List[typing.Dict]):
                Dictionary of parameters of each *func*
        """
        for i in range(self.timesteps):
            for functions, parameters in zip(funcs, params):
                functions(i, **parameters if parameters else parameters)

            # execute one physics step and one rendering step
            self.world.step(render=True)

    def update_while(self, funcs: List[Callable], params: List[Dict]) -> None:
        """
        Main loop for updating the app with *funcs_and_params* per timestep

        Args:
            funcs (typing.List[typing.Callable]):
                Functions to execute for each timestep
            params (typing.List[typing.Dict]):
                Dictionary of parameters of each *func*
        """
        i = 0
        while True:
            for functions, parameters in zip(funcs, params):
                functions(i, **parameters if parameters else parameters)

            # execute one physics step and one rendering step
            self.world.step(render=True)
            i += 1

    def on_loaded_assets(self) -> None:
        """
        Resetting the world needs to be called before querying anything related
        to an articulation specifically. Its recommended to always do a reset
        after adding your assets, for physics handles to be propagated properly
        """
        self.world.reset()

    def on_shutdown(self, funcs_and_params: List[List[Union[Callable, Tuple]]]) -> None:
        """Close Isaac Sim

        Args:
            funcs_and_params (typing.List[typing.List[typing.Union[typing.Callable,
                              typing.Tuple]]]): Functions to execute once before shutdown
        """
        if funcs_and_params:
            for func, param in funcs_and_params:
                assert func(*param), "The function {func} failed with {param}"

        self.app.close()
