import os
#import cv2 as cv
import omni.kit
import numpy as np
import omni.graph.core as og
import glob

from PIL import Image
from os.path import join, dirname, basename
from variables import WRITER_VIDEO_FOLDER
from typing import Dict, Union, List, Tuple, Callable
from isaac_utils import create_camera

from pxr import UsdGeom, Sdf
from omni.replicator import core as rep
from omni.replicator.core import Writer, AnnotatorRegistry
from omni.replicator.core.scripts.utils import ReplicatorItem



def save_rgb(rgb_data, file_name: str) -> None:
    """
    Save *rgb_data* image to *file_name*

    Args:
        rgb_data (buffer): RGB data
        file_name (str): Output filename
    """
    rgb_img_data = np.frombuffer(rgb_data, dtype=np.uint8)
    rgb_reshaped = rgb_img_data.reshape(*rgb_data.shape, -1)
    rgb_img = Image.fromarray(rgb_reshaped, "RGBA")

    rgb_img.save(file_name + ".png")


def clean_folder(folder: str, ext: str=".avi"):
    """
    Remove all files with extension *ext* (include folders) from *folder*

    Args:
        folder (str): Remove all files within this folder
        ext (str): Output filename
    """
    files = glob.glob(f'{join(folder,"*" + ext)}')
    clean_up(files)


def clean_up(files: List[str]):
    """
    Remove all the *files*

    Args:
        files (typing.List): List of files to delete
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)
    print("All files have been removed successfully")


# def convert_frames_to_video(path: str, delete_images: bool) -> None:
#     """
#     Convert frames to video using **

#     Args:
#         path (str): Path to the folder containing the frames
#         delete_images (bool): Whether remove all png files within *path* or not 
#     """
#     folders = glob.glob(join(path, "*/"))
#     for folder in folders:
#         folder_path = join(dirname(dirname(dirname(folder))), WRITER_VIDEO_FOLDER)
#         os.makedirs(folder_path, exist_ok=True)

#         frames = glob.glob(join(folder, '*.png'))
#         fourcc = cv.VideoWriter_fourcc(*'mp4v')

#         assert frames, f"{folder} has not frames inside"
#         img = Image.open(frames[0])
#         (width, height) = img.size
#         filename_full_path = f'{join(folder_path, basename(dirname(folder)))}.avi'

#         if os.path.exists(filename_full_path):
#             os.remove(filename_full_path)
#         video = cv.VideoWriter(filename_full_path, fourcc, 24, (width, height))
#         for frame in frames:
#             img = cv.imread(frame)
#             video.write(img)

#         cv.destroyAllWindows()
#         video.release()
        
#         if delete_images:
#             clean_up(frames)
#     return True


def orchestrated_save(frame: int, file_path: str, render_name: str, rgb_data):
    """
    Save a *frame* at *file_path* with prefix *render_name* and *rgb_data*

    Args:
        frame (int): Frame to process
        file_path (str): Folder path where to output the frame
        render_name (str): Filename of the file
        rgb_data: Annotator information
    """
    frame = str(frame).zfill(10)
    folder_path = join(file_path, render_name)
    os.makedirs(folder_path, exist_ok=True)

    if rep.orchestrator.get_is_started():
        save_rgb(rgb_data, join(folder_path, frame))


class RGBWriter(Writer):
    """ Access data through a custom replicator writer """
    def __init__(self, path: str, rp_names) -> None:
        """
        Initialize the custom replicator at *path*

        Args:
            path (str): Folder output 
        """
        self.frame_id = 0
        self.file_path = os.path.abspath(path)
        self.rp_names = rp_names
        if len(rp_names) > 1:
            self.build_rp_name = self.build_rp_name_among_many
        else:
            self.build_rp_name = self.build_singular_rp_name
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        os.makedirs(self.file_path, exist_ok=True)
        clean_folder(self.file_path, '.png')


    def build_rp_name_among_many(self, annotator_split):
        return "rgb-" + "_".join(annotator_split[1:])

    def build_singular_rp_name(self, _):
        return "rgb"

    def write(self, data: Dict) -> None:
        """
        Write ground truth *data* (annotation)

        Args:
            data (typing.Dict): Annotator data `<annotator_name>: <annotator_data>`
        """
        idx_product_name = 0
        for annotator in data.keys():
            if annotator.startswith("rp_"):
                annotator_split = annotator.split("_")
                render_product_name = self.build_rp_name(annotator_split)
                render_name = clean_replicator_name(self.rp_names[idx_product_name])
                idx_product_name += 1
                orchestrated_save(self.frame_id, self.file_path,
                                  render_name, data[render_product_name])
        self.frame_id += 1


def set_up_view_replicator(camera: Union[ReplicatorItem, str, List[str], Sdf.Path, List[Sdf.Path]], 
                           resolution: Tuple[int, int]) -> og.Node:
    """
    Create a render product. A RenderProduct describes images or other 
    file-like artifacts produced by a render, such as rgb (LdrColor), 
    normals, depth, etc.

    Args:
        camera (typing.Union[ReplicatorItem, str, typing.List[str], 
                Sdf.Path, typing.List[Sdf.Path]]): Camera replicator
        resolution (typing.Tuple[int, int]): Resolution of the camera

    Returns:
        (og.Node): Render product
    """
    return rep.create.render_product(camera, resolution=resolution)


def access_data_through_custom_writer(views: List[og.Node], 
                                      replicators: Union[str, List], 
                                      path: str) -> None:
    """
    Access data through a custom writer which has the advantage of running
    using the main loop

    Args:
        replicators ([str, list[str]], typing.Optional): 
            Render Product prim(s)
    """
    # TODO: It outputs more frames than requested, backend error not ours
    def update(frame):
        rep.orchestrator.step()

    rep.WriterRegistry.register(RGBWriter)
    rp_names = [clean_replicator_name(view[0]) for view in views]

    writer = rep.WriterRegistry.get("RGBWriter")
    writer.initialize(path=path, rp_names=rp_names)
    writer.attach(replicators)
    return update, {}


def clean_replicator_name(name):
    replicator_name = name.lower()
    return replicator_name.replace("/", "")


def access_data_through_annotators(views: List[og.Node],
                                   replicators: Union[str, List], 
                                   path: str) -> List:
    """
    Access data through annotators

    Args:
        views (typing.List[og.Node]): Camera view parameters
        replicators ([str, list[str]], str): 
            Render Product prim(s)
    """
    def update(frame, file_path, render_name, rgb_data):
        for i in range(len(rgb_data)):
            orchestrated_save(frame, file_path, render_name[i], 
                              rgb_data[i].get_data())

    registry, render_name = [], []
    file_path = os.path.abspath(path)
    for i in range(len(replicators)):
        registry.append(rep.AnnotatorRegistry.get_annotator("rgb"))
        registry[i].attach([replicators[i]])
        replicator_name = clean_replicator_name(views[i][0])
        render_name.append(replicator_name)

    rep.orchestrator.step()
    return update, {'file_path': file_path, 
                    'render_name': render_name, 'rgb_data': registry}

def get_data_through_annotations(views: List[Union[str, Tuple, Tuple, Tuple, bool]]) -> List:
    """
    Return data from cameras through annotations

    Args:
        views (typing.List[og.Node]): Camera view parameters
        replicators ([str, list[str]], str): 
            Render Product prim(s)
    """
    replicators = []
    for (prim_path, res, transl, rot, perspective) in views:
        if not perspective:
            prim_path = create_camera(prim_path, transl, rot)
        replicators.append(set_up_view_replicator(prim_path, res))
    registry = []
    for i in range(len(replicators)):
            registry.append(rep.AnnotatorRegistry.get_annotator("rgb"))
            registry[i].attach([replicators[i]])
    rep.orchestrator.step()
    return [reg.get_data() for reg in registry]

def set_up_view(views: List[Union[str, Tuple, Tuple, Tuple, bool]], 
                path: str, type_: str) -> Union[Callable, Tuple]:
    """
    Set up a camera *view* in a custom replicator writer

    Args:
        view (typing.Union[str, typing.Tuple, typing.Tuple, typing.Tuple, bool]): 
            Camera view 
        path (str): Folder where to store the output of cameras
        type_ (str): Choose between 'custom_writer' or 'annotator'
    """
    replicators = []
    for (prim_path, res, transl, rot, perspective) in views:
        if not perspective:
            prim_path = create_camera(prim_path, transl, rot)
        replicators.append(set_up_view_replicator(prim_path, res))

    replicator_func = access_data_through_custom_writer if type_ == 'custom_writer' \
        else access_data_through_annotators

    return replicator_func(views, replicators, path)
