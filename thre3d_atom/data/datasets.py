import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from torch import Tensor

from thre3d_atom.data.constants import (
    INTRINSIC,
    BOUNDS,
    HEIGHT,
    WIDTH,
    FOCAL,
    EXTRINSIC,
    ROTATION,
    TRANSLATION,
)
from thre3d_atom.data.utils import get_torch_vision_image_transform
from thre3d_atom.utils.imaging_utils import (
    CameraBounds,
    CameraIntrinsics,
    CameraPose,
    adjust_dynamic_range,
)
from thre3d_atom.utils.logging import log


class PosedImagesDataset(torch_data.Dataset):
    def __init__(
        self,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[float, float] = (0.0, 1.0),
        normalize_scene_scale: bool = False,
        downsample_factor: float = 1.0,  # no downsampling by default
        rgba_white_bkgd: bool = False,  # whether to convert rgba images to have white background
    ) -> None:
        assert images_dir.exists(), f"Images dir doesn't exist: {images_dir}"
        assert (
            camera_params_json.exists()
        ), f"CameraParams file doesn't exist: {camera_params_json}"

        super().__init__()

        # setup image file paths and corresponding camera parameters
        image_file_paths = list(images_dir.iterdir())
        with open(str(camera_params_json)) as camera_params_json_file:
            self._camera_parameters = json.load(camera_params_json_file)
        self._image_file_paths = self._filter_image_file_paths(
            image_file_paths, self._camera_parameters
        )

        # initialize the state of the object
        self._images_dir = images_dir
        self._camera_params_json = camera_params_json
        self._image_data_range = image_data_range
        self._normalize_scene_scale_bool = normalize_scene_scale
        self._downsample_factor = downsample_factor
        self._rgba_white_bkgd = rgba_white_bkgd

        self._camera_bounds = self._setup_camera_bounds()
        self._camera_intrinsics = self._setup_camera_intrinsics()
        self._image_transform = get_torch_vision_image_transform(
            new_size=(self._camera_intrinsics.height, self._camera_intrinsics.width)
        )

        # apply normalization to the camera parameters if requested
        if normalize_scene_scale:
            self._normalize_scene_scale()

        # -----------------------------------------------------------------------------------------
        #  Data Caching code :)                                                                   |
        # -----------------------------------------------------------------------------------------
        self._cached_data_mode = False
        self._cached_images, self._cached_poses = None, None
        cache_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # attempt caching of the data in gpu-memory first
            self._cached_images, self._cached_poses = self._cache_all_data(
                cache_device=cache_device
            )
            log.info(
                f"Caching of all {len(self._cached_images)} data-samples "
                f"on {cache_device} successful! yay! ..."
            )
        except RuntimeError:
            log.info(f"Couldn't fit all images on {cache_device} :(")
            log.info(f"Attempting a caching on default ``cpu'' device ... ")
            try:
                # If failed on GPU :(,  attempt caching of the data in cpu memory
                fallback_cpu_device = torch.device("cpu")
                self._cached_images, self._cached_poses = self._cache_all_data(
                    fallback_cpu_device
                )
                log.info(
                    f"Caching of all {len(self._cached_images)} data-samples "
                    f"on {fallback_cpu_device} successful! :|"
                )
            except RuntimeError:
                # if no caching worked :(, then revert to the default batch-stream mode ...
                log.info(
                    f"Unfortunately, none of the caching worked :(, so reverting to the default"
                    f"batch-streaming mode. This unfortunately slows down the pipeline by quite a bit."
                )
                # the following is not needed, but I am a paranoid programmer :D.
                # better safe than sorry ;) :D.
                self._cached_data_mode = False
        # -----------------------------------------------------------------------------------------

    @property
    def cached_data_mode(self) -> bool:
        return self._cached_data_mode

    @property
    def camera_bounds(self) -> CameraBounds:
        return self._camera_bounds

    @camera_bounds.setter
    def camera_bounds(self, camera_bounds: CameraBounds) -> None:
        # TODO: check if any specific checking code needs to be added here.
        self._camera_bounds = camera_bounds

    @property
    def camera_intrinsics(self) -> CameraIntrinsics:
        return self._camera_intrinsics

    @property
    def camera_parameters(self) -> Dict[str, Any]:
        return self._camera_parameters

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "images_dir": self._images_dir,
            "camera_params_json": self._camera_params_json,
            "image_data_range": self._image_data_range,
            "normalize_scene_scale": self._normalize_scene_scale_bool,
            "downsample_factor": self._downsample_factor,
            "rgba_white_bkgd": self._rgba_white_bkgd,
        }

    @staticmethod
    def _filter_image_file_paths(
        image_file_paths: List[Path], camera_parameters: Dict[str, Any]
    ) -> List[Path]:
        # just checks if a camera pose is available for each of the image and ignores the images
        # for which a pose is not available
        filtered_image_file_paths = image_file_paths
        if len(image_file_paths) != len(camera_parameters):
            filtered_image_file_paths = []
            img_file_names = [file_path.name for file_path in image_file_paths]
            for index, img_file_name in enumerate(img_file_names):
                if img_file_name in list(camera_parameters.keys()):
                    filtered_image_file_paths.append(image_file_paths[index])
        return filtered_image_file_paths

    def _cache_all_data(
        self, cache_device: torch.device
    ) -> Tuple[Dict[Path, Tensor], Dict[Path, Tensor]]:
        images_cache, poses_cache = {}, {}
        for image_file_path in self._image_file_paths:
            # load -> process -> cache the image
            image = Image.open(image_file_path)
            image = self._process_image(image).to(cache_device)
            images_cache[image_file_path] = image

            # load -> cache the pose
            camera_params = self._camera_parameters[image_file_path.name]
            pose = self.extract_pose(camera_params)
            unified_pose = torch.from_numpy(
                np.hstack((pose.rotation, pose.translation))
            ).to(cache_device)
            poses_cache[image_file_path] = unified_pose

        # successfully cached all the data! yay!!
        self._cached_data_mode = True

        return images_cache, poses_cache

    def _normalize_scene_scale(self):
        # obtain all the locations of the cameras and compute the distance of the farthest camera from origin
        # please note that relative rotations of the cameras remain the same
        all_poses = [
            self.extract_pose(camera_param)
            for camera_param in self._camera_parameters.values()
        ]
        all_locations = np.concatenate(
            [pose.translation for pose in all_poses], axis=-1
        )
        max_norm = np.max(np.linalg.norm(all_locations, axis=0))

        # this loop is a bit weird, but need to update dict-data structure containing all the
        # camera data to have the locations in the normalized range.
        # basically, update all the extrinsic translations by scaling them by the max_norm:
        for k, v in self._camera_parameters.items():
            old_values = self._camera_parameters[k][EXTRINSIC][TRANSLATION]
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][0][0] = str(
                (float(old_values[0][0]) / max_norm)
            )
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][1][0] = str(
                (float(old_values[1][0]) / max_norm)
            )
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][2][0] = str(
                (float(old_values[2][0]) / max_norm)
            )

        # finally, also update the scene_bounds
        self._camera_bounds = CameraBounds(
            (self._camera_bounds.near / max_norm),
            (self._camera_bounds.far / max_norm),
        )

    def get_hemispherical_radius_estimate(self) -> float:
        """estimates the radius of the hemisphere imagined by the cameras"""
        all_camera_locations = np.squeeze(
            np.array(
                [
                    camera_param[EXTRINSIC][TRANSLATION]
                    for camera_param in self._camera_parameters.values()
                ]
            ).astype(np.float32),
        )
        hemispherical_radius_estimate = (
            np.linalg.norm(all_camera_locations, axis=-1).mean().item()
        )
        return hemispherical_radius_estimate

    # noinspection PyArgumentList
    def _setup_camera_bounds(self) -> CameraBounds:
        all_bounds = np.vstack(
            [
                np.array(camera_param[INTRINSIC][BOUNDS]).astype(np.float32)
                for camera_param in self._camera_parameters.values()
            ]
        )

        near = all_bounds.min() * 0.9
        far = all_bounds.max() * 1.1
        return CameraBounds(near, far)

    def _setup_camera_intrinsics(self) -> CameraIntrinsics:
        all_camera_intrinsics = np.vstack(
            [
                np.array(
                    [
                        camera_param[INTRINSIC][HEIGHT],
                        camera_param[INTRINSIC][WIDTH],
                        camera_param[INTRINSIC][FOCAL],
                    ]
                ).astype(np.float32)
                for camera_param in self._camera_parameters.values()
            ]
        )
        # make sure that all the intrinsics are the same
        assert np.all(all_camera_intrinsics == all_camera_intrinsics[0, :])

        height, width, focal = all_camera_intrinsics[0, :] / self._downsample_factor
        return CameraIntrinsics(int(height), int(width), focal)

    def _process_image(self, image: Image.Image) -> Tensor:
        # some simple and not so interesting pre-processing of the image
        image = self._image_transform(image)
        if image.shape[0] > 3:
            if image.shape[0] == 4:
                # RGBA image case
                if self._rgba_white_bkgd:
                    # need to composite the image on a white background:
                    rgb, alpha = image[:-1, ...], image[-1:, ...]
                    image = (rgb * alpha) + (1 - alpha)
                else:
                    # premultiply the RGB with alpha to get correct
                    # interpolation
                    image = image[:3, ...] * image[3:, ...]
            else:
                # some god knows what fancy-spectral/even-non image case
                # just use the first three channels and treat them as RGB
                image = image[:3, ...]
        return image

    @staticmethod
    def extract_pose(camera_params: Dict[str, Any]) -> CameraPose:
        """could be private utility function to turn the pose in dictionary form
        into the CameraPose NamedTuple"""
        rotation = np.array(camera_params[EXTRINSIC][ROTATION]).astype(
            np.float32
        )  # 3 x 3 rotation matrix
        translation = np.array(camera_params[EXTRINSIC][TRANSLATION]).astype(
            np.float32
        )  # 3 x 1 translation vector
        return CameraPose(rotation, translation)

    def __len__(self) -> int:
        return len(self._image_file_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # pull the image at the index
        image_file_path = self._image_file_paths[index]

        # -----------------------------------------------------------------------------------------
        #  Actual data-loading code :)                                                            |
        # -----------------------------------------------------------------------------------------
        if self._cached_data_mode:
            # just access the cached images and poses if operating in the cached data mode,
            # otherwise, revert to the default behaviour of loading and pre-processing the
            # data per request.
            image, unified_pose = (
                self._cached_images[image_file_path],
                self._cached_poses[image_file_path],
            )
        else:
            # retrieve the camera_parameters of the image and make a single tensor
            # for using the pytorch's data-loading machinery :)
            camera_params = self._camera_parameters[image_file_path.name]
            pose = self.extract_pose(camera_params)
            unified_pose = torch.from_numpy(
                np.hstack((pose.rotation, pose.translation))
            )

            # load and normalize the image per request:
            image = Image.open(image_file_path)
            image = self._process_image(image)
        # -----------------------------------------------------------------------------------------

        # change the dynamic range of the image values if they are different from the pytorch's default range
        # (0.0, 1.0)
        default_image_range = (0.0, 1.0)
        if self._image_data_range != default_image_range:
            image = adjust_dynamic_range(
                image, drange_in=default_image_range, drange_out=self._image_data_range
            )

        # return the image and it's camera pose
        return image, unified_pose
