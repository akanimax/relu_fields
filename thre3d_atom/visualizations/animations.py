from typing import Sequence, Optional

import numpy as np

from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    CameraIntrinsics,
    scale_camera_intrinsics,
    postprocess_depth_map,
    to8b,
)
from thre3d_atom.utils.logging import log


def render_camera_path_for_volumetric_model(
    vol_mod: VolumetricModel,
    camera_path: Sequence[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    render_scale_factor: Optional[float] = None,
    overridden_num_samples_per_ray: Optional[int] = None,
) -> np.array:
    if render_scale_factor is not None:
        # Render downsampled images for speed if requested
        camera_intrinsics = scale_camera_intrinsics(
            camera_intrinsics, render_scale_factor
        )

    overridden_config_dict = {}
    if overridden_num_samples_per_ray is not None:
        overridden_config_dict.update(
            {"num_samples_per_ray": overridden_num_samples_per_ray}
        )

    rendered_frames = []
    total_frames = len(camera_path) + 1
    for frame_num, render_pose in enumerate(camera_path):
        log.info(f"rendering frame number: ({frame_num + 1}/{total_frames})")
        rendered_output = vol_mod.render(
            render_pose,
            camera_intrinsics,
            gpu_render=False,
            verbose=True,
            **overridden_config_dict,
        )
        colour_frame = rendered_output.colour.numpy()
        depth_frame = rendered_output.depth.numpy()
        acc_frame = rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()

        # apply post-processing to the depth frame
        colour_frame = to8b(colour_frame)
        depth_frame = postprocess_depth_map(
            depth_frame, vol_mod.render_config.camera_bounds
        )
        # tile the acc_frame to have 3 channels
        acc_frame = to8b(np.tile(acc_frame, (1, 1, NUM_COLOUR_CHANNELS)))

        # create grand concatenated frame horizontally
        frame = np.concatenate([colour_frame, depth_frame, acc_frame], axis=1)
        rendered_frames.append(frame)

    return np.stack(rendered_frames)
