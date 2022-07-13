import copy
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import torch
from torch.nn import Module
from tqdm import tqdm

from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.rendering.volumetric.utils.misc import (
    cast_rays,
    flatten_rays,
    reshape_rendered_output,
    collate_rendered_output,
)
from thre3d_atom.thre3d_reprs.constants import (
    RENDER_CONFIG,
    RENDER_PROCEDURE,
    STATE_DICT,
    CONFIG_DICT,
    THRE3D_REPR,
    RENDER_CONFIG_TYPE,
)
from thre3d_atom.thre3d_reprs.renderers import RenderProcedure, RenderConfig
from thre3d_atom.utils.constants import EXTRA_INFO
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose


class VolumetricModel:
    def __init__(
        self,
        thre3d_repr: Module,
        render_procedure: RenderProcedure,
        render_config: RenderConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # state of the object:
        self._thre3d_repr = thre3d_repr.to(device)
        self._render_procedure = render_procedure
        self._render_config = render_config
        self._device = device

    @property
    def thre3d_repr(self) -> Module:
        return self._thre3d_repr

    @thre3d_repr.setter
    def thre3d_repr(self, thre3d_repr: Module) -> None:
        self._thre3d_repr = thre3d_repr

    @property
    def render_procedure(self) -> RenderProcedure:
        return self._render_procedure

    @property
    def render_config(self) -> RenderConfig:
        return self._render_config

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _update_render_config(
        render_config: RenderConfig, update_dict: Dict[str, Any]
    ) -> RenderConfig:
        # create a new copy for keeping the original safe
        updated_render_config = copy.deepcopy(render_config)

        # update the render configuration with the overridden kwargs:
        for field, value in update_dict.items():
            if not hasattr(updated_render_config, field):
                raise ValueError(
                    f"Unknown render configuration field {field} requested for overriding :("
                )
            setattr(updated_render_config, field, value)

        return updated_render_config

    def get_save_info(
        self, extra_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        save_info = {
            THRE3D_REPR: {
                STATE_DICT: self._thre3d_repr.state_dict(),
                CONFIG_DICT: self._thre3d_repr.get_save_config_dict(),
            },
            RENDER_PROCEDURE: self._render_procedure,
            RENDER_CONFIG_TYPE: type(self._render_config),
            RENDER_CONFIG: dataclasses.asdict(self._render_config),
        }
        if extra_info is not None:
            save_info.update({EXTRA_INFO: extra_info})
        return save_info

    def render_rays(
        self, rays: Rays, parallel_points_chunk_size: Optional[int] = None, **kwargs
    ) -> RenderOut:
        """
        renders the rays for the underlying thre3d_repr using the render
        procedure and render config ``differentiably''
        Args:
            rays: The rays to be rendered :)
            parallel_points_chunk_size: used for point-based parallelism
            **kwargs: any configuration parameters if required to be overridden
        Returns:
        """
        render_config = self._update_render_config(self._render_config, kwargs)
        return self._render_procedure(
            self._thre3d_repr, rays, render_config, parallel_points_chunk_size
        )

    def render(
        self,
        camera_pose: CameraPose,
        camera_intrinsics: CameraIntrinsics,
        parallel_rays_chunk_size: Optional[int] = 32768,
        parallel_points_chunk_size: Optional[int] = None,
        gpu_render: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> RenderOut:
        """
        renders the underlying thre3d_repr for the given camera parameters. Please
        note that this method works in pytorch's no_grad mode.
        Args:
            camera_pose: pose of the render camera
            camera_intrinsics: camera intrinsics for the render Camera
            parallel_rays_chunk_size: chunk size used for parallel ray-rendering
            parallel_points_chunk_size: chunk size used for points-based parallel processing
            gpu_render: whether to keep the rendered output on the GPU or bring to cpu. Consider turning this False
            for High resolution renders. The performance decreases quite a lot though.
            verbose: whether to show progress bar for the render.
            **kwargs: any overridden render configuration
        Returns: rendered_output :)
        """
        progress_bar = tqdm if verbose else lambda x: x

        # cast the rays for the given camera pose:
        casted_rays = cast_rays(
            camera_intrinsics=camera_intrinsics, pose=camera_pose, device=self._device
        )
        flat_rays = flatten_rays(casted_rays)

        # note that we are not using `batchify` here because of the gpu_cpu handling code
        # TODO: Improve the batchify utility to account for this following use case :)
        rendered_chunks = []
        parallel_rays_chunk_size = (
            len(flat_rays)
            if parallel_rays_chunk_size is None
            else parallel_rays_chunk_size
        )
        with torch.no_grad():
            for chunk_index in progress_bar(
                range(0, len(flat_rays), parallel_rays_chunk_size)
            ):
                rendered_chunk = self.render_rays(
                    flat_rays[chunk_index : chunk_index + parallel_rays_chunk_size],
                    parallel_points_chunk_size,
                    **kwargs,
                )
                if not gpu_render:
                    rendered_chunk = rendered_chunk.to(torch.device("cpu"))
                rendered_chunks.append(rendered_chunk)

        rendered_output = reshape_rendered_output(
            collate_rendered_output(rendered_chunks),
            camera_intrinsics=camera_intrinsics,
        )

        return rendered_output


def create_volumetric_model_from_saved_model(
    model_path: Path,
    thre3d_repr_creator: Callable[[Dict[str, Any]], Module],
    device: torch.device = torch.device("cpu"),
) -> Tuple[VolumetricModel, Dict[str, Any]]:
    # load the saved model's data using
    model_data = torch.load(model_path)
    thre3d_repr = thre3d_repr_creator(model_data)
    render_config = model_data[RENDER_CONFIG_TYPE](**model_data[RENDER_CONFIG])

    # return a newly constructed VolumetricModel using the info above
    # and the additional information saved at the time of training :)
    return (
        VolumetricModel(
            thre3d_repr=thre3d_repr,
            render_procedure=model_data[RENDER_PROCEDURE],
            render_config=render_config,
            device=device,
        ),
        model_data[EXTRA_INFO],
    )
