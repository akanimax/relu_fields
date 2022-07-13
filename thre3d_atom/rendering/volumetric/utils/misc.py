from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import Rays, RenderOut
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose


def cast_rays(
    camera_intrinsics: CameraIntrinsics,
    pose: CameraPose,
    device: torch.device = torch.device("cpu"),
) -> Rays:
    # convert the camera pose into tensors if they are numpy arrays
    if not (isinstance(pose.rotation, Tensor) and isinstance(pose.translation, Tensor)):
        rot = torch.from_numpy(pose.rotation)
        trans = torch.from_numpy(pose.translation)
        pose = CameraPose(rot, trans)
    # bring the pose on the requested device
    if not (pose.rotation.device == device and pose.translation.device == device):
        pose = CameraPose(pose.rotation.to(device), pose.translation.to(device))

    # cast the rays for the given CameraPose
    height, width, focal = camera_intrinsics
    # note the specific use of torch.float32. Which means, even if the poses have higher
    # precision (float64), the casted rays will have 32-bit precision only.
    x_coords, y_coords = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
        indexing="ij",  # this is done to suppress the warning. Stupid PyTorch :sweat_smile:!
    )
    # not that this transpose is needed because torch's meshgrid is in ij format
    # instead of numpy's xy format
    x_coords, y_coords = x_coords.T, y_coords.T

    dirs = torch.stack(
        [
            (x_coords - width * 0.5) / focal,
            -(y_coords - height * 0.5) / focal,
            -torch.ones_like(x_coords, device=device),
        ],
        -1,
    )

    rays_d = (pose.rotation @ dirs[..., None])[..., 0]
    rays_o = torch.broadcast_to(pose.translation.squeeze(), rays_d.shape)
    return Rays(rays_o, rays_d)


def flatten_rays(rays: Rays) -> Rays:
    return Rays(
        origins=rays.origins.reshape(-1, NUM_COORD_DIMENSIONS),
        directions=rays.directions.reshape(-1, NUM_COORD_DIMENSIONS),
    )


def collate_rays(rays_list: Sequence[Rays]) -> Rays:
    """utility method for collating rays"""
    return Rays(
        origins=torch.cat([rays.origins for rays in rays_list], dim=0),
        directions=torch.cat([rays.directions for rays in rays_list], dim=0),
    )


def compute_expected_density_scale_for_relu_field_grid(
    grid_world_size: Tuple[float, float, float]
) -> float:
    """Einstien came in my dream and taught me this formula :sweat_smile: :D ..."""
    diagonal_norm = np.sqrt(
        np.sum([dim_extent**2 for dim_extent in grid_world_size])
    ).item()
    percent_density_scale, constant_grid_norm = 100.0, np.sqrt(3.0**3).item()
    return (
        (constant_grid_norm * percent_density_scale) / diagonal_norm
    ) / NUM_COORD_DIMENSIONS


def ndcize_rays(rays: Rays, camera_intrinsics: CameraIntrinsics) -> Rays:
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    """
    # unpack everything
    height, width, focal = camera_intrinsics
    near = 1.0
    rays_o, rays_d = rays

    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (width / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (height / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (width / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (height / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return Rays(rays_o, rays_d)


def sample_random_rays_and_pixels_synchronously(
    rays: Rays,
    pixels: Tensor,
    sample_size: int,
) -> Tuple[Rays, Tensor]:
    dtype, device = pixels.dtype, pixels.device
    permutation = torch.randperm(pixels.shape[0], dtype=torch.long, device=device)
    sampled_subset = permutation[:sample_size]
    rays_origins, rays_directions = rays.origins, rays.directions
    selected_rays_origins = rays_origins[sampled_subset, :]
    selected_rays_directions = rays_directions[sampled_subset, :]
    selected_pixels = pixels[sampled_subset, :]
    return Rays(selected_rays_origins, selected_rays_directions), selected_pixels


def collate_rendered_output(rendered_chunks: Sequence[RenderOut]) -> RenderOut:
    """Defines how a sequence of rendered_chunks can be
    collated into a render_out"""
    # collect all the rendered_chunks into lists
    colour, depth, extra = [], [], {}
    for rendered_chunk in rendered_chunks:
        colour.append(rendered_chunk.colour)
        depth.append(rendered_chunk.depth)
        for key, value in rendered_chunk.extra.items():
            extra[key] = extra.get(key, []) + [value]

    # combine all the tensor information
    colour = torch.cat(colour, dim=0)
    depth = torch.cat(depth, dim=0)
    extra = {key: torch.cat(extra[key], dim=0) for key in extra}

    # return the collated rendered_output
    return RenderOut(colour=colour, depth=depth, extra=extra)


def reshape_rendered_output(
    rendered_output: RenderOut, camera_intrinsics: CameraIntrinsics
) -> RenderOut:
    new_shape = (camera_intrinsics.height, camera_intrinsics.width, -1)
    return RenderOut(
        colour=rendered_output.colour.reshape(*new_shape),
        depth=rendered_output.depth.reshape(*new_shape),
        extra={
            key: value.reshape(*new_shape)
            for key, value in rendered_output.extra.items()
        },
    )
