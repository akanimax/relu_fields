import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import imageio
import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from thre3d_atom.data.datasets import PosedImagesDataset
from thre3d_atom.data.utils import infinite_dataloader
from thre3d_atom.modules.testers import test_sh_vox_grid_vol_mod_with_posed_images
from thre3d_atom.modules.volumetric_model import VolumetricModel
from thre3d_atom.rendering.volumetric.utils.misc import (
    cast_rays,
    collate_rays,
    sample_random_rays_and_pixels_synchronously,
    flatten_rays,
)
from thre3d_atom.thre3d_reprs.renderers import render_sh_voxel_grid
from thre3d_atom.thre3d_reprs.voxels import (
    VoxelGrid,
    scale_voxel_grid_with_required_output_size,
)
from thre3d_atom.utils.constants import (
    CAMERA_BOUNDS,
    CAMERA_INTRINSICS,
    HEMISPHERICAL_RADIUS,
)
from thre3d_atom.utils.imaging_utils import CameraPose, to8b

# All the TrainProcedures below follow this function-type
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.metric_utils import mse2psnr
from thre3d_atom.utils.misc import compute_thre3d_grid_sizes
from thre3d_atom.visualizations.static import (
    visualize_camera_rays,
    visualize_sh_vox_grid_vol_mod_rendered_feedback,
)


# TrainProcedure = Callable[[VolumetricModel, Dataset, ...], VolumetricModel]


def train_sh_vox_grid_vol_mod_with_posed_images(
    vol_mod: VolumetricModel,
    train_dataset: PosedImagesDataset,
    # required arguments:
    output_dir: Path,
    # optional arguments:)
    random_initializer: Callable[[Tensor], Tensor] = partial(
        torch.nn.init.uniform_, a=-1.0, b=1.0
    ),
    test_dataset: Optional[PosedImagesDataset] = None,
    image_batch_cache_size: int = 8,
    ray_batch_size: int = 32768,
    num_stages: int = 4,
    num_iterations_per_stage: int = 2000,
    scale_factor: float = 2.0,
    # learning_rate and related arguments
    learning_rate: float = 0.03,
    lr_decay_gamma_per_stage: float = 0.1,
    lr_decay_steps_per_stage: int = 1000,
    stagewise_lr_decay_gamma: float = 0.9,
    # option to have a specific feedback_pose_for_visual feedback rendering
    render_feedback_pose: Optional[CameraPose] = None,
    # various training-loop frequencies
    save_freq: int = 1000,
    test_freq: int = 1000,
    feedback_freq: int = 100,
    summary_freq: int = 10,
    # regularization option:
    apply_diffuse_render_regularization: bool = True,
    # miscellaneous options can be left untouched
    num_workers: int = 4,
    verbose_rendering: bool = True,
    fast_debug_mode: bool = False,
) -> VolumetricModel:
    """
    ------------------------------------------------------------------------------------------------------
    |                               !!! :D LONG FUNCTION ALERT :D !!!                                    |
    ------------------------------------------------------------------------------------------------------
    trains a volumetric model given a dataset of images and corresponding poses
    Args:
        vol_mod: the volumetricModel to be trained with this procedure. Please note that it should have
                 an sh-based VoxelGrid as its underlying thre3d_repr.
        train_dataset: PosedImagesDataset used for training
        output_dir: path to the output directory where the assets of the training are to be written
        random_initializer: the pytorch initialization routine used for features of voxel_grid
        test_dataset: optional dataset of test images and poses :)
        image_batch_cache_size: batch of images from which rays are sampled per training iteration
        ray_batch_size: number of randomly sampled rays used per training iteration
        num_stages: number of stages in the training routine
        num_iterations_per_stage: iterations performed per stage
        scale_factor: factor by which the grid is up-scaled after each stage
        learning_rate: learning rate used for differential optimization
        lr_decay_gamma_per_stage: value of gamma for learning rate-decay in a single stage
        lr_decay_steps_per_stage: steps after which exponential learning rate decay is kicked in
        stagewise_lr_decay_gamma: gamma reduction of learning rate after each stage
        render_feedback_pose: optional feedback pose used for generating the rendered feedback
        save_freq: number of iterations after which checkpoints are saved
        test_freq: number of iterations after which testing scores are computed
        feedback_freq: number of iterations after which feedback is generated
        summary_freq: number of iterations after which current loss is logged to console
        apply_diffuse_render_regularization: whether to apply the diffuse render regularization
        num_workers: num_workers used by pytorch dataloader
        verbose_rendering: bool to control whether to show verbose details while generating rendered feedback
        fast_debug_mode: bool to control fast_debug_mode, skips testing and some other things
    Returns: the trained version of the VolumetricModel. Also writes multiple assets to disk
    """
    # assertions about the VolumetricModel being used with this TrainProcedure :)
    assert isinstance(vol_mod.thre3d_repr, VoxelGrid), (
        f"sorry, cannot use a {type(vol_mod.thre3d_repr)} with this TrainProcedure :(; "
        f"only a {type(VoxelGrid)} can be used"
    )
    assert (
        vol_mod.render_procedure == render_sh_voxel_grid
    ), f"sorry, non SH-based VoxelGrids cannot be used with this TrainProcedure"

    # fix the sizes of the feature grids at different stages
    stagewise_voxel_grid_sizes = compute_thre3d_grid_sizes(
        final_required_resolution=vol_mod.thre3d_repr.grid_dims,
        num_stages=num_stages,
        scale_factor=scale_factor,
    )

    # create downsampled versions of the train_dataset for lower training stages
    stagewise_train_datasets = [train_dataset]
    dataset_config_dict = train_dataset.get_config_dict()
    for stage in range(1, num_stages):
        dataset_config_dict.update({"downsample_factor": (scale_factor**stage)})
        stagewise_train_datasets.insert(0, PosedImagesDataset(**dataset_config_dict))

    # downscale the feature-grid to the smallest size:
    with torch.no_grad():
        # TODO: Possibly create a nice interface for reprs as a resolution of the below warning
        # noinspection PyTypeChecker
        vol_mod.thre3d_repr = scale_voxel_grid_with_required_output_size(
            vol_mod.thre3d_repr,
            output_size=stagewise_voxel_grid_sizes[0],
            mode="trilinear",
        )
        # reinitialize the scaled features and densities to remove any bias
        random_initializer(vol_mod.thre3d_repr.densities)
        random_initializer(vol_mod.thre3d_repr.features)

    # setup render_feedback_pose
    real_feedback_image = None
    if render_feedback_pose is None:
        feedback_dataset = test_dataset if test_dataset is not None else train_dataset
        render_feedback_pose = CameraPose(
            rotation=feedback_dataset[0][-1][:, :3].cpu().numpy(),
            translation=feedback_dataset[0][-1][:, 3:].cpu().numpy(),
        )
        real_feedback_image = feedback_dataset[0][0].permute(1, 2, 0).cpu().numpy()

    train_dl = _make_dataloader_from_dataset(
        train_dataset, image_batch_cache_size, num_workers
    )
    test_dl = (
        _make_dataloader_from_dataset(test_dataset, 1, num_workers)
        if test_dataset is not None
        else None
    )

    # dataset size aka number of total pixels
    dataset_size = (
        len(train_dl)
        * train_dataset.camera_intrinsics.height
        * train_dataset.camera_intrinsics.width
    )

    # setup output directories
    # fmt: off
    model_dir = output_dir / "saved_models"
    logs_dir = output_dir / "training_logs"
    tensorboard_dir = logs_dir / "tensorboard"
    render_dir = logs_dir / "rendered_output"
    for directory in (model_dir, logs_dir, tensorboard_dir,
                      render_dir):
        directory.mkdir(exist_ok=True, parents=True)
    # fmt: on

    # save the real_feedback_test_image if it exists:
    if real_feedback_image is not None:
        log.info(f"Logging real feedback image")
        imageio.imwrite(
            render_dir / f"1__real_log.png",
            to8b(real_feedback_image),
        )

    # extract the camera_bounds and camera_intrinsics for rest of the procedure
    camera_bounds, camera_intrinsics = (
        train_dataset.camera_bounds,
        train_dataset.camera_intrinsics,
    )

    # setup tensorboard writer
    tensorboard_writer = SummaryWriter(str(tensorboard_dir))

    # create camera-rays visualization:
    if not fast_debug_mode:
        log.info(
            "creating a camera-rays visualization... please wait... "
            "this is a slow operation :D"
        )
        visualize_camera_rays(
            train_dataset,
            output_dir,
            num_rays_per_image=1,
        )

    # start actual training
    log.info("beginning training")
    time_spent_actually_training = 0

    # -----------------------------------------------------------------------------------------
    #  Main Training Loop                                                                     |
    # -----------------------------------------------------------------------------------------
    for stage in range(1, num_stages + 1):
        # setup the dataset for the current training stage
        # followed by creating an infinite training data-loader
        current_stage_train_dataset = stagewise_train_datasets[stage - 1]
        train_dl = _make_dataloader_from_dataset(
            current_stage_train_dataset, image_batch_cache_size, num_workers
        )
        infinite_train_dl = iter(infinite_dataloader(train_dl))

        # setup volumetric_model's optimizer
        current_stage_lr = learning_rate * (stagewise_lr_decay_gamma ** (stage - 1))
        optimizeable_parameters = vol_mod.thre3d_repr.parameters()
        assert (
            optimizeable_parameters
        ), f"No optimizeable parameters :(. Nothing will happen"
        optimizer = torch.optim.Adam(
            params=[{"params": optimizeable_parameters, "lr": current_stage_lr}],
            betas=(0.9, 0.999),
        )

        # setup learning rate schedulers for the optimizer
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_gamma_per_stage
        )

        # display logs related to this training stage:
        train_image_height, train_image_width = (
            current_stage_train_dataset.camera_intrinsics.height,
            current_stage_train_dataset.camera_intrinsics.width,
        )
        log.info(
            f"training stage: {stage}   "
            f"voxel grid resolution: {vol_mod.thre3d_repr.grid_dims} "
            f"training images resolution: [{train_image_height} x {train_image_width}]"
        )
        current_stage_lrs = [
            param_group["lr"] for param_group in optimizer.param_groups
        ]
        log_string = f"current stage learning rates: {current_stage_lrs} "
        log.info(log_string)
        last_time = time.perf_counter()
        # -------------------------------------------------------------------------------------
        #  Single Stage Training Loop                                                         |
        # -------------------------------------------------------------------------------------
        for stage_iteration in range(1, num_iterations_per_stage + 1):
            # ---------------------------------------------------------------------------------
            #  Main Operations Performed Per Iteration                                        |
            # ---------------------------------------------------------------------------------
            # sample a batch rays and pixels for a single iteration
            # load a batch of images and poses (These could already be cached on GPU)
            # please check the `data.datasets` module
            images, poses = next(infinite_train_dl)

            # cast rays for all the loaded images:
            rays_list = []
            for pose in poses:
                casted_rays = flatten_rays(
                    cast_rays(
                        current_stage_train_dataset.camera_intrinsics,
                        CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                        device=vol_mod.device,
                    )
                )
                rays_list.append(casted_rays)
            rays = collate_rays(rays_list)

            # images are of shape [B x C x H x W] and pixels are [B * H * W x C]
            pixels = (
                images.permute(0, 2, 3, 1)
                .reshape(-1, images.shape[1])
                .to(vol_mod.device)
            )

            # sample a subset of rays and pixels synchronously
            rays_batch, pixels_batch = sample_random_rays_and_pixels_synchronously(
                rays, pixels, ray_batch_size
            )

            # render a small chunk of rays and compute a loss on it
            specular_rendered_batch = vol_mod.render_rays(rays_batch)
            specular_rendered_pixels_batch = specular_rendered_batch.colour

            # compute loss and perform gradient update
            # Main, specular loss
            total_loss = l1_loss(specular_rendered_pixels_batch, pixels_batch)

            # logging info:
            specular_loss_value = total_loss
            specular_psnr_value = mse2psnr(
                mse_loss(specular_rendered_pixels_batch, pixels_batch)
            )

            # Diffuse render loss, for better and stabler geometry extraction if requested:
            diffuse_loss_value, diffuse_psnr_value = None, None
            if apply_diffuse_render_regularization:
                # render only the diffuse version for the rays
                diffuse_rendered_batch = vol_mod.render_rays(
                    rays_batch, render_diffuse=True
                )
                diffuse_rendered_pixels_batch = diffuse_rendered_batch.colour

                # compute diffuse loss
                diffuse_loss = l1_loss(diffuse_rendered_pixels_batch, pixels_batch)
                total_loss = total_loss + diffuse_loss

                # logging info:
                diffuse_loss_value = diffuse_loss
                diffuse_psnr_value = mse2psnr(
                    mse_loss(diffuse_rendered_pixels_batch, pixels_batch)
                )

            # optimization steps:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # ---------------------------------------------------------------------------------

            # rest of the code per iteration is related to saving/logging/feedback/testing
            time_spent_actually_training += time.perf_counter() - last_time
            global_step = ((stage - 1) * num_iterations_per_stage) + stage_iteration

            # tensorboard summaries feedback
            if (
                global_step % summary_freq == 0
                or stage_iteration == 1
                or stage_iteration == num_iterations_per_stage
            ):
                for summary_name, summary_value in (
                    ("specular_loss", specular_loss_value),
                    ("diffuse_loss", diffuse_loss_value),
                    ("specular_psnr", specular_psnr_value),
                    ("diffuse_psnr", diffuse_psnr_value),
                    ("total_loss", total_loss),
                    ("num_epochs", (ray_batch_size * global_step) / dataset_size),
                ):
                    if summary_value is not None:
                        tensorboard_writer.add_scalar(
                            summary_name, summary_value, global_step=global_step
                        )

            # console loss feedback
            if (
                global_step % summary_freq == 0
                or stage_iteration == 1
                or stage_iteration == num_iterations_per_stage
            ):
                loss_info_string = (
                    f"Stage: {stage} "
                    f"Global Iteration: {global_step} "
                    f"Stage Iteration: {stage_iteration} "
                    f"specular_loss: {specular_loss_value.item(): .3f} "
                    f"specular_psnr: {specular_psnr_value.item(): .3f} "
                )
                if apply_diffuse_render_regularization:
                    loss_info_string += (
                        f"diffuse_loss: {diffuse_loss_value.item(): .3f} "
                        f"diffuse_psnr: {diffuse_psnr_value.item(): .3f} "
                        f"total_loss: {total_loss: .3f} "
                    )
                log.info(loss_info_string)

            # step the learning rate schedulers
            if stage_iteration % lr_decay_steps_per_stage == 0:
                lr_scheduler.step()
                new_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                log_string = f"Adjusted learning rate | learning rates: {new_lrs} "
                log.info(log_string)

            # generated rendered feedback visualizations
            if (
                global_step % feedback_freq == 0
                or stage_iteration == 1
                or stage_iteration == num_iterations_per_stage
            ):
                log.info(
                    f"TIME CHECK: time spent actually training "
                    f"till now: {timedelta(seconds=time_spent_actually_training)}"
                )
                visualize_sh_vox_grid_vol_mod_rendered_feedback(
                    vol_mod=vol_mod,
                    render_feedback_pose=render_feedback_pose,
                    camera_intrinsics=camera_intrinsics,
                    global_step=global_step,
                    feedback_logs_dir=render_dir,
                    parallel_rays_chunk_size=vol_mod.render_config.parallel_rays_chunk_size,
                    training_time=time_spent_actually_training,
                    log_diffuse_rendered_version=True,
                    overridden_num_samples_per_ray=vol_mod.render_config.render_num_samples_per_ray,
                    verbose_rendering=verbose_rendering,
                )

            # obtain and log the test metrics
            if (
                test_dl is not None
                and not fast_debug_mode
                and (
                    global_step % test_freq == 0
                    or stage_iteration == num_iterations_per_stage
                )
            ):
                test_sh_vox_grid_vol_mod_with_posed_images(
                    vol_mod=vol_mod,
                    test_dl=test_dl,
                    parallel_rays_chunk_size=ray_batch_size,
                    tensorboard_writer=tensorboard_writer,
                    global_step=global_step,
                )

            # save the model
            if (
                global_step % save_freq == 0
                or stage_iteration == 1
                or stage_iteration == num_iterations_per_stage
            ):
                log.info(
                    f"saving model-snapshot at stage {stage}, global step {global_step}"
                )
                torch.save(
                    vol_mod.get_save_info(
                        extra_info={
                            CAMERA_BOUNDS: camera_bounds,
                            CAMERA_INTRINSICS: camera_intrinsics,
                            HEMISPHERICAL_RADIUS: train_dataset.get_hemispherical_radius_estimate(),
                        }
                    ),
                    model_dir / f"model_stage_{stage}_iter_{global_step}.pth",
                )

            # ignore all the time spent doing verbose stuff :) and update
            # the last_time clock event
            last_time = time.perf_counter()
        # -------------------------------------------------------------------------------------

        # don't upsample the feature grid if the last stage is complete
        if stage != num_stages:
            # upsample the feature-grid after the completion of the stage:
            with torch.no_grad():
                # noinspection PyTypeChecker
                vol_mod.thre3d_repr = scale_voxel_grid_with_required_output_size(
                    vol_mod.thre3d_repr,
                    output_size=stagewise_voxel_grid_sizes[stage],
                    mode="trilinear",
                )
    # -----------------------------------------------------------------------------------------

    # save the final trained model
    log.info(f"Saving the final model-snapshot :)! Almost there ... yay!")
    torch.save(
        vol_mod.get_save_info(
            extra_info={
                "camera_bounds": camera_bounds,
                "camera_intrinsics": camera_intrinsics,
                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
            }
        ),
        model_dir / f"model_final.pth",
    )

    # training complete yay! :)
    log.info("Training complete")
    log.info(
        f"Total actual training time: {timedelta(seconds=time_spent_actually_training)}"
    )
    return vol_mod


def _make_dataloader_from_dataset(
    dataset: PosedImagesDataset, batch_size: int, num_workers: int = 0
) -> DataLoader:
    # setup the data_loader:
    # There are a bunch of fancy CPU-GPU configuration being done here.
    # Nothing too hard to understand, just refer the documentation page of PyTorch's
    # dataloader -> https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # And, read the book titled "CUDA_BY_EXAMPLE" https://developer.nvidia.com/cuda-example
    # Takes not long, just about 1-2 weeks :). But worth it :+1: :+1: :smile:!
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 if dataset.cached_data_mode else dataset,
        pin_memory=not dataset.cached_data_mode and num_workers > 0,
        prefetch_factor=num_workers
        if not dataset.cached_data_mode and num_workers > 0
        else 2,
        persistent_workers=not dataset.cached_data_mode and num_workers > 0,
    )
