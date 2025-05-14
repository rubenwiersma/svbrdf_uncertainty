import sys
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

import drjit as dr
import gin
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from pyexr import read, write
from skimage.transform import resize
import torch
from tqdm import tqdm

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
device = 'cuda' if 'cuda' in mi.variant() else 'cpu'

from svbrdf_uncertainty.plugins.integrators import CustomRayIntegrator
from svbrdf_uncertainty.util import render_texture, map_world_to_local, fps_mask
from svbrdf_uncertainty.util.render_ray import sample_rays_multiple_sensors, render_ray, integrate_ray_samples
from svbrdf_uncertainty.util.color import to_log_dr, linear_to_gamma_dr
from svbrdf_uncertainty.util.nvdiffrast import read_mesh, get_mvp
from svbrdf_uncertainty.loss import total_variation_loss_dr, compute_global_sharing_inputs, global_sharing_dr
from svbrdf_uncertainty.spherical_harmonics import gaussian_filter_spherical, lsq_spherical_harmonics, power_spectrum
from svbrdf_uncertainty.uncertainty import best_view_selection, specular_entropy

# Register custom integrator that takes rays as input
mi.register_integrator('customray', lambda props: CustomRayIntegrator(props))

# Get folder of current file to easily refer to relative paths
current_path = Path(__file__).parent.absolute()


@gin.configurable
def optimize_material(label, dataset, keys, texture_res=512, experiment_name=None,                   # Dataset, general settings
                      epoch_count=10, learning_rate=0.01, images_per_batch=1, early_stopping=0,   # Optimization settings
                      shuffle_rays=True, grad_spp=32, max_depth=-1,                                  # Rendering settings
                      smoothness_weight=1e-3, global_sharing_weight=0, smoothness_k=20, smoothness_kernel_width=0.4,          # Regularization
                      init_textures_path=None,                                                       # Application: initialization
                      dropout_views=0, best_view=False,
                      vis_sensor_count=4, vis_bake_textures=True, verbose=False): # Output settings
    # Bookkeeping
    # ---

    # Create a time and date stamp for the current run
    now = datetime.now()
    experiment_name = now.strftime('%Y-%m-%d_%H-%M-%S') if experiment_name is None else experiment_name

    # Output folder
    out_path = current_path / 'out' / label / 'mitsuba' / experiment_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Data loading and initialization
    # ---

    # Load dataset
    if callable(dataset):
        dataset = dataset()
    sensors = dataset.sensors
    frames = dataset.frames
    scene = dataset.scene
    sensor_count = len(dataset)

    # Compute entropy inputs for applications if necessary
    if (best_view and dropout_views > 0) or smoothness_k > 0:
        lsq_args = {
            'sample_margin': 1.0,
            'sample_weight_exponent': 1,
            'sh_regularizer_func': 'exp',
            'sh_regularizer': 1e-2
        }

        bandlimit_L = int(np.sqrt(sensor_count))
        envmap = read(dataset.envmap_path)[..., :3]
        envmap_transform = None
        if hasattr(dataset, 'envmap_transform'):
            envmap_transform = dataset.envmap_transform.matrix.torch().to(device)
        # Resize envmap if necessary
        if envmap.shape[0] > 256:
            envmap = resize(envmap, (256, 512))

        envmap_torch = torch.from_numpy(envmap).to(device) # [H, W, 3]
        # Bandlimit the incoming radiance to ensure
        # that the sampling rate is high enough to estimate SH coefficients
        envmap_smoothed = gaussian_filter_spherical(envmap_torch, 1 / bandlimit_L)

        mvp, camera_pos = [], []
        for sensor in sensors:
            mvp_sensor, camera_pos_sensor = get_mvp(sensor, return_camera_pos=True)
            mvp.append(mvp_sensor)
            camera_pos.append(camera_pos_sensor)
        camera_pos = torch.stack(camera_pos, dim=0)

        reflected_radiance, exitant_radiance, theta_local, phi_local, _, _, _, _ = \
        map_world_to_local(dataset, mvp, camera_pos, envmap_smoothed, dataset.mesh, envmap_transform, texture_res)

    # Select cameras to drop
    if dropout_views > 0:
        if best_view:
            # dropout_mask = fps_mask(camera_pos, 10)
            lmax = 5
            dropout_mask = torch.zeros(camera_pos.shape[0], dtype=torch.bool, device=device)
            dropout_mask = best_view_selection(sensor_count - dropout_views, lmax,
                                               dropout_mask,
                                               reflected_radiance, exitant_radiance, theta_local, phi_local, lsq_args,
                                               8, 8, 1e-4)
        else:
            dropout_mask = fps_mask(camera_pos, sensor_count - dropout_views)
        sensors = [sensors[i] for i in range(len(sensors)) if dropout_mask[i]]
        sensor_count = len(sensors)
        frames = frames[dropout_mask]

    # Precompute knn for TV-norm regularization
    with torch.no_grad():
        if global_sharing_weight > 0 and smoothness_k > 0:
            entropy_sigma = 1e-5
            entropy_resolution = 8
            lmax = 5

            theta_weight = 1 - torch.pow(1 - torch.cos(theta_local).clamp(0, 1), 1.0) # [texture_res, texture_res, sensor_count]
            exitant_radiance_mask = exitant_radiance.sum(dim=-1) > 1e-3
            sample_weight = theta_weight * exitant_radiance_mask
            reflected_sh_coeffs = lsq_spherical_harmonics( # [texture_res, texture_res, (lmax + 1)^2, 3]
                reflected_radiance, theta_local, phi_local,
                lmax=lmax, weight=sample_weight, regularizer_func=torch.exp, regularizer=1e-2)
            exitant_sh_coeffs = lsq_spherical_harmonics( # [texture_res, texture_res, (lmax + 1)^2, 3]
                exitant_radiance, theta_local, phi_local,
                lmax=lmax, weight=sample_weight, regularizer_func=torch.exp, regularizer=1e-2)
            reflected_spectrum = power_spectrum(reflected_sh_coeffs, unit='per_lm') # [texture_res, texture_res, lmax + 1, 3]
            exitant_spectrum = power_spectrum(exitant_sh_coeffs, unit='per_lm') # [texture_res, texture_res, lmax + 1, 3]

            entropy, min_alpha, min_specular, param_probabilities = specular_entropy(reflected_spectrum, exitant_spectrum, sigma_probability=entropy_sigma, alpha_resolution=entropy_resolution, specular_resolution=entropy_resolution, return_p=True)
            kl_divergence_weights, local_min_entropy_idx = compute_global_sharing_inputs(param_probabilities, entropy, k=smoothness_k, kl_divergence_kernel_width=smoothness_kernel_width)

    # Extract available parameters
    params = mi.traverse(scene)
    params_true = {}
    for key in keys:
        params_true[key] = params[key].numpy()

    # Initialize parameters
    params_shape = [params[key].shape for key in keys]
    params_initial = [dr.full(mi.TensorXf, 0.5, shape=(texture_res, texture_res) + param_shape[2:]) for param_shape in params_shape]

    # Set textures to new values
    for key, p in zip(keys, params_initial):
        if 'roughness' in key or 'metallic' in key:
            p = p[..., 0, None]
        params[key] = p
    params.update()

    if init_textures_path is not None:
        base_color_init = read(str(init_textures_path / 'base_color.exr'))[..., :3]
        roughness_init = read(str(init_textures_path / 'roughness.exr'))
        metallic_init = read(str(init_textures_path / 'metallic.exr'))
        if roughness_init.ndim > 2: roughness_init = roughness_init[..., 0, None]
        if metallic_init.ndim > 2: metallic_init = metallic_init[..., 0, None]
        params['shape.bsdf.base_color.data'] = mi.TensorXf(base_color_init)
        params['shape.bsdf.roughness.data'] = mi.TensorXf(roughness_init)
        params['shape.bsdf.metallic.data'] = mi.TensorXf(metallic_init)
        params.update()

    # Optimization
    # ---

    start_t = perf_counter()

    # Concatenate all images into a single tensor
    # denoted y, as the target variable
    # Note: frames stores the frames as [N, H, W, C], where C is in RGB order.
    y = frames.reshape(-1, 3)

    opt = mi.ad.Adam(lr=learning_rate)
    for key in keys:
        opt[key] = params[key]
    params.update(opt)

    integrator = mi.load_dict({
        'type': 'customray',
        'integrator': {
            'type': 'prb',
            'max_depth': max_depth
        }
    }, parallel=False)

    previous_loss = float('inf')
    for epoch in tqdm(range(epoch_count), disable=not verbose):
        total_loss = 0.0

        # Sample rays
        # We keep the origins (o), directions(d), and wavelengths associated with each ray
        o, d, wavelengths = sample_rays_multiple_sensors(integrator, scene, sensors, seed=epoch)

        # Shuffle indices of rays
        indices = torch.randperm(y.shape[0]) if shuffle_rays else torch.arange(y.shape[0])

        # Create batches of rays from indices
        batch_size = dr.prod(sensors[0].film().size()) * images_per_batch
        max_it = y.shape[0] // batch_size
        batches = indices[:max_it * batch_size].reshape(-1, batch_size)

        for batch in batches:
            # Move target and rays to device (should be cuda)
            target = dr.cuda.Array3f(y[batch].to(device))
            # Construct ray object containing random rays
            # Repeat the same primal ray grad_spp times to get better convergence
            ray = mi.RayDifferential3f(
                o=o[batch].to(device).repeat_interleave(grad_spp, dim=0),
                d=d[batch].to(device).repeat_interleave(grad_spp, dim=0),
                wavelengths=wavelengths
            )
            batch = batch.to(device)

            # Render image
            L = render_ray(scene, params, integrator=integrator, ray=ray,
                           spp=grad_spp, seed=epoch)

            # Integrate samples per pixel
            L_integrated = integrate_ray_samples(L, grad_spp)

            # Convert to log space
            if hasattr(dataset, 'hdr') and not dataset.hdr or not hasattr(dataset, 'hdr'):
                L_integrated = dr.clamp(L_integrated, 0.0, 1.0)
            L_integrated = linear_to_gamma_dr(to_log_dr(L_integrated * (1 << 16)))
            target = linear_to_gamma_dr(to_log_dr(target * (1 << 16)))

            loss = dr.mean(dr.abs(dr.ravel(L_integrated - target)))
            # regularizer_loss = sum([smoothness_loss_dr(params[key]) for key in keys])
            if global_sharing_weight > 0:
                global_sharing_loss = global_sharing_weight * sum([dr.mean(global_sharing_dr(params[key], local_min_entropy_idx, kl_divergence_weights)) for key in keys])
                loss = loss + global_sharing_loss
            if smoothness_weight > 0:
                smoothness_loss = smoothness_weight * sum([dr.mean(total_variation_loss_dr(params[key])) for key in keys])
                loss = loss + smoothness_loss

            # Apply one gradient descent step
            dr.backward(loss)
            opt.step()
            # Make sure values don't exceed allowed range
            for key in keys:
                if not 'envmap' in key:
                    opt[key] = dr.clamp(opt[key], 0.0, 1.0)
            # Update parameters
            params.update(opt)
            # Record loss
            total_loss += loss[0]

        if abs(total_loss - previous_loss) < early_stopping and epoch > 3: break
        previous_loss = total_loss

    timing = perf_counter() - start_t

    write_textures(params, keys, out_path)
    # if experiment_name is None:
    vis_result(scene, params, keys, dataset, vis_sensor_count, vis_bake_textures, out_path, params_true)

    if verbose:
        print(f'\nOptimization complete. Time: {timing:.2f}s. Results written in {out_path}')
    return timing


def write_textures(params, keys, out_path):
    """Store optimized textures in output folder.
    """
    for i, key in enumerate(keys):
        label = key.split('.')[-2]
        # Store optimized textures in output folder
        write(str(out_path / (label + '.exr')), params[key].numpy(), ['Y'] if params[key].shape[-1] == 1 else ['R', 'G', 'B'])


def vis_result(scene, params, keys, dataset, sensor_count, bake_textures, out_path, params_true=None):
    """Store optimized textures and visualize resulting renders.
    A new envmap can be chose to simulate relighting.

    Args:
        scene (mi.Scene): The scene to render.
        params (mi.SceneParameters): Parameters that were optimized.
        keys (list): List of keys that were optimized.
        dataset (torch...Dataset): The dataset that was used during optimization.
        sensor_count (int): Number of sensors to visualize.
            An equally spaced set of sensors will be picked.
        out_path (pathlib.Path): The path to store the results.
        params_true (mi.SceneParameters): Ground truth parameters of the dataset.
    """
    # Visualize rerendered results
    sensors = dataset.sensors
    num_params = len(keys)
    mesh_pos, mesh_faces, mesh_uv, _ = read_mesh(dataset.mesh)
    mesh_attributes = (mesh_pos, mesh_faces, mesh_uv)

    fig, ax = plt.subplots(2 + bake_textures * num_params, num_params + sensor_count, figsize=(6 * (num_params + sensor_count), 6 * (2 + bake_textures * num_params)))
    fig.suptitle('Mitsuba')

    for i in range(2 + bake_textures * num_params):
        for j in range(num_params + sensor_count):
            ax[i, j].axis('off')

    for i, key in enumerate(keys):
        label = key.split('.')[-2]
        ax[0, i].imshow(params[key], vmin=0, vmax=1, cmap='gray')
        ax[0, i].set_title(f'{label} pred')

        if params_true is not None:
            ax[1, i].imshow(params_true[key], vmin=0, vmax=1, cmap='gray')
            ax[1, i].set_title(f'{label} gt')

        if bake_textures:
            for j in range(sensor_count):
                # Bake textures
                sensor_idx = len(dataset.sensors) // sensor_count * j
                ax[2 + i, num_params + j].imshow(render_texture(params[key].torch(), *mesh_attributes, sensors[sensor_idx]).cpu(), cmap='gray')
                ax[2 + i, num_params + j].set_title(f'{label} pred')

    for i in range(sensor_count):
        sensor_idx = len(dataset.sensors) // sensor_count * i
        new_render = mi.render(scene, sensor=sensors[sensor_idx], spp=512)
        ax[0, num_params + i].imshow(mi.util.convert_to_bitmap(new_render))
        ax[0, num_params + i].set_title('Re-render')
        ax[1, num_params + i].imshow(dataset[sensor_idx].cpu().clip(0, 1) ** (1 / 2.2))
        ax[1, num_params + i].set_title('Original')

    plt.savefig(out_path / 'brdf_overview.png', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = os.path.join(sys.argv[1], 'opt_mitsuba.gin') if os.path.isdir(sys.argv[1]) else sys.argv[1]
    else:
        config_file = 'experiments/scenes/plane/opt_mitsuba.gin'
    gin.add_config_file_search_path(os.path.dirname(config_file))
    gin.parse_config_file(config_file)
    optimize_material()
