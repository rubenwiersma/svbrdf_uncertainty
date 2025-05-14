from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time

import gin
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
from pyexr import read, write
from skimage.transform import resize
import torch
from tqdm import tqdm

# Set Mitsuba variant
mi.set_variant('cuda_ad_rgb')
device = 'cuda' if 'cuda' in mi.variant() else 'cpu'

from svbrdf_uncertainty.spherical_harmonics import gaussian_filter_spherical, lsq_spherical_harmonics, evaluate_spherical_harmonics, power_spectrum
from svbrdf_uncertainty.brdf import PrincipledSH, PrincipledSpectrum, PrincipledAngular, irradiance_per_point
from svbrdf_uncertainty.brdf.microfacet import trowbridge_reitz_masking, schlick_weight
from svbrdf_uncertainty.uncertainty import specular_entropy, best_view_selection
from svbrdf_uncertainty.loss import total_variation_loss, compute_global_sharing_inputs, global_sharing
from svbrdf_uncertainty.util import fps_mask
from svbrdf_uncertainty.util.mapping import map_world_to_local, render_texture, read_mesh, sample_envmap, get_mvp

# Get folder of current file to easily refer to relative paths
current_path = Path(__file__).parent.absolute()


@gin.configurable
def optimize_material(label, dataset, texture_res, experiment_name=None, verbose=False,                                        # Dataset, general settings
                      lmax=6, sh_regularizer=1e-3, sh_regularizer_func='exp', sample_margin=1.0, sample_weight_exponent=1,     # SH fitting settings
                      fresnel_enabled=True, shadowing_enabled=True, masking_enabled=True, update_masking_every=50,             # BRDF settings
                      roughness_admitted=(0, 1), metallic_admitted=(0, 1), base_color_admitted=(0, 1),                         # Parameter settings
                      loss_domain='angular', num_iterations=200, learning_rate=0.01, init_global=False, early_stopping=1e-7,   # Optimization settings
                      smoothness_weight=0, global_sharing_weight=0, smoothness_k=20, smoothness_kernel_width=0.4,              # Regularization
                      loss_entropy_weight=0, loss_entropy_exponent=1,                                                          # Loss settings
                      entropy_resolution=8, entropy_sigma=1e-5,                                                                # Entropy settings
                      vis_sensor_count=4, vis_bake_textures=True, vis_envmap_path=None, config_file=None,                      # Output settings
                      dropout_views=0, best_view=False):
    """Optimizes a material texture given a set of views
    with registered cameras and a UV-mapped mesh.
    """
    assert loss_domain in ['sh_coeffs', 'spectrum', 'angular'], "loss_domain must be one of 'sh_coeffs', 'spectrum', or 'angular'"

    # Bookkeeping
    # ---

    # Create a time and date stamp for the current run
    now = datetime.now()
    experiment_name = now.strftime('%Y-%m-%d_%H-%M-%S') if experiment_name is None else experiment_name

    # Output folder
    out_path = current_path / 'out' / label / 'sh' / experiment_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Data loading and initialization
    # ---

    timings = []
    if verbose: print('[Pre] Loading dataset', end='\r')
    t = time.perf_counter()
    # Load dataset
    if callable(dataset):
        dataset = dataset()
    sensors = dataset.sensors
    mesh = dataset.mesh
    sensor_count = len(dataset)
    bandlimit_L = int(np.sqrt(sensor_count))

    timings.append(time.perf_counter() - t)
    if verbose:
        print(f'[Pre] Loading dataset - Time: {timings[-1]:.2f}s')
        print('[Pre] Envmap filtering', end='\r')
    t = time.perf_counter()

    # Load environment map
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

    timings.append(time.perf_counter() - t)
    if verbose:
        print(f'[Pre] Envmap filtering - Time: {timings[-1]:.2f}s')
        print('[Pre] Mapping to local frame', end='\r')
    timings = []
    t = time.perf_counter()

    # Computing incoming and outgoing radiance
    # ---

    # Map observations to radiances in local directional space
    # reflected_radiance, exitant_radiance : [texture_res, texture_res, sensor_count, 3]
    # theta_local, phi_local               : [texture_res, texture_res, sensor_count]
    # normal, t_basis, s_basis             : [texture_res, texture_res, 3]
    mvp, camera_pos = [], []
    for sensor in sensors:
        mvp_sensor, camera_pos_sensor = get_mvp(sensor, return_camera_pos=True)
        mvp.append(mvp_sensor)
        camera_pos.append(camera_pos_sensor)
    camera_pos = torch.stack(camera_pos, dim=0)
    reflected_radiance, exitant_radiance, theta_local, phi_local, normal, t_basis, s_basis, pos = \
        map_world_to_local(dataset, mvp, camera_pos, envmap_smoothed, mesh, envmap_transform, texture_res)
    # Sample incoming radiance on Fibonacci lattice for better recovery
    # incoming_radiance                    : [1, 1, (2lmax)^2, 3]
    # theta_incoming, phi_incoming         : [1, 1, (2lmax)^2]
    incoming_radiance, theta_incoming, phi_incoming = sample_envmap((lmax * 2)**2, envmap_smoothed, normal, t_basis, s_basis, envmap_transform)

    timings.append(time.perf_counter() - t)
    if verbose:
        print(f'[Pre] Mapping to local frame - Time: {timings[-1]:.2f}s')
        print('[Opt] Setup', end='\r')
    t = time.perf_counter()

    # Precompute weights based on cos(theta)
    # ---
    clamped_cos = torch.cos(theta_local).clamp(0, 1)

    # Schlick weight for Fresnel term
    schlick = schlick_weight(clamped_cos)
    # Weight for optimization
    theta_weight = 1 - torch.pow(1 - torch.cos(theta_local / sample_margin).clamp(0, 1), sample_weight_exponent) # [texture_res, texture_res, sensor_count]
    incoming_weight = 1 - torch.pow(1 - torch.cos(theta_incoming / sample_margin).clamp(0, 1), sample_weight_exponent) # [texture_res, texture_res, (2lmax)^2]
    if sample_weight_exponent == -1:
        theta_weight = torch.ones_like(theta_weight)
        incoming_weight = torch.ones_like(incoming_weight)
    # Mask for outgoing samples with zero-radiance (occluded or unobserved points)
    exitant_radiance_mask = exitant_radiance.sum(dim=-1) > 1e-3
    if dropout_views > 0:
        if best_view:
            lsq_args = {
                'sample_margin': sample_margin,
                'sample_weight_exponent': sample_weight_exponent,
                'sh_regularizer_func': sh_regularizer_func,
                'sh_regularizer': sh_regularizer
            }
            dropout_mask = fps_mask(camera_pos, 10)
            # dropout_mask = torch.zeros(camera_pos.shape[0], dtype=torch.bool, device=device)
            dropout_mask = best_view_selection(sensor_count - dropout_views, lmax,
                                               dropout_mask,
                                               reflected_radiance, exitant_radiance, theta_local, phi_local, lsq_args,
                                               entropy_resolution, entropy_resolution, entropy_sigma)
        else:
            dropout_mask = fps_mask(camera_pos, sensor_count - dropout_views)
        exitant_radiance_mask = dropout_mask[None, None] * exitant_radiance_mask
    sample_weight = theta_weight * exitant_radiance_mask

    timings.append(time.perf_counter() - t)
    if verbose:
        print(f'[Opt] Setup - Time: {timings[-1]:.2f}s')
        print('[Pre] Fitting SH', end='\r')
    t = time.perf_counter()

    # Fitting spherical harmonics
    # ---

    # Compute integral of incoming radiance for diffuse component
    # envmap_irradiance = torch.from_numpy(resize(envmap, (32, 64))).float().to(normal.device)
    envmap_irradiance = torch.from_numpy(resize(envmap, (16, 32))).float().to(normal.device)
    irradiance = irradiance_per_point(envmap_irradiance, normal, envmap_transform)[:, :, None]

    # Estimate spherical harmonic coefficients for incoming and outgoing light
    regularizer_func = torch.exp if sh_regularizer_func == 'exp' else torch.ones_like
    incoming_sh_coeffs = lsq_spherical_harmonics( # [texture_res, texture_res, (lmax + 1)^2, 3]
        incoming_radiance, theta_incoming, phi_incoming,
        lmax=lmax, weight=incoming_weight, regularizer_func=regularizer_func, regularizer=sh_regularizer)
    reflected_sh_coeffs = lsq_spherical_harmonics( # [texture_res, texture_res, (lmax + 1)^2, 3]
        reflected_radiance, theta_local, phi_local,
        lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=1e-3)
    exitant_sh_coeffs = lsq_spherical_harmonics( # [texture_res, texture_res, (lmax + 1)^2, 3]
        exitant_radiance, theta_local, phi_local,
        lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=1e-3)
    reflected_spectrum = power_spectrum(reflected_sh_coeffs, unit='per_lm') # [texture_res, texture_res, lmax + 1, 3]
    exitant_spectrum = power_spectrum(exitant_sh_coeffs, unit='per_lm') # [texture_res, texture_res, lmax + 1, 3]

    timings.append(time.perf_counter() - t)
    if verbose:
        print(f'[Pre] Fitting SH - Time: {timings[-1]:.2f}s')
        print('-----------------')
        print('[Opt] Entropy...', end='\r')
    t = time.perf_counter()

    # Optimization and uncertainty
    # ---

    # Compute uncertainty
    entropy, min_alpha, min_specular, param_probabilities = specular_entropy(reflected_spectrum, exitant_spectrum, sigma_probability=entropy_sigma, alpha_resolution=entropy_resolution, specular_resolution=entropy_resolution, return_p=True)

    timings.append(time.perf_counter() - t)
    if verbose: print(f'[Opt] Entropy - Time: {timings[-1]:.4f}s')

    # Set up the Principled BRDF model that either outputs SH coefficients, an SH spectrum or angular samples
    if loss_domain == 'sh_coeffs':
        model = PrincipledSH(texture_res, roughness_admitted, metallic_admitted, base_color_admitted).to(device)
    elif loss_domain == 'spectrum':
        model = PrincipledSpectrum(texture_res, roughness_admitted, metallic_admitted, base_color_admitted).to(device)
    elif loss_domain == 'angular':
        target = sample_weight[..., None] * exitant_radiance
        SH_basis = evaluate_spherical_harmonics(lmax, theta_local, phi_local)
        model = PrincipledAngular(texture_res, SH_basis, roughness_admitted, metallic_admitted, base_color_admitted, fresnel_enabled, masking_enabled).to(device)

    # Initialize parameters with optimal values from grid-search
    if init_global:
        roughness_init = torch.sqrt(min_alpha).clamp(0, 1)
        base_color_init = (exitant_sh_coeffs[:, :, 0] / incoming_sh_coeffs[:, :, 0].clip(1e-8)).clamp(0, 1)
        metallic_init = (min_specular / base_color_init.clip(1e-8)).mean(dim=-1).clamp(0, 1)
        model.roughness = torch.nn.Parameter(roughness_init)
        model.base_color = torch.nn.Parameter(base_color_init)
        model.metallic = torch.nn.Parameter(metallic_init)

    # Precompute knn for TV-norm regularization
    if global_sharing_weight > 0 and smoothness_k > 0:
        kl_divergence_weights, local_min_entropy_idx = compute_global_sharing_inputs(param_probabilities, entropy, k=smoothness_k, kl_divergence_kernel_width=smoothness_kernel_width)

    if num_iterations > 1:
        if verbose: print('[Opt] Optimizing...')
        t = time.perf_counter()
        # Setup the optimizer
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        previous_loss = float('inf')
        for it in tqdm(range(num_iterations), disable=not verbose):
            with torch.no_grad():
                if (shadowing_enabled or masking_enabled) and (it + (not init_global)) % update_masking_every == 0:
                    alpha = torch.square(model.roughness.detach()).clamp(0.001)
                    if shadowing_enabled:
                        if loss_domain == 'angular':
                            G_i = trowbridge_reitz_masking(theta_incoming, alpha)[..., None]
                            incoming_sh_coeffs = lsq_spherical_harmonics(
                                G_i * incoming_radiance, theta_incoming, phi_incoming,
                                lmax=lmax, weight=incoming_weight, regularizer_func=regularizer_func, regularizer=sh_regularizer)
                        else:
                            G_i = trowbridge_reitz_masking(theta_local, alpha)[..., None]
                            reflected_sh_coeffs = lsq_spherical_harmonics(
                                G_i * reflected_radiance, theta_local, phi_local,
                                lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=sh_regularizer)
                            reflected_spectrum = power_spectrum(incoming_sh_coeffs, unit='per_lm')
                    if masking_enabled and loss_domain in ['sh_coeffs', 'spectrum']:
                        G_o = trowbridge_reitz_masking(theta_local, alpha)[..., None]
                        exitant_sh_coeffs = lsq_spherical_harmonics(
                            exitant_radiance / G_o, theta_local, phi_local,
                            lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=sh_regularizer)
                        exitant_spectrum = power_spectrum(exitant_sh_coeffs, unit='per_lm')

            prefilter_alpha = 1 / bandlimit_L

            # Entropy weighting for loss to enforce sharing from accurate results
            reconstruction_weight = torch.ones_like(entropy)
            if loss_entropy_weight > 0 and it > num_iterations // 2:
                reconstruction_weight = loss_entropy_weight * torch.pow(1 - entropy, loss_entropy_exponent) + (1 - loss_entropy_weight) * reconstruction_weight
            reconstruction_weight = reconstruction_weight[..., None, None]

            if loss_domain == 'sh_coeffs':
                coeffs_pred = model(reflected_sh_coeffs, irradiance, prefilter_alpha)
                loss = criterion(reconstruction_weight * coeffs_pred, reconstruction_weight * exitant_sh_coeffs)
            elif loss_domain == 'spectrum':
                spectrum_pred = model(reflected_spectrum, irradiance, prefilter_alpha)
                loss = criterion(reconstruction_weight * spectrum_pred, reconstruction_weight * exitant_spectrum)
            elif loss_domain == 'angular':
                y_pred = model(incoming_sh_coeffs, irradiance, schlick, theta_local, prefilter_alpha)
                loss = criterion(reconstruction_weight * sample_weight[..., None] * y_pred, reconstruction_weight * target)

            if smoothness_weight > 0:
                mask = entropy < 1
                smoothness_term = (
                    total_variation_loss(model.roughness, mask=mask) +
                    total_variation_loss(model.metallic, mask=mask) +
                    total_variation_loss(model.base_color, mask=mask)
                )
                smoothness_term = smoothness_weight * (smoothness_term * mask).mean()
                loss = loss + smoothness_term

            if global_sharing_weight > 0:
                global_sharing_term = (
                    global_sharing(model.roughness, local_min_entropy_idx, kl_divergence_weights) +
                    global_sharing(model.metallic, local_min_entropy_idx, kl_divergence_weights) +
                    global_sharing(model.base_color, local_min_entropy_idx, kl_divergence_weights)
                )
                global_sharing_term = global_sharing_weight * (global_sharing_term *  mask.flatten()).mean()
                loss = loss + global_sharing_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.clamp_parameters()

            if abs(loss.item() - previous_loss) < early_stopping and it > update_masking_every: break
            previous_loss = loss.item()


        timings.append(time.perf_counter() - t)
        if verbose: print(f'[Opt] Optimizing - Loss: {loss.item():.4f}, Time: {timings[-1]:.2f}s')

    if verbose: print(f'Total time: {sum(timings):.2f}s')

    if config_file is not None:
        shutil.copyfile(config_file, out_path / 'config.gin')
    write_textures(model, entropy, out_path)
    # if experiment_name is None:
    vis_result(model, dataset, entropy, vis_envmap_path, vis_sensor_count, vis_bake_textures, out_path)
    if verbose: print(f'Done! Results written in {out_path}')
    torch.cuda.empty_cache()
    return sum(timings)


def write_textures(brdf, entropy, out_path):
    """Store optimized textures in output folder.
    """
    roughness_pred = brdf.roughness.detach().cpu().numpy()
    metallic_pred = brdf.metallic.detach().cpu().numpy()
    base_color_pred = brdf.base_color.detach().cpu().numpy()
    write(str(out_path / 'roughness.exr'), roughness_pred, ['Y'])
    write(str(out_path / 'metallic.exr'), metallic_pred, ['Y'])
    write(str(out_path / 'base_color.exr'), base_color_pred, ['R', 'G', 'B'])
    write(str(out_path / 'entropy.exr'), entropy.cpu().numpy(), ['Y'])


def vis_result(brdf, dataset, entropy, envmap_path, sensor_count, bake_textures, out_path):
    """Store optimized textures and visualize resulting renders.
    A new envmap can be chose to simulate relighting.

    Args:
        brdf (SHPrincipledBRDF): The brdf that was optimized.
        dataset (torch...Dataset): The dataset that was used during optimization.
        entropy (torch.Tensor): Entropy for the given shape.
        envmap_path (string): Path to an environment map to use during rendering.
        sensor_count (int): Number of sensors to visualize.
            An equally spaced set of sensors will be picked.
        out_path (pathlib.Path): The path to store the results.
    """
    roughness_pred = brdf.roughness.detach().cpu().numpy()
    metallic_pred = brdf.metallic.detach().cpu().numpy()
    base_color_pred = brdf.base_color.detach().cpu().numpy()

    # Visualize rerendered results
    material_optimized = brdf.mitsuba_material
    sensors = dataset.sensors
    envmap_path = dataset.envmap_path if envmap_path is None else envmap_path
    rerender_scene = dataset.get_scene(dataset.mesh, material_optimized, envmap_path)

    n_rows = 2 + bake_textures * 4
    fig, ax = plt.subplots(n_rows, 4 + sensor_count, figsize=(6 * (4 + sensor_count), 6 * n_rows))
    fig.suptitle('Spherical Harmonics')

    for i in range(n_rows):
        for j in range(4 + sensor_count):
            ax[i, j].axis('off')

    ax[0, 0].imshow(base_color_pred, vmin=0, vmax=1)
    ax[0, 0].set_title('Base color pred')
    ax[0, 1].imshow(roughness_pred, vmin=0, vmax=1, cmap='gray')
    ax[0, 1].set_title('Roughness pred')
    ax[0, 2].imshow(metallic_pred, vmin=0, vmax=1, cmap='gray')
    ax[0, 2].set_title('Metallic pred')
    ax[0, 3].imshow(entropy.cpu(), cmap='turbo', vmin=0, vmax=1)
    ax[0, 3].set_title('Entropy')

    if hasattr(dataset, 'base_color'):
        ax[1, 0].imshow(dataset.base_color, vmin=0, vmax=1)
        ax[1, 0].set_title('Base color gt')
    if hasattr(dataset, 'roughness'):
        ax[1, 1].imshow(dataset.roughness, vmin=0, vmax=1, cmap='gray')
        ax[1, 1].set_title('Roughness gt')
    if hasattr(dataset, 'metallic'):
        ax[1, 2].imshow(dataset.metallic, vmin=0, vmax=1, cmap='gray')
        ax[1, 2].set_title('Metallic gt')

    for i in range(sensor_count):
        sensor_idx = len(dataset.sensors) // sensor_count * i
        new_render = mi.render(rerender_scene, sensor=sensors[sensor_idx], spp=64)
        ax[0, i + 4].imshow(mi.util.convert_to_bitmap(new_render))
        ax[0, i + 4].set_title('Re-render')
        ax[1, i + 4].imshow(dataset[sensor_idx].cpu().clip(0, 1) ** (1 / 2.2))
        ax[1, i + 4].set_title('Original')

        if bake_textures:
            mesh_pos, mesh_faces, mesh_uv, _ = read_mesh(dataset.mesh)
            mesh_attributes = (mesh_pos, mesh_faces, mesh_uv)
            # Bake textures
            ax[2, i + 4].imshow(render_texture(brdf.base_color.detach(), *mesh_attributes, sensors[sensor_idx]).cpu())
            ax[2, i + 4].set_title('Base color pred')
            ax[3, i + 4].imshow(render_texture(brdf.roughness.detach()[..., None], *mesh_attributes, sensors[sensor_idx]).cpu(), cmap='gray')
            ax[3, i + 4].set_title('Roughness pred')
            ax[4, i + 4].imshow(render_texture(brdf.metallic.detach()[..., None], *mesh_attributes, sensors[sensor_idx]).cpu(), cmap='gray')
            ax[4, i + 4].set_title('Metallic pred')
            ax[5, i + 4].imshow(render_texture(entropy[..., None], *mesh_attributes, sensors[sensor_idx]).cpu(), cmap='turbo', vmin=0, vmax=1)
            ax[5, i + 4].set_title('Entropy')

    plt.savefig(out_path / 'brdf_overview.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = os.path.join(sys.argv[1], 'opt_sh.gin') if os.path.isdir(sys.argv[1]) else sys.argv[1]
    else:
        config_file = 'experiments/scenes/plane/opt_sh.gin'
    gin.add_config_file_search_path(os.path.dirname(config_file))
    gin.parse_config_file(config_file)
    optimize_material(config_file=config_file)