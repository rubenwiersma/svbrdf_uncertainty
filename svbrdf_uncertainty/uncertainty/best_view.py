import torch

import time

from .entropy import specular_entropy, brdf_probabilities
from ..spherical_harmonics import power_spectrum
from ..spherical_harmonics.transforms import lsq_spherical_harmonics


def best_view_selection(n, lmax,
                        sensor_mask,
                        reflected_radiance, exitant_radiance, theta_local, phi_local, lsq_args,
                        alpha_resolution=8, specular_resolution=8, sigma_probability=1e-4):
    """Selects the best n views to minimize the expected entropy of the BRDF parameters.

    Args:
        n (int): Number of views to select.
        lmax (int): Maximum degree of the SH representation.
        sensor_mask (torch.Tensor): Binary mask of selected views to start with.
        reflected_radiance (torch.Tensor): Incoming radiance from the environment
            in reflection direction from exitant radiance.
        exitant_radiance (torch.Tensor): Exitant radiance from the material.
        theta_local (torch.Tensor): Local polar angles of the viewing positions.
        phi_local (torch.Tensor): Local azimuthal angles of the viewing positions.
        lsq_args (dict): Arguments for the least squares optimization.
        alpha_resolution (int): Number of options to try for the alpha parameter.
        specular_resolution (int): Number of options to try for the specular parameter.
        sigma_probability (float): Width of the Gaussian kernel used to compute probability scores
    """
    # Initialize sensor mask with evenly spaced points
    n_start = int(sensor_mask.sum())

    # Fit SH coefficients for the incoming light for these sensors
    theta_weight = 1 - torch.pow(1 - torch.cos(theta_local / lsq_args['sample_margin']).clamp(0, 1), lsq_args['sample_weight_exponent'])
    # Mask for outgoing samples with zero-radiance (occluded or unobserved points)
    exitant_radiance_mask = exitant_radiance.sum(dim=-1) > 1e-3
    theta_weight = theta_weight * exitant_radiance_mask

    s_total = time.perf_counter()

    # Initialize probability grid with uniform distribution
    param_probabilities = torch.ones(reflected_radiance.shape[:2] + (reflected_radiance.shape[-1], alpha_resolution, specular_resolution), device=reflected_radiance.device)
    param_probabilities = param_probabilities / (alpha_resolution * specular_resolution)
    # Add the next best view iteratively
    for i in range(n_start, n):
        s = time.perf_counter()

        # If there are enough points, inform the probabilities based on the current sensor mask
        if i >= 10:
            sample_weight = sensor_mask[None, None] * theta_weight
            regularizer_func = torch.exp if lsq_args['sh_regularizer_func'] == 'exp' else torch.ones_like
            reflected_sh_coeffs = lsq_spherical_harmonics(
                reflected_radiance, theta_local, phi_local,
                lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=1e-3)
            reflected_spectrum = power_spectrum(reflected_sh_coeffs, unit='per_lm')
            exitant_sh_coeffs = lsq_spherical_harmonics(
                exitant_radiance, theta_local, phi_local,
                lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=1e-3)
            exitant_spectrum = power_spectrum(exitant_sh_coeffs, unit='per_lm')

            # Calculate likelihood for grid
            # param_probabilities : [texture_res, texture_res, C, alpha_resolution, specular_resolution]
            param_probabilities, _, _ = brdf_probabilities(reflected_spectrum, exitant_spectrum,
                                                        alpha_resolution=alpha_resolution, specular_resolution=specular_resolution,
                                                        sigma_probability=sigma_probability)

        next_idx = next_best_view(sensor_mask, reflected_radiance, param_probabilities,
                                  lmax, theta_local, phi_local, theta_weight, lsq_args,
                                  alpha_resolution=alpha_resolution, specular_resolution=specular_resolution, sigma_probability=sigma_probability)
        sensor_mask[next_idx] = 1
        print(f"Selected view {next_idx} in {time.perf_counter() - s:.2f}s")

    print(f"Total time: {time.perf_counter() - s_total:.2f}s")

    return sensor_mask

def next_best_view(sensor_mask, reflected_radiance, param_probabilities,
                   lmax, theta_local, phi_local, theta_weight, lsq_args,
                   alpha_resolution=8, specular_resolution=8, sigma_probability=1e-4):
    """Selects the next best view to minimize the expected entropy of the BRDF parameters.

    Args:
        sensor_mask (torch.Tensor): Binary mask of selected views.
        reflected_radiance (torch.Tensor): Reflected radiance from the material.
        theta_local (torch.Tensor): Local polar angles of the viewing positions.
        phi_local (torch.Tensor): Local azimuthal angles of the viewing positions.
        lsq_args (dict): Arguments for the least squares optimization.
    """

    # Initialize expected entropy for each view
    expected_entropy_per_view = torch.zeros(sensor_mask.shape, device=sensor_mask.device)
    expected_entropy_per_view[sensor_mask] = float('inf')

    # Test every sensor that has not been selected yet
    for view_idx in torch.nonzero(sensor_mask == 0)[:, 0]:
        # Turn on the new sensor
        sensor_mask[view_idx] = 1

        # Fit SH coefficients for the incoming light for these sensors
        sample_weight = sensor_mask[None, None] * theta_weight
        regularizer_func = torch.exp if lsq_args['sh_regularizer_func'] == 'exp' else torch.ones_like
        reflected_sh_coeffs = lsq_spherical_harmonics(
            reflected_radiance, theta_local, phi_local,
            lmax=lmax, weight=sample_weight, regularizer_func=regularizer_func, regularizer=1e-3)
        reflected_spectrum = power_spectrum(reflected_sh_coeffs, unit='per_lm')

        expected_entropy = torch.zeros_like(param_probabilities[..., 0, 0])

        for i, alpha in enumerate(torch.linspace(0, 1, alpha_resolution)):
            for j, specular in enumerate(torch.linspace(0, 1, specular_resolution)):
                l = torch.arange(0, reflected_spectrum.shape[-2], device=reflected_spectrum.device).view(*([1] * (reflected_spectrum.dim() - 2)), -1)
                brdf_spectrum = torch.square(specular * torch.exp(-torch.square(alpha * l)))
                hypothetical_exitant_spectrum = reflected_spectrum * brdf_spectrum[..., None]
                hypothetical_entropy, _, _ = specular_entropy(reflected_spectrum, hypothetical_exitant_spectrum,
                                                              sigma_probability=sigma_probability, alpha_resolution=alpha_resolution, specular_resolution=specular_resolution,
                                                              min_channel=False)
                mask = hypothetical_entropy < 1.0
                expected_entropy += (param_probabilities[..., i, j] * mask * hypothetical_entropy)

        expected_entropy_per_view[view_idx] = expected_entropy.mean()

        # Turn off the new sensor
        sensor_mask[view_idx] = 0

    return torch.argmin(expected_entropy_per_view)
