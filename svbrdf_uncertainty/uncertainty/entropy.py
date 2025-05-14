from math import log
import torch


def specular_entropy(incoming_spectrum, exitant_spectrum, sigma_probability=0.1, alpha_resolution=10, specular_resolution=10, min_channel=True, return_p=False):
    """Computes entropy of the distribution of probable parameters
    that would fit the specular component of a BRDF reflection.

    Args:
        incoming_spectrum (torch.Tensor): Spectrum of the incoming radiance.
            Tensor of size [H, W, L, C], where H, W are the height and width of the BRDF texture,
            L is the number of degrees and C is the number of channels in the outgoing light.
        exitant_spectrum (torch.Tensor): Spectrum of the exitant radiance. Same size as incoming_spectrum.
        sigma_probability (float): Width of the Gaussian kernel used to compute probability scores
            for each possible parameter combination.
        grid_resolution (int): Number of options to try for each parameter.
    """
    p, min_alpha, min_specular = brdf_probabilities(incoming_spectrum, exitant_spectrum, sigma_probability, alpha_resolution, specular_resolution)
    entropy = -(p * torch.log(p.clip(1e-12))).sum(dim=(-2, -1)) / log(alpha_resolution * specular_resolution)

    min_alpha = torch.gather(min_alpha, -1, torch.argmin(entropy, dim=-1, keepdim=True)).squeeze()
    if min_channel:
        entropy = entropy.min(dim=-1)[0]

    if return_p:
        return entropy, min_alpha, min_specular, p
    return entropy, min_alpha, min_specular


def brdf_probabilities(incoming_spectrum, exitant_spectrum, sigma_probability, alpha_resolution, specular_resolution):
    """Computes the likelihood of each parameter combination
    based on a simplified BRDF model in the Spherical Harmonics power spectrum."""

    device = incoming_spectrum.device

    # Setup grid of possible parameter combinations
    alpha_grid = torch.square(torch.linspace(0, 1, alpha_resolution, device=device).view(*([1] * incoming_spectrum.dim()), -1, 1))
    specular_grid = torch.linspace(0, 1, specular_resolution, device=device).view(*([1] * incoming_spectrum.dim()), 1, -1)

    # Compute simplified BRDF model on power spectrum
    l = torch.arange(1, incoming_spectrum.shape[-2], device=device).view(*([1] * (incoming_spectrum.dim() - 2)), -1, 1, 1, 1)
    brdf_spectrum = torch.square(specular_grid * torch.exp(-torch.square(alpha_grid * l)))
    exitant_spectrum_pred = brdf_spectrum * incoming_spectrum[..., 1:, :, None, None]

    # MSE over degrees
    loss_grid = torch.mean(torch.square(exitant_spectrum_pred - exitant_spectrum[..., 1:, :, None, None]), dim=(-4))

    # Normalize by exitant spectrum average energy
    loss_grid = loss_grid * exitant_spectrum[..., 0, :, None, None].square()

    # Find the minimum loss
    min_loss, min_idx = loss_grid.flatten(-2, -1).min(dim=-1)
    # And convert to associated parameter values
    min_alpha = alpha_grid.flatten()[min_idx // specular_resolution]
    min_specular = specular_grid.flatten()[min_idx % specular_resolution]

    # Convert loss to likelihood over parameter combinations
    likelihood = torch.exp(-(loss_grid - min_loss[..., None, None]) / sigma_probability)

    # Entropy
    p = likelihood / likelihood.sum(dim=(-2, -1), keepdims=True).clip(1e-12)
    return p, min_alpha, min_specular