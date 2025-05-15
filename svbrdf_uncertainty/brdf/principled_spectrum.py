from math import sqrt, pi

import mitsuba as mi
import torch


class PrincipledSpectrum(torch.nn.Module):
    """Implementation of the principled BRDF in the Spherical Harmonics (SH) power spectrum.

    This torch module stores the parameters of the BRDF (base color, roughness, metallic).
    In a forward pass, it computes the SH power spectrum for reflected light,
    given the SH power spectrum of the incoming light and irradiance.

    This model is faster than the other variants computed in the angular domain and SH domain,
    but less accurate, especially for grazing viewing angles,
    due to the exclusion of shadowing and masking and Fresnel.

    Args:
        texture_res (int): Resolution of the material textures.
        roughness_admitted (tuple): Admitted range for roughness.
        metallic_admitted (tuple): Admitted range for metallic.
        base_color_admitted (tuple): Admitted range for base color.
    """
    def __init__(self, texture_res, roughness_admitted=(0, 1), metallic_admitted=(0, 1), base_color_admitted=(0, 1)):
        super(PrincipledSpectrum, self).__init__()
        self.roughness = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res), requires_grad=True))
        self.metallic = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res), requires_grad=True))
        self.base_color = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res, 3), requires_grad=True))

        self.roughness_admitted = roughness_admitted
        self.metallic_admitted = metallic_admitted
        self.base_color_admitted = base_color_admitted

    def clamp_parameters(self):
        self.roughness.data = self.roughness.data.abs().clamp(*self.roughness_admitted)
        self.base_color.data = self.base_color.data.clamp(*self.base_color_admitted)
        self.metallic.data = self.metallic.data.clamp(*self.metallic_admitted)

    def forward(self, incoming_spectrum, irradiance, prefilter_alpha=0):
        return principled_brdf_spectrum(incoming_spectrum, irradiance, self.roughness, self.metallic, self.base_color, prefilter_alpha)

    @property
    def mitsuba_material(self):
        return {
            'type': 'principled',
            'eta': 1.5,
            'metallic': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.metallic.detach().cpu())
            },
            'base_color': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.base_color.detach().cpu())
            },
            'roughness': {
                'type': 'bitmap',
                'bitmap': mi.util.convert_to_bitmap(self.roughness.detach().cpu())
            }
        }


def principled_brdf_spectrum(incoming_spectrum, irradiance, roughness, metallic, base_color, prefilter_alpha=0):
    assert isinstance(roughness, torch.Tensor)
    assert isinstance(metallic, torch.Tensor)
    assert isinstance(base_color, torch.Tensor)
    assert base_color.dim() == 3 and base_color.shape[-1] == 3, "Base color should be given as RGB"

    # Specular
    lmax = incoming_spectrum.shape[-2]
    l = torch.arange(lmax, device=roughness.device)
    # Based on the Disney course notes and Mitsuba's implementation, we map alpha = roughness^2
    alpha = torch.square(roughness[..., None]).clamp(0.001)
    microfacet_spectrum = torch.exp(- (torch.square(alpha) - prefilter_alpha ** 2) * torch.square(l[None, None])) ** 2
    specular_coeffs = incoming_spectrum * microfacet_spectrum[..., None]
    specular_reflectance = (0.04 + (base_color - 0.04) * metallic[..., None]) ** 2
    out_spectrum = specular_reflectance[..., None, :] * specular_coeffs

    # Diffuse
    diffuse_reflectance = (1 - metallic[..., None]) * base_color
    diffuse = diffuse_reflectance[..., None, :] * irradiance / sqrt(pi)
    out_spectrum[..., 0, :] = torch.square(torch.sqrt(out_spectrum[..., 0, :] + 1e-8) + diffuse[..., 0, :])
    return out_spectrum
