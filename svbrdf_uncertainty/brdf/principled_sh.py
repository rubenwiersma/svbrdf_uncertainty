from math import sqrt, pi

import mitsuba as mi
import torch

from ..spherical_harmonics.util import degree_order_indices, get_lmax


class PrincipledSH(torch.nn.Module):
    def __init__(self, texture_res, roughness_admitted=(0, 1), metallic_admitted=(0, 1), base_color_admitted=(0, 1)):
        super(PrincipledSH, self).__init__()
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

    def forward(self, incoming_coeffs, irradiance, prefilter_alpha=0):
        return principled_brdf_sh(incoming_coeffs, irradiance, self.roughness, self.metallic, self.base_color, prefilter_alpha)

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


def principled_brdf_sh(incoming_coeffs, irradiance, roughness, metallic, base_color, prefilter_alpha=0):
    assert isinstance(roughness, torch.Tensor)
    assert isinstance(metallic, torch.Tensor)
    assert isinstance(base_color, torch.Tensor)
    assert base_color.dim() == 3 and base_color.shape[-1] == 3, "Base color should be given as RGB"

    # Specular
    lmax = get_lmax(incoming_coeffs)
    l, _ = degree_order_indices(lmax, device=roughness.device)
    alpha = torch.square(roughness[..., None]).clamp(0.001)
    microfacet_coeffs = torch.exp(- (torch.square(alpha) - prefilter_alpha ** 2) * torch.square(l[None, None]))
    specular_coeffs = incoming_coeffs * microfacet_coeffs[..., None]
    specular_reflectance = 0.04 + (base_color - 0.04) * metallic[..., None]
    specular = specular_reflectance[..., None, :] * specular_coeffs

    # Diffuse
    diffuse_reflectance = (1 - metallic[..., None]) * base_color
    diffuse = diffuse_reflectance[..., None, :] * irradiance / sqrt(pi)
    diffuse = torch.cat([diffuse, torch.zeros_like(specular[:, :, 1:])], dim=-2)
    return diffuse + specular
