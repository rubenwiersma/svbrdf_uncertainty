import mitsuba as mi
import torch

from .microfacet import trowbridge_reitz_masking, beckmann_masking
from ..spherical_harmonics.util import degree_order_indices, get_lmax


class PrincipledAngular(torch.nn.Module):
    def __init__(self, texture_res, SH_basis, roughness_admitted=(0, 1), metallic_admitted=(0, 1), base_color_admitted=(0, 1), fresnel_enabled=True, masking_enabled=True):
        super(PrincipledAngular, self).__init__()
        self.base_color = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res, 3), requires_grad=True))
        self.roughness = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res), requires_grad=True))
        self.metallic = torch.nn.Parameter(0.5 * torch.ones((texture_res, texture_res), requires_grad=True))

        self.base_color_admitted = base_color_admitted
        self.roughness_admitted = roughness_admitted
        self.metallic_admitted = metallic_admitted

        self.SH_basis = SH_basis
        self.fresnel_enabled = fresnel_enabled
        self.masking_enabled = masking_enabled


    def clamp_parameters(self):
        self.roughness.data = self.roughness.data.abs().clamp(*self.roughness_admitted)
        self.base_color.data = self.base_color.data.clamp(*self.base_color_admitted)
        self.metallic.data = self.metallic.data.clamp(*self.metallic_admitted)

    def forward(self, incoming_coeffs, irradiance, schlick_weight, theta, prefilter_alpha=0):
        return principled_brdf_angular(incoming_coeffs, irradiance, schlick_weight, theta,
                                       self.SH_basis, self.roughness, self.metallic, self.base_color,
                                       self.fresnel_enabled, self.masking_enabled, prefilter_alpha)

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


def principled_brdf_angular(incoming_coeffs, irradiance, schlick_weight, theta_local,
                            SH_basis, roughness, metallic, base_color,
                            fresnel_enabled=True, masking_enabled=True, distribution='trowbridge_reitz',
                            prefilter_alpha=0):
    assert isinstance(roughness, torch.Tensor)
    assert isinstance(metallic, torch.Tensor)
    assert isinstance(base_color, torch.Tensor)
    assert base_color.dim() == 3 and base_color.shape[-1] == 3, "Base color should be given as RGB"

    # Specular
    # Filter incoming light
    lmax = get_lmax(incoming_coeffs)
    l, _ = degree_order_indices(lmax, device=roughness.device)
    # Compute coefficients for filtering with to the normal distribution function
    alpha = torch.square(roughness[..., None]).clamp(0.001)
    microfacet_coeffs = torch.exp(- (torch.square(alpha) - prefilter_alpha ** 2) * torch.square(l[None, None]))
    specular_coeffs = incoming_coeffs * microfacet_coeffs[..., None]
    # Transform back to angular domain
    specular = SH_basis @ specular_coeffs

    # Compute Fresnel term
    # Approximation with schlick weight
    R_0 = 0.04 + (base_color - 0.04) * metallic[..., None]
    if fresnel_enabled:
        F = R_0[..., None, :] + (1 - R_0[..., None, :]) * schlick_weight[..., None]
    else:
        F = R_0[..., None, :]

    # Compute masking term
    if masking_enabled:
        masking_fun = beckmann_masking if distribution == 'beckmann' else trowbridge_reitz_masking
        G = masking_fun(theta_local, alpha)[..., None]
        specular = G * specular
    specular = F * specular

    # Diffuse
    invpi = 1 / torch.pi
    diffuse_reflectance = (1 - metallic[..., None]) * base_color * invpi
    diffuse = irradiance * diffuse_reflectance[..., None, :]

    return specular + diffuse