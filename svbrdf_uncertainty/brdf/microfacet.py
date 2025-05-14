import torch


def trowbridge_reitz_lambda(theta, roughness):
    if roughness.dim() == theta.dim() - 1:
        roughness = roughness[..., None]
    return (torch.sqrt(1 + torch.square(roughness * torch.tan(theta))) - 1) / 2


def trowbridge_reitz_masking(theta_o, roughness):
    return 1 / (1 + trowbridge_reitz_lambda(theta_o, roughness))


def beckmann_lambda(theta, roughness):
    if roughness.dim() == theta.dim() - 1:
        roughness = roughness[..., None]
    a = 1 / (roughness * torch.abs(torch.tan(theta))).clip(1e-8)
    l = (1 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a)
    return torch.where(a >= 1.6, torch.zeros_like(l), l)


def beckmann_masking(theta_o, roughness):
    return 1 / (1 + beckmann_lambda(theta_o, roughness))


def fresnel(cos_theta_i, eta):
    sin2_theta_i = (1 - torch.square(cos_theta_i)).clamp(0)
    sin2_theta_t = sin2_theta_i / (eta * eta)
    cos_theta_t = torch.sqrt(1 - sin2_theta_t)
    r_parl = ((eta * cos_theta_i - cos_theta_t) /
              (eta * cos_theta_i + cos_theta_t))
    r_perp = ((cos_theta_i - eta * cos_theta_t) /
              (cos_theta_i + eta * cos_theta_t))
    return 0.5 * (torch.square(r_parl) + torch.square(r_perp))


def schlick_weight(cos_theta):
    return torch.pow(1 - cos_theta, 5)