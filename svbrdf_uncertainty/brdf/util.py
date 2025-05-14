import torch


def irradiance_per_point(envmap, normal, envmap_transform):
    envmap = envmap.to(torch.float16)
    normal = normal.to(torch.float16)
    if envmap_transform is not None:
        envmap_transform = envmap_transform.to(torch.float16)
    # Compute direction in world space for every envmap pixel
    envmap_theta, envmap_phi = torch.meshgrid(
        torch.linspace(0, torch.pi, envmap.shape[0], device=normal.device, dtype=torch.float16),
        torch.linspace(-torch.pi, torch.pi, envmap.shape[1], device=normal.device, dtype=torch.float16),
        indexing='ij')
    envmap_sinphi = torch.sin(envmap_phi)
    envmap_sintheta = torch.sin(envmap_theta)
    envmap_pos = torch.stack([
        -envmap_sinphi * envmap_sintheta,
        torch.cos(envmap_theta),
        torch.cos(envmap_phi) * envmap_sintheta
    ], axis=-1) # [256, 512, 3]

    # Compute cosine of the angle between the normal at each point and the direction of envmap pixels
    if envmap_transform is not None:
        normal = (normal.view(-1, 3) @ envmap_transform[:3, :3]).view(normal.shape)
    envmap_cos_theta_local = (normal[None, None] * envmap_pos[:, :, None, None]).sum(dim=-1, keepdim=True).clamp(0)

    # Multiply incoming radiance with cos(theta) and integrate over sphere
    irradiance = (envmap_sintheta[:, :, None, None, None] * envmap_cos_theta_local * envmap[:, :, None, None]).mean(dim=(0, 1)) * 2 * torch.pi * torch.pi
    return irradiance