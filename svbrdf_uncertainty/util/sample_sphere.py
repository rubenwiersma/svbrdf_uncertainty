import numpy as np
import torch


def fps_mask(pos, n_samples):
    from torch_cluster import fps
    
    ratio = n_samples / pos.shape[0]

    device = pos.device
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    index = fps(pos, batch, ratio=ratio, random_start=False)

    # Create mask
    mask = torch.zeros(batch.shape[0], dtype=torch.bool, device=device)
    mask[index] = 1
    return mask


# Copied from ivt: pypi.org/project/ivt
def sample_sphere(n_samples, radius=1, method='fibonacci', axis=1):
    # assumes y-axis is up by default, otherwise we swap
    if method == 'uniform':
        r1 = np.random.rand(n_samples, 1)
        r2 = np.random.rand(n_samples, 1)
        samples = np.concatenate([
            2 * np.cos(2 * np.pi * r1) * np.sqrt(r2 * (1 - r2)),
            2 * np.sin(2 * np.pi * r1) * np.sqrt(r2 * (1 - r2)),
            (1 - 2 * r2)
        ], axis=1) * radius

    elif method == 'stratified':
        cosPhi = np.linspace(1.0 - 0.01, -1.0 + 0.01, n_samples)[..., np.newaxis]
        theta  = np.linspace(0, np.pi * 10, n_samples)[..., np.newaxis] \
            + np.random.rand(n_samples, 1) * 0.01
        sinPhi = np.sqrt(1 - cosPhi * cosPhi)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        samples = np.concatenate([
            sinPhi * cosTheta,
            cosPhi,
            sinPhi * sinTheta
        ], axis=1) * radius

    elif method == 'fibonacci':
        # From http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        golden_ratio = (1 + 5**0.5) / 2
        i = np.arange(n_samples)[..., np.newaxis]
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n_samples)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        samples = np.concatenate([
            sinPhi * cosTheta,
            cosPhi,
            sinPhi * sinTheta
        ], axis=1) * radius

    # Switch axis
    index = np.array([0, 1, 2], dtype=np.int64)
    index[axis] = 1
    index[1] = axis
    points = samples[:, index]

    return points

def sample_hemisphere(n_samples, radius, method='stratified', axis=1):
    # assumes y-axis is up by default, otherwise we swap
    if method == 'uniform':
        phi = np.random.rand(n_samples, 1) * np.pi * 0.5
        theta = np.random.rand(n_samples, 1) * 2 * np.pi
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        samples = np.concatenate([
            sinPhi * cosTheta,
            cosPhi,
            sinPhi * sinTheta
        ], axis=1) * radius

    elif method == 'stratified':
        cosPhi = np.linspace(1.0 - 0.01, 0.0 + 0.01, n_samples)[..., np.newaxis]
        theta  = np.linspace(0, np.pi * 10, n_samples)[..., np.newaxis] \
            + np.random.rand(n_samples, 1) * 0.01

        sinPhi = np.sqrt(1 - cosPhi * cosPhi)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        samples = np.concatenate([
            sinPhi * cosTheta,
            cosPhi,
            sinPhi * sinTheta
        ], axis=1) * radius

    elif method == 'fibonacci':
        # From http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        golden_ratio = (1 + 5**0.5) / 2
        i = np.arange(n_samples)[..., np.newaxis]
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - (i + 0.5) / n_samples)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        samples = np.concatenate([
            sinPhi * cosTheta,
            cosPhi,
            sinPhi * sinTheta
        ], axis=1) * radius

    # Switch axis
    index = np.array([0, 1, 2], dtype=np.int64)
    index[axis] = 1
    index[1] = axis
    points = samples[:, index]

    return points

def sample_fibonacci_hemisphere(n_samples, device='cpu'):
    golden_ratio = (1 + 5 ** 0.5) / 2
    i = torch.arange(n_samples, device=device)
    phi = 2 * torch.pi * i / golden_ratio
    theta = torch.arccos(1 - (i + 0.5) / n_samples)

    return theta % torch.pi, phi % (2 * torch.pi) - torch.pi

def sample_fibonacci_sphere(n_samples, device='cpu'):
    golden_ratio = (1 + 5 ** 0.5) / 2
    i = torch.arange(n_samples, device=device)
    phi = 2 * torch.pi * i / golden_ratio
    theta = torch.arccos(1 - 2 * (i + 0.5) / n_samples)

    return theta % torch.pi, phi % (2 * torch.pi) - torch.pi
