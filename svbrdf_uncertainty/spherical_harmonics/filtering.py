import torch
import torch_harmonics as th


def bandlimit_spherical(data, bandlimit_L):
    """Bandlimit a spherical signal.

    Args:
        data (torch.Tensor): spherical image of shape [H, W, C].
        bandlimit_L (int): the degree L at which to bandlimit the data.
    """
    sht = th.RealSHT(data.shape[0], data.shape[1], bandlimit_L, bandlimit_L, grid="equiangular").to(data.device)
    isht = th.InverseRealSHT(data.shape[0], data.shape[1], bandlimit_L, bandlimit_L, grid="equiangular").to(data.device)
    return isht(sht(data.permute(2, 0, 1))).permute(1, 2, 0)


def gaussian_filter_spherical(data, sigma):
    """Smooth a spherical signal with a Gaussian kernel.

    Args:
        data (torch.Tensor): spherical image of shape [H, W, C].
        sigma (float): the kernel width of the gaussian.
    """
    bandlimit_L = min(int(2 / sigma), 100)
    sht = th.RealSHT(data.shape[0], data.shape[1], bandlimit_L, bandlimit_L, grid="equiangular").to(data.device)
    isht = th.InverseRealSHT(data.shape[0], data.shape[1], bandlimit_L, bandlimit_L, grid="equiangular").to(data.device)

    coeffs = sht(data.permute(2, 0, 1))
    gaussian = torch.exp(-(sigma * torch.arange(coeffs.shape[1], device=coeffs.device)) ** 2)
    coeffs = coeffs * gaussian[None, :, None]

    return isht(coeffs).permute(1, 2, 0)