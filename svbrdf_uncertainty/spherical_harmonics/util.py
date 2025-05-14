from math import sqrt

import torch

from ..util.sample_sphere import sample_fibonacci_hemisphere


def get_lmax(coeff):
    """Compute the number of spherical harmonics degrees
    in a list of coefficients given in the format [*, n_coefficients, n_channels].
    """
    n_coefficients = coeff.shape[-2]
    return int(sqrt(n_coefficients) - 1)


def degree_order_indices(lmax, device='cpu'):
    """Compute indices of spherical harmonics coefficients to index into a flat array.
    """
    order_m = torch.arange(-lmax, lmax + 1, device=device).repeat(lmax + 1, 1)
    degree_l = torch.arange(lmax + 1, device=device).repeat(lmax * 2 + 1, 1).T
    mask = order_m.abs() <= degree_l
    return degree_l[mask], order_m[mask]


def coeffs_list_to_matrix(coeffs_list):
    """
    Converts a list of coefficients for spherical harmonics to
    a matrix following the layout of the Vandermonde matrix.

    Args:
        coeffs_list (torch.Tensor): Coefficients of spherical harmonics.
            Format should be [*, n_coefficients, n_channels],
            where * denotes any number of dimensions.

    Returns:
        A complex Tensor of size [*, n_channels, lmax, mmax].
    """
    lmax = get_lmax(coeffs_list)
    row, col = degree_order_indices(lmax, device=coeffs_list.device)
    coeffs_mat = torch.zeros(coeffs_list.shape[:-2] + (coeffs_list.shape[-1], lmax + 1, lmax * 2 + 1), device=coeffs_list.device, dtype=coeffs_list.dtype)
    coeffs_mat[..., row, col] = coeffs_list.transpose(-2, -1)
    coeffs_real = coeffs_mat[..., :lmax + 1]
    coeffs_imag = coeffs_mat[..., lmax:].flip(dims=[-2])
    coeffs_imag[..., 0] = 0
    coeffs_array = torch.stack([coeffs_real, coeffs_imag], dim=-1)
    return torch.view_as_complex(coeffs_array)


def coeffs_matrix_to_list(coeffs_matrix):
    """
    Converts a matrix of coefficients for spherical harmonics
    following the Vandermonde matrix layout to a list of coefficients.

    Args:
        coeffs_matrix: A complex Tensor of size [*, n_channels, lmax, mmax].

    Returns:
        Coefficients of spherical harmonics.
            Format should be [*, n_coefficients, n_channels],
            where * denotes any number of dimensions.
    """
    lmax = coeffs_matrix.shape[-2] - 1
    row, col = degree_order_indices(lmax, device=coeffs_matrix.device)
    coeffs_mat = torch.zeros(coeffs_matrix.shape[:-2] + (lmax + 1, lmax * 2 + 1), device=coeffs_matrix.device, dtype=torch.double)
    coeffs_mat[..., :lmax + 1] = coeffs_matrix.real
    coeffs_mat[..., lmax + 1:] = coeffs_matrix.imag[..., 1:].flip(dims=[-1])
    return coeffs_mat[..., row, col].transpose(-2, -1)


def zero_pad_hemisphere(data, theta, phi, n_points):
    """Pad a signal in data for points on the sphere at (theta, phi)
    with zeros on the bottom hemisphere at points sampled on a Fibonacci lattice.

    Args:
        data (torch.Tensor): data to be padded in the format [H, W, N, C],
            where H and W are the height and with of a surface texture,
            N is the number of original points and C the number of channels.
        theta (torch.Tensor): theta coordinate of the input data [H, W, N]
        phi (torch.Tensor): phi coordinate of the input data [H, W, N]
    """
    zeros = torch.zeros(data.shape[:2] + (n_points,) + data.shape[-1:], dtype=data.dtype, device=data.device)
    data_padded = torch.cat([data, zeros], dim=-2)

    fib_theta, fib_phi = sample_fibonacci_hemisphere(n_points, device=theta.device)
    theta_padded = torch.cat([theta, torch.pi - fib_theta[None, None].expand(*theta.shape[:2], -1)], dim=-1)
    phi_padded = torch.cat([phi, fib_phi[None, None].expand(*phi.shape[:2], -1)], dim=-1)
    return data_padded, theta_padded, phi_padded