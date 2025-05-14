import torch

from .spherical_harmonics import evaluate_spherical_harmonics
from .util import get_lmax, degree_order_indices


def expand_spherical_harmonics(coeff, res_theta, res_phi):
    """Expand a list of coefficients for spherical harmonics to a grid in angular space.
    """
    lmax = get_lmax(coeff)
    lat, long = torch.meshgrid(
        torch.linspace(0, torch.pi, res_theta, dtype=coeff.dtype, device=coeff.device),
        torch.linspace(-torch.pi, torch.pi, res_phi, dtype=coeff.dtype, device=coeff.device),
        indexing='ij')
    S = evaluate_spherical_harmonics(lmax, lat, long)
    return (S @ coeff).reshape(res_theta, res_phi, coeff.shape[1])


def lsq_spherical_harmonics(values, lat, long, lmax, weight=None, regularizer=0, regularizer_func=torch.exp, return_factorization=False):
    """Fit spherical harmonics bases using least-squares.
    This function is compatible with batch-wise computation.
    That means values can be of shape [*, n_samples, n_channels], where * denotes
    any number of dimensions. Lat and long must be of shape [*, n_samples].

    Args:
        values (torch.Tensor): Values to fit.
        lat (torch.Tensor): Coordinates in latitudal direction in radians (from pole to pole).
        long (torch.Tensor): Coordinates in longtitudal direction in radians (along azimuth).
        lmax (int): Maximum degree of spherical harmonics.
        mask (torch.Tensor, optional): Mask to apply to values (default: None).
        regularizer (float, optional): Regularization parameter (default: 0).
        regularizer_func (callable, optional): Function to compute the regularization
            term per degree (default: None)
        return_basis (boolean): Whether to return the spherical harmonics basis.
    """
    assert values.dim() > 1, "values must be of shape [*, n_samples, n_channels]"
    assert lat.dim() == values.dim() - 1, "lat must be of shape [*, n_samples]"
    assert long.dim() == values.dim() - 1, "long must be of shape [*, n_samples]"
    assert weight is None or weight.shape == lat.shape, "weight must be of shape [*, n_samples]"

    # 1. Compute the matrix of basis functions S [*, n_samples, n_bases]
    S = evaluate_spherical_harmonics(lmax, lat, long)
    if weight is not None:
        S = S * weight.unsqueeze(-1)
        values = values * weight.unsqueeze(-1)

    # 2. Compute the least-squares solution matrix
    ST = S.transpose(-1, -2)
    lhs = ST @ S
    if regularizer > 0:
        if regularizer_func is not None:
            degree_l, _ = degree_order_indices(lmax, device=values.device)
            tikhonov_matrix = torch.diag(regularizer_func(degree_l))
        else:
            tikhonov_matrix = torch.eye(lhs.shape[0], device=values.device, dtype=values.dtype)
        lhs = lhs + regularizer * tikhonov_matrix

    rhs = ST @ values

    if return_factorization:
        lhs_factor = torch.linalg.lu_factor(lhs)
        coeff = torch.linalg.lu_solve(*lhs_factor, rhs)
        return coeff, lhs_factor, S

    coeff = torch.linalg.solve(lhs, rhs)
    return coeff
