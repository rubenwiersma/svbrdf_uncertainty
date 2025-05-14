from math import pi, sqrt
from functools import reduce
from operator import mul
import torch

from functools import lru_cache, wraps

from .util import get_lmax, degree_order_indices


def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner
# constants

CACHE = {}

def clear_spherical_harmonics_cache():
    CACHE.clear()

def lpmv_cache_key_fn(l, m, x):
    return (l, m)

# spherical harmonics

@lru_cache(maxsize = 1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.)

@lru_cache(maxsize = 1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))

def negative_lpmv(l, m, y):
    if m < 0:
        y *= ((-1) ** m / pochhammer(l + m + 1, -2 * m))
    return y

@cache(cache = CACHE, key_fn = lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)

    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1)**m_abs * semifactorial(2*m_abs-1)
        y *= torch.pow(1-x*x, m_abs/2)
        return negative_lpmv(l, m, y)

    # Recursively precompute lower degree harmonics
    lpmv(l-1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2*l-1) / (l-m_abs)) * x * lpmv(l-1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l+m_abs-1)/(l-m_abs)) * CACHE[(l-2, m_abs)]

    if m < 0:
        y = negative_lpmv(l, m, y)
    return y

def evaluate_spherical_harmonics_element(l, m, theta, phi):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2*l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))

    if m == 0:
        return N * leg

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y *= leg
    N *= sqrt(2. / pochhammer(l - m_abs + 1, 2 * m_abs))
    Y *= N
    return Y

def evaluate_spherical_harmonic_degree(l, theta, phi):
    """Tesseral harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    clear_spherical_harmonics_cache()
    return torch.stack([evaluate_spherical_harmonics_element(l, m, theta, phi) for m in range(-l, l+1)], dim = -1)


def evaluate_spherical_harmonics(lmax, theta, phi):
    """Evaluate all spherical harmonics up to and including degree lmax
    for positions (theta, phi).
    """
    return torch.cat([evaluate_spherical_harmonic_degree(l, theta, phi) for l in range(lmax + 1)], dim=-1)


def power_spectrum(coeff, unit='per_lm'):
    """Compute the power spectrum of spherical harmonics coefficients.

    Args:
        coeff (torch.Tensor): Coefficients of spherical harmonics.
        unit (string, optional): Whether to compute the power per degree ('per_l')
            or per degree, per order ('per_lm'). Defaults to 'per_lm'.
    """
    assert coeff.dim() > 1, "coeff must be of shape [*, n_coeff, n_channels]"

    lmax = get_lmax(coeff)
    row, _ = degree_order_indices(lmax, coeff.device)
    row = row.view(*[1] * (coeff.dim() - 2), -1, 1).expand(*coeff.shape[:-2], -1, coeff.shape[-1])
    spectrum = torch.scatter_add(torch.zeros(coeff.shape[:-2] + (lmax + 1, coeff.shape[-1]), device=coeff.device, dtype=coeff.dtype), dim=-2, index=row, src=coeff.square())
    if unit == 'per_lm':
        spectrum /= (torch.arange(0, lmax + 1, device=coeff.device, dtype=coeff.dtype) + 1).unsqueeze(-1)
    return spectrum

