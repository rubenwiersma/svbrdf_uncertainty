import drjit as dr
import mitsuba as mi
import torch


def smoothness_loss(img):
    """Compute dirichlet energy (smoothness term) on image."""
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    # Compute dirichlet energy as the sum of squared gradients
    return torch.mean(torch.square(grad_x)) + torch.mean(torch.square(grad_y))

def smoothness_loss_dr(img):
    """Compute dirichlet energy (smoothness term) on image."""
    assert isinstance(img, mi.TensorXf)

    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    # Compute dirichlet energy as the sum of squared gradients
    return dr.mean(dr.sqr(grad_x)) + dr.mean(dr.sqr(grad_y))

def gradient_x(img):
    """Compute gradient along x-axis. Assumes img is [H, W, C]"""
    return img[:, :-1] - img[:, 1:]

def gradient_y(img):
    """Compute gradient along y-axis. Assumes img is [H, W, C]"""
    return img[:-1, :] - img[1:, :]