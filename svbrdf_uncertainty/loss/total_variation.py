import gc

import drjit as dr
import torch


def total_variation_loss_knn(input, knn, norm: str = 'l1'):
    """Total variation loss.

    Args:
        input: Input tensor.
        norm: Norm type, 'l1' or 'l2'.

    Returns:
        Total variation loss.
    """
    # Input is expected to be [H, W, C]
    if input.dim() < 3: input = input.unsqueeze(-1)

    # Flatten H, W input dimensions
    input_ = input.view(-1, *input.shape[2:])
    # [N, K]
    if knn.shape[0] < input_.shape[0]:
        diff = input_[:, None] - input_[None, knn]
    else:
        diff = input_[:, None] - input_[knn].view(input_.shape[0], -1, *input_.shape[1:])

    # The norm is first computed over the channels and then summed over the spatial dimensions
    if norm == 'l2':
        loss = torch.sqrt(torch.square(diff).sum(dim=-1))
    elif norm == 'l2sq':
        loss = torch.square(diff).sum(dim=-1)
    elif isinstance(norm, float):
        loss = torch.pow(diff.abs(), norm).sum(dim=-1)
    else:
        loss = diff.abs().sum(dim=-1)

    return loss.mean(dim=-1)

def total_variation_loss(input, norm: str = 'l1', mask = None):
    """Total variation loss.

    Args:
        input: Input tensor.
        norm: Norm type, 'l1' or 'l2'.

    Returns:
        Total variation loss.
    """
    # Input is expected to be [H, W, C]
    if input.dim() < 3: input = input.unsqueeze(-1)
    sh = input.shape
    # We need to permute to [1, C, H, W] for torch's unfold to work
    input = input.permute(2, 0, 1).unsqueeze(0)

    input_unfold = torch.nn.functional.unfold(input, kernel_size=3, padding=1).view(1, sh[2], -1, sh[0], sh[1])
    diff = input_unfold - input.unsqueeze(2)

    if mask is not None:
        mask = mask[None, None] * 1.0
        mask_unfold = torch.nn.functional.unfold(mask, kernel_size=3, padding=1).view(1, 1, -1, sh[0], sh[1])
        diff_mask = mask_unfold * mask.unsqueeze(2)
        diff = diff_mask * diff

    # The norm is first computed over the channels and then summed over the spatial dimensions
    if norm == 'l2':
        loss = torch.sqrt(torch.square(diff).sum(dim=1))
    elif norm == 'l2sq':
        loss = torch.square(diff).sum(dim=1)
    elif isinstance(norm, float):
        loss = torch.pow(diff.abs(), norm).sum(dim=1)
    else:
        loss = diff.abs().sum(dim=1)

    return loss


@dr.wrap_ad(source='drjit', target='torch')
def total_variation_loss_dr(input, norm: str = 'l1'):
    return total_variation_loss(input, norm)