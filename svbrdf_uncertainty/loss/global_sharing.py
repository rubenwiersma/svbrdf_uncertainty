import drjit as dr

import torch
from skimage.feature import peak_local_max

def compute_global_sharing_inputs(param_probabilities, entropy, k=10, kl_divergence_kernel_width=0.2):
    local_min_entropy_idx = peak_local_max((1 - entropy).cpu().numpy(), min_distance=1, num_peaks=k)
    local_min_entropy_idx = torch.from_numpy(local_min_entropy_idx[:, 0] * param_probabilities.shape[0] + local_min_entropy_idx[:, 1]).to(param_probabilities.device)

    param_probabilities = param_probabilities.view(-1, *param_probabilities.shape[2:])
    kl_divergence = (param_probabilities[:, None] * torch.log((param_probabilities[:, None] / param_probabilities[None, local_min_entropy_idx].clip(1e-12)).clip(1e-12))).sum(dim=(-1, -2, -3))

    sigma = kl_divergence_kernel_width * kl_divergence.mean(dim=-1, keepdim=True)
    kl_divergence_weight = torch.exp(-torch.square(kl_divergence) / (2 * torch.square(sigma)))

    # Normalize weights
    kl_divergence_weight = kl_divergence_weight / kl_divergence_weight.sum(dim=-1, keepdim=True).clip(1e-12)

    return kl_divergence_weight, local_min_entropy_idx


def global_sharing(input, sharing_from_idx, kl_divergence_weight):
    # Input is expected to be [H, W, C]
    if input.dim() < 3: input = input.unsqueeze(-1)

    # Flatten H, W input dimensions
    input_ = input.view(-1, *input.shape[2:])
    # [N, K]
    parameter_difference = torch.square(input_[:, None] - input_[None, sharing_from_idx]).sum(dim=-1)

    bilateral_loss = (kl_divergence_weight * parameter_difference).mean(dim=-1)

    return bilateral_loss

@dr.wrap_ad(source='drjit', target='torch')
def global_sharing_dr(input, sharing_from_idx, kl_divergence_weight):
    return global_sharing(input, sharing_from_idx, kl_divergence_weight)

