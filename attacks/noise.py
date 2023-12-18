import numpy as np
import torch

from torch import Tensor
from typing import Optional

def noise(x: Tensor, epsilon: float, clip_min: Optional[float] = None, clip_max: Optional[float] = None) -> Tensor:
    """Implementation of the Noise attack with PyTorch.
    
    Args:
    
        x (Tensor): input tensor
        epsilon (float): perturbation size
        clip_min (Optional[float]): minimum value to clip the adversarial example to
        clip_max (Optional[float]): maximum value to clip the adversarial example to
        
    Returns:    
        Tensor: adversarial example, a perturbed version of the input tensor
    """
    
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    noise = torch.zeros_like(x_adv).to(device)
    noise.uniform_(-epsilon, epsilon)
    x_adv = x_adv + noise
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    return x_adv
