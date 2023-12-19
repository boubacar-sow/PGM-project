import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable

def fast_gradient_sign_method(model_fn: Callable[[Tensor], Tensor], x: Tensor, y: Tensor, epsilon: float, clip_min=None, clip_max=None) -> Tensor:
    """Implementation of the Fast Gradient Sign Method (FGSM) attack with PyTorch.

    Args:
        model_fn (Callable[[Tensor], Tensor]): takes an input tensor and returns the model's output tensor
        x (Tensor): input tensor
        y (Tensor): target tensor
        epsilon (float): perturbation size
        criterion (_Loss): takes the model's output tensor and the target tensor and returns the loss tensor

    Returns:
        Tensor: adversarial example, a perturbed version of the input tensor
    """
    
    device = x.device
    criterion = nn.MSELoss()
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    x_recon, _, _ = model_fn(x_adv)
    loss = criterion(x_recon.view(-1, 784), x.view(-1, 784))
    loss.backward()

    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

    return x_adv






