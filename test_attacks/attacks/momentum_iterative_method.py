import numpy as np
import torch

from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable


def momentum_iterative_method(
    model_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    epsilon: float,
    alpha: float,
    num_iter: int,
    decay_factor: float,
    criterion: _Loss,
) -> Tensor:
    """Implementation of the Momentum Iterative Method (MIM) attack with PyTorch.

    Args:

        model_fn (Callable[[Tensor], Tensor]): takes an input tensor and returns the model's output tensor
        x (Tensor): input tensor
        y (Tensor): target tensor
        epsilon (float): perturbation size
        alpha (float): step size
        num_iter (int): number of iterations
        decay_factor (float): decay factor for the momentum term
        criterion (_Loss): takes the model's output tensor and the target tensor and returns the loss tensor

    Returns:
        Tensor: adversarial example, a perturbed version of the input tensor
    """

    device = x.device
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    momentum = torch.zeros_like(x_adv).to(device)

    for _ in range(num_iter):
        outputs = model_fn(x_adv)
        loss = criterion(outputs, y)
        loss.backward()

        grad = x_adv.grad.detach()
        grad_norm = torch.norm(grad, p=1, dim=(1, 2, 3), keepdim=True)
        grad = grad / grad_norm

        momentum = decay_factor * momentum + grad
        x_adv = x_adv + alpha * momentum.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv
