import numpy as np
import torch


from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable


def spsa(
    model_fn: Callable[[Tensor], Tensor],
    x: Tensor,
    y: Tensor,
    num_iter: int,
    lr: float,
    c: float,
    criterion: _Loss,
) -> Tensor:
    """Implementation of the Simultaneous Perturbation Stochastic Approximation (SPSA) attack with PyTorch.

    Args:

        model_fn (Callable[[Tensor], Tensor]): takes an input tensor and returns the model's output tensor
        x (Tensor): input tensor
        y (Tensor): target tensor
        num_iter (int): number of iterations
        lr (float): learning rate
        c (float): confidence
        criterion (_Loss): takes the model's output tensor and the target tensor and returns the loss tensor

    Returns:
        Tensor: adversarial example, a perturbed version of the input tensor
    """

    device = x.device
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for i in range(num_iter):
        outputs = model_fn(x_adv)
        loss = criterion(outputs, y)

        # Adaptive perturbation size
        c = c / (i + 1) ** 0.15

        delta = torch.randint_like(x_adv, 2) * 2 - 1
        x_plus = x_adv + c * delta
        x_minus = x_adv - c * delta

        outputs_plus = model_fn(x_plus)
        outputs_minus = model_fn(x_minus)

        loss_plus = criterion(outputs_plus, y)
        loss_minus = criterion(outputs_minus, y)

        grad = (loss_plus - loss_minus) * delta / (2 * c)
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        grad = grad / grad_norm

        # Line search for optimal step size
        best_loss = float("inf")
        best_lr = lr
        for factor in [-0.5, 1, 1.5]:
            lr_t = lr * factor
            x_t = x_adv + lr_t * grad
            x_t = torch.min(torch.max(x_t, x - c), x + c)
            x_t = torch.clamp(x_t, 0, 1)
            loss_t = criterion(model_fn(x_t), y)
            if loss_t < best_loss:
                best_loss = loss_t
                best_lr = lr_t

        x_adv = x_adv + best_lr * grad
        x_adv = torch.min(torch.max(x_adv, x - c), x + c)
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv
