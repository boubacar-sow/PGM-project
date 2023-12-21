import numpy as np
import torch

from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable
import torch.nn.functional as F
import torch.nn as nn

variational_beta = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


def projected_gradient_descent(
    model_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor, 
    epsilon: float, 
    alpha: float, 
    num_iter: int) -> Tensor:
    """Implementation of the Projected Gradient Descent (PGD) attack with PyTorch.
    
    Args:
    
        model_fn (Callable[[Tensor], Tensor]): takes an input tensor and returns the model's output tensor
        x (Tensor): input tensor
        y (Tensor): target tensor
        epsilon (float): perturbation size
        alpha (float): step size
        num_iter (int): number of iterations
        
    Returns:    
        Tensor: adversarial example, a perturbed version of the input tensor
    """
    
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for _ in range(num_iter):
        x_recon, y_pred, latent_mu, latent_logvar = model_fn(x_adv, y)
        loss = vae_loss(x_recon, x, latent_mu, latent_logvar)
        loss.backward()

        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv