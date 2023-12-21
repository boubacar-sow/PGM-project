import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable

variational_beta = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence

def momentum_iterative_method(
    model_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float, 
    num_iter: int, 
    decay_factor: float) -> Tensor:
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
    
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    momentum = torch.zeros_like(x_adv).to(device)

    for _ in range(num_iter):
        x_recon, y_pred, latent_mu, latent_logvar = model_fn(x_adv, y)
        loss = vae_loss(x_recon, x, latent_mu, latent_logvar)
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