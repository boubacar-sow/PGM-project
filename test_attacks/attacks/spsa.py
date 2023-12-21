import numpy as np
import torch

from torch import Tensor
from typing import Callable
import torch.nn.functional as F
import torch.nn as nn

variational_beta = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence



def spsa(
    model_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_iter: int, 
    lr: float, 
    c: float) -> Tensor:
    """Implementation of the Simultaneous Perturbation Stochastic Approximation (SPSA) attack with PyTorch.
    
    Args:
    
        model_fn (Callable[[Tensor], Tensor]): takes an input tensor and returns the model's output tensor
        x (Tensor): input tensor
        y (Tensor): target tensor
        num_iter (int): number of iterations
        lr (float): learning rate
        c (float): confidence
        
    Returns:    
        Tensor: adversarial example, a perturbed version of the input tensor
    """
    
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for i in range(num_iter):
        # Adaptive perturbation size
        c = c / (i + 1)**0.15

        delta = torch.randint_like(x_adv, 2) * 2 - 1
        x_plus = x_adv + c * delta
        x_minus = x_adv - c * delta

        x_recon_plus, y_pred_plus, latent_mu_plus, latent_logvar_plus = model_fn(x_plus, y)
        x_recon_minus, y_pred_minus, latent_mu_minus, latent_logvar_minus = model_fn(x_minus, y)

        loss_plus = vae_loss(x_recon_plus, x, latent_mu_plus, latent_logvar_plus)
        loss_minus = vae_loss(x_recon_minus, x, latent_mu_minus, latent_logvar_minus)

        grad = (loss_plus - loss_minus) * delta / (2 * c)
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        grad = grad / grad_norm

        # Line search for optimal step size
        best_loss = float('inf')
        best_lr = lr
        for factor in [-0.5, 1, 1.5]:
            lr_t = lr * factor
            x_t = x_adv + lr_t * grad
            x_t = torch.min(torch.max(x_t, x - c), x + c)
            x_t = torch.clamp(x_t, 0, 1)
            x_recon_t, y_pred_t, latent_mu_t, latent_logvar_t = model_fn(x_t, y)
            loss_t = vae_loss(x_recon_t, x, latent_mu_t, latent_logvar_t)
            if loss_t < best_loss:
                best_loss = loss_t
                best_lr = lr_t

        x_adv = x_adv + best_lr * grad
        x_adv = torch.min(torch.max(x_adv, x - c), x + c)
        x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv
