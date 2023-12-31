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
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence

def fast_gradient_sign_method(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    clip_min=None,
    clip_max=None,
    targeted=False,
    sanity_checks=True,
) -> torch.Tensor:
    """
    Implementation of the Fast Gradient Sign Method (FGSM) attack with PyTorch.
    """
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    x_recon, y_pred, latent_mu, latent_logvar = model(x_adv, y)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')  # Use only the reconstruction loss
    if targeted:
        recon_loss = -recon_loss
    recon_loss.backward(retain_graph=True)  # Add retain_graph=True here

    # Check that gradients have been computed
    assert x_adv.grad is not None, "No gradients for x_adv"
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
    if sanity_checks:
        assert (x_adv - x).abs().max() <= epsilon + 1e-6, "Max perturbation exceeded"

    return x_adv








