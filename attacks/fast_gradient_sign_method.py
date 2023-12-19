import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Callable

variational_beta = 1
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
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

    Args:
        model (nn.Module): the model to attack
        x (torch.Tensor): input tensor
        epsilon (float): perturbation size
        clip_min (float, optional): minimum value to clip the adversarial examples to
        clip_max (float, optional): maximum value to clip the adversarial examples to
        targeted (bool, optional): whether to perform a targeted attack
        sanity_checks (bool, optional): whether to perform sanity checks

    Returns:
        torch.Tensor: adversarial examples
    """
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    x_recon, latent_mu, latent_logvar = model(x_adv, y)
    loss = vae_loss(x_recon, x, latent_mu, latent_logvar)
    if targeted:
        loss = -loss
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)
    if sanity_checks:
        assert (x_adv - x).abs().max() <= epsilon + 1e-6, "Max perturbation exceeded"

    return x_adv






