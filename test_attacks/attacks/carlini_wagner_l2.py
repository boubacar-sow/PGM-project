import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

variational_beta = 1

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


def carlini_wagner_l2(
    model_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_iter: int, 
    lr: float, 
    c: float, 
    clip_min=None, 
    clip_max=None):
    """Implementation of the Carlini & Wagner L2 attack with PyTorch.
    
    Args:
    
        model_fn (callable): takes an input tensor and returns the model's output tensor
        x (tensor): input tensor
        y (tensor): target tensor
        num_iter (integer): number of iterations
        lr (integer): learning rate
        c (integer): confidence
        kappa (integer): kappa
        criterion (callable): takes the model's output tensor and the target tensor and returns the loss tensor
        
    Returns:    
        tensor: adversarial example, a perturbed version of the input tensor
    """
    
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for _ in range(num_iter):
        x_recon, y_pred, latent_mu, latent_logvar = model_fn(x_adv, y)
        loss = vae_loss(x_recon, x, latent_mu, latent_logvar)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        grad = grad / grad_norm

        x_adv = x_adv + lr * grad
        x_adv = torch.min(torch.max(x_adv, x - c), x + c)
        x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv
