import numpy as np
import torch

def carlini_wagner_l2(model_fn, x, y, num_iter, lr, c, kappa, criterion, clip_min=None, clip_max=None):
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
    
    device = x.device
    x_adv = x.clone().detach().requires_grad_(True).to(device)

    for _ in range(num_iter):
        outputs = model_fn(x_adv)
        loss = criterion(outputs, y)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        grad = grad / grad_norm

        x_adv = x_adv + lr * grad
        x_adv = torch.min(torch.max(x_adv, x - c), x + c)
        x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv
