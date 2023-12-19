import numpy as np
import torch
import torch.nn as nn
from GBZ import Encoder, Decoder


class BayesModel(nn.Module):
    def __init__(self, data_name, hidden_channels, dimZ=64):
        super(BayesModel, self).__init__()
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
        self.num_labels = 10
        self.hidden_channels = hidden_channels
        self.use_mean = True
        self.training = True
        
        self.encoder = Encoder(hidden_channels=self.hidden_channels, latent_dim=dimZ)
        self.decoder = Decoder(hidden_channels=self.hidden_channels, latent_dim=dimZ)


    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        if self.use_mean:
            latent = latent_mu
        else:
            latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def log_gaussian_prob(self, x, mu, log_sig):
        logprob = -(0.5 * torch.log(2 * torch.pi * torch.ones(1)) + log_sig) \
                    - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
        return torch.sum(logprob, dim=1)
    
    def log_bernoulli_prob(self, x, p=0.5):
        logprob = x * torch.log(torch.clamp(p, 1e-9, 1.0)) \
                + (1 - x) * torch.log(torch.clamp(1.0 - p, 1e-9, 1.0))
        ind = list(range(1, len(x.shape)))
        return torch.sum(logprob, ind)
    
    def predict(self, x, K=100):
        batch_size = x.shape[0]
        num_classes = 10  # number of classes for MNIST

        mu, logvar = self.encoder(x)
        z_samples = [self.latent_sample(mu, logvar) for _ in range(K)]

        p_y = []
        for c in range(num_classes):
            y_c = torch.full((batch_size,), c, dtype=torch.long)
            log_pxz = [self.log_bernoulli_prob(x, self.decoder(z, y_c)) for z in z_samples]
            log_pxz = torch.stack(log_pxz, dim=1)
            log_pz = [self.log_gaussian_prob(z, torch.tensor(0.0), torch.tensor(0.0)) for z in z_samples]
            log_pz = torch.stack(log_pz, dim=1)
            log_qz = [self.log_gaussian_prob(z, mu, logvar) for z in z_samples]
            log_qz = torch.stack(log_qz, dim=1)

            log_w = log_pxz + log_pz - log_qz
            log_w = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            p_y_c = torch.exp(torch.logsumexp(log_w, dim=1))
            print(p_y_c)
            p_y.append(p_y_c)

        p_y = torch.stack(p_y, dim=1)

        return p_y



