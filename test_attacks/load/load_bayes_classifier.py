import numpy as np
import torch

class BayesModel(nn.Module):
    def __init__(self, data_name, vae_type, conv, K, attack_snapshot=False, use_mean=False, fix_samples=False, dimZ=64):
        super(BayesModel, self).__init__()
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
        self.num_labels = 10
        self.conv = conv
        self.K = K
        if no_z:
            use_mean = False
            attack_snapshot = False
            fix_samples = False
        if fix_samples:
            use_mean = False
            attack_snapshot = False
            no_z = False
        if use_mean:
            attack_snapshot = False
            fix_samples = False
            no_z = False
        if attack_snapshot:
            use_mean = False
            fix_samples = False
            no_z = False

        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=dimZ)
        self.decoder = Decoder(hidden_channels=hidden_channels, latent_dim=dimZ)

        self.use_mean = use_mean
        self.fix_samples = fix_samples
        self.attack_snapshot = attack_snapshot
        self.no_z = no_z

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
