import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from GBZ import Encoder, Decoder


class BayesModel(nn.Module):
    def __init__(self, data_name, hidden_channels, dimZ=64):
        super(BayesModel, self).__init__()
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
            self.num_labels = 10
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
            self.num_labels = 10
        self.hidden_channels = hidden_channels
        self.use_mean = True
        self.training = True
        
        self.encoder = Encoder(hidden_channels=self.hidden_channels, latent_dim=dimZ, num_labels=self.num_labels)
        self.decoder = Decoder(hidden_channels=self.hidden_channels, latent_dim=dimZ, num_labels=self.num_labels)

        # MLP for p(y|z)
        self.fc_py_z = nn.Linear(dimZ, self.num_labels)
        
    def log_softmax_prob(self, z, y):
        """
        Compute the log probability log p(y|z).
        """
        logit_y = self.fc_py_z(z)
        log_p_y_z = -F.cross_entropy(logit_y, torch.argmax(y, dim=1), reduction='none')
        return log_p_y_z
    
    def forward(self, x, y):
        latent_mu, latent_logvar = self.encoder(x, y)
        if self.use_mean:
            latent = latent_mu
        else:
            latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, y)
        return x_recon, latent_mu, latent_logvar


    import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/content/PGM-project')

from GBZ import Encoder, Decoder

# Assuming you have already imported torch and torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BayesModel(nn.Module):
    def __init__(self, data_name, hidden_channels, dimZ=64):
        super(BayesModel, self).__init__()
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
            self.num_labels = 10
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
            self.num_labels = 10
        self.hidden_channels = hidden_channels
        self.use_mean = True
        self.training = True
        
        self.encoder = Encoder(hidden_channels=self.hidden_channels, latent_dim=dimZ, num_labels=self.num_labels)
        self.decoder = Decoder(hidden_channels=self.hidden_channels, latent_dim=dimZ, num_labels=self.num_labels)

        # MLP for p(y|z)
        self.fc_py_z = nn.Linear(dimZ, self.num_labels)
        
    def log_softmax_prob(self, z, y):
        """
        Compute the log probability log p(y|z).
        """
        logit_y = self.fc_py_z(z)
        log_p_y_z = -F.cross_entropy(logit_y, torch.argmax(y, dim=1), reduction='none')
        return log_p_y_z
    
    def forward(self, x, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        y = y.to(device)
        latent_mu, latent_logvar = self.encoder(x, y)
        if self.use_mean:
            latent = latent_mu
        else:
            latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, y)
        return x_recon, latent_mu, latent_logvar


    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def log_gaussian_prob(self, x, mu, log_sig):
        logprob = -(0.5 * torch.log(2 * torch.pi * torch.ones(1).to(device)) + log_sig) \
                    - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
        return torch.sum(logprob, dim=1)

    
    def log_bernoulli_prob(self, x, p=0.5):
        logprob = x * torch.log(torch.clamp(p, 1e-9, 1.0)) \
                + (1 - x) * torch.log(torch.clamp(1.0 - p, 1e-9, 1.0))
        ind = list(range(1, len(x.shape)))
        return torch.sum(logprob, ind)
    
    def lower_bound(self, x, y_one_hot, K=100, beta=1.0):
        """
        Compute the lower bound for the input x and label y using importance sampling.
        """
        log_joint_probs = torch.zeros(x.shape[0], device=x.device)

        z_samples = [self.latent_sample(*self.encoder(x, y_one_hot)) for _ in range(K)]

        for z in z_samples:
            log_p_x_z = self.log_bernoulli_prob(self.decoder(z, y_one_hot), x)
            log_p_z = self.log_gaussian_prob(z, torch.zeros_like(z), torch.zeros_like(z))
            log_p_y_z = self.log_softmax_prob(z, y_one_hot)
            log_q_z_x = self.log_gaussian_prob(z, *self.encoder(x, y_one_hot))
            log_joint_probs += beta * log_p_x_z + log_p_y_z + log_p_z - log_q_z_x

        log_joint_probs /= K

        return log_joint_probs

    def train_model(self, data_loader, optimizer, num_epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train()  # Set the model to training mode
        for epoch in range(num_epochs):
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()  # Zero out any gradients from the previous iteration

                y = y.long()  # Ensure y is of type long
                y_one_hot = torch.zeros(y.shape[0], self.num_labels, device=y.device)
                y_one_hot.scatter_(1, y.unsqueeze(1), 1)

                # Now pass y_one_hot to the lower_bound function
                log_joint_probs = self.lower_bound(x, y_one_hot)

                # Compute the loss as the negative lower bound
                loss = -torch.mean(log_joint_probs)

                loss.backward()  # Compute the gradients
                optimizer.step()  # Update the model parameters

            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {loss.item()}")


    
    def predict(self, x, K=100):
        """
        Predict the class probabilities for the input x using importance sampling.
        """
        log_joint_probs = torch.zeros(x.shape[0], self.num_labels, device=x.device)

        for c in range(self.num_labels):
            y = torch.zeros(x.shape[0], self.num_labels, device=x.device)
            y[:, c] = 1

            z_samples = [self.latent_sample(*self.encoder(x, y)) for _ in range(K)]

            for z in z_samples:
                log_p_x_z = self.log_bernoulli_prob(self.decoder(z, y), x)
                log_p_z = self.log_gaussian_prob(z, torch.zeros_like(z), torch.zeros_like(z))
                log_p_y_z = self.log_softmax_prob(z, y)
                log_joint_probs[:, c] += (log_p_x_z + log_p_z + log_p_y_z).detach()
        log_joint_probs /= K

        y_pred = F.softmax(log_joint_probs, dim=1)

        return y_pred





