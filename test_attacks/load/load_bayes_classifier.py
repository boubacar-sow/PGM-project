

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/content/PGM-project')
import matplotlib.pyplot as plt
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
        x_recon, y_pred = self.decoder(latent)
        return x_recon, y_pred, latent_mu, latent_logvar


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
            log_p_x_z = self.log_bernoulli_prob(self.decoder(z)[0], x)
            log_p_z = self.log_gaussian_prob(z, torch.zeros_like(z), torch.zeros_like(z))
            log_p_y_z = self.log_softmax_prob(z, y_one_hot)
            log_q_z_x = self.log_gaussian_prob(z, *self.encoder(x, y_one_hot))
            log_joint_probs += beta * log_p_x_z + log_p_y_z + log_p_z - log_q_z_x

        log_joint_probs /= K

        return log_joint_probs


    def train_model(self, data_loader, optimizer, num_epochs=30, variational_beta=1.0, K=100):
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

                # Forward pass
                x_recon, y_pred, latent_mu, latent_logvar = self.forward(x, y_one_hot)
               
                recon_loss = F.mse_loss(x_recon, x, reduction='sum')
                label_loss = F.cross_entropy(y_pred, y, reduction='sum')  # Use y_pred directly
                kldivergence = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
                loss = recon_loss + label_loss + variational_beta * kldivergence

                loss.backward()  # Compute the gradients
                optimizer.step()  # Update the model parameters

            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {loss.item()}")

    def display_images(self, original, reconstructed, epoch):
        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(30, 6))

        # Display original images
        for i, ax in enumerate(axes[0, :]):
            ax.imshow(original[i].reshape(28, 28).cpu(), cmap='gray')
            ax.axis('off')

        # Display reconstructed images
        for i, ax in enumerate(axes[1, :]):
            ax.imshow(reconstructed[i].reshape(28, 28).cpu(), cmap='gray')
            ax.axis('off')

        # Save the figure
        plt.savefig(f'images_epoch_{epoch}.png')
        plt.close(fig)  # Close the figure to free up memory



    
    def predict(self, x, K=100, batch_size=32):
      """
      Predict the class probabilities for the input x using importance sampling.
      """
      num_batches = x.shape[0] // batch_size
      y_pred = torch.zeros(x.shape[0], self.num_labels, device=x.device)

      for i in range(num_batches):
          start = i * batch_size
          end = start + batch_size
          x_batch = x[start:end]

          log_joint_probs = torch.zeros(x_batch.shape[0], self.num_labels, device=x.device)

          for c in range(self.num_labels):
              y = torch.zeros(x_batch.shape[0], self.num_labels, device=x.device)
              y[:, c] = 1

              z_samples = [self.latent_sample(*self.encoder(x_batch, y)) for _ in range(K)]

              for z in z_samples:
                  # Reconstruct input and get predicted labels
                  x_recon, y_pred_batch = self.decoder(z)
                  # Compute MSE loss (replace log_bernoulli_prob)
                  log_p_x_z = F.mse_loss(x_recon, x_batch, reduction='sum')

                  # Other log probabilities (if needed)
                  log_p_z = self.log_gaussian_prob(z, torch.zeros_like(z), torch.zeros_like(z))
                  log_p_y_z = torch.log(y_pred_batch[:, c] + 1e-9)
                  log_q_z_x = self.log_gaussian_prob(z, *self.encoder(x_batch, y))
                  
                  log_joint_probs[:, c] +=  log_p_x_z + log_p_y_z + log_p_z - log_q_z_x
                  print(log_joint_probs[:, c])

          log_joint_probs /= K
          y_pred[start:end] = F.softmax(log_joint_probs, dim=1)

      return y_pred







