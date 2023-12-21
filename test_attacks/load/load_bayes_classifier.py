

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/content/PGM-project')
sys.path.append('/content/PGM-project')
sys.path.append('/content/PGM-project')
sys.path.append('../')
sys.path.append('../test_attacks')
sys.path.append('/home/boubacar/Documents/PGM project/PGM-project/test_attacks')
sys.path.append('./PGM-project/test_attacks')
sys.path.append('/home/boubacar/Documents/PGM project/PGM-project/models')


import matplotlib.pyplot as plt
# Assuming you have already imported torch and torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BayesModel(nn.Module):
    def __init__(self, model_name, data_name, hidden_channels, dimZ=64):
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
        
        if model_name == 'GBZ':
            from models.gbz import Encoder, Decoder
        elif model_name == 'DBX':
            from models.dbx import Encoder, Decoder
        else:
            raise ValueError('model type not recognised')
        
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
    
    def lowerbound_F(self, x, y, enc, dec, K=1, IS=False,  beta=1.0, use_mean=False):

        z, logq = self.encoding(enc,x, y, K, use_mean)
        log_prior_z = self.log_gaussian_prob(z, torch.zeros_like(z), torch.zeros_like(z))
        pxz, pyz = dec(torch.flatten(z,end_dim=1))
        pxz = torch.stack(torch.tensor_split(pxz,K,dim=0),dim=0)
        pyz = torch.stack(torch.tensor_split(pyz,K,dim=0),dim=0)
        ind = list(range(2, len(x.shape)+1))
        print("pxz shape: ", pxz.shape)
        print("x shape: ", x.unsqueeze(0).shape)
        logp = -torch.sum((x.unsqueeze(0) - pxz)**2, dim=ind)
        logit_y = F.softmax(pyz,dim=2)

        y_rep = torch.stack([y  for i in range(K)],dim=0)
        log_pyz = -F.cross_entropy(logit_y.flatten(end_dim=1), y_rep.flatten(end_dim=1),
                                reduction='none').reshape(y_rep.shape[:-1])
        bound = logp * beta + log_pyz + (log_prior_z - logq)
        if IS and K > 1 and use_mean==False:
            bound = torch.logsumexp(bound,dim=0) - np.log(float(K))
        return bound.squeeze()
    
    def encoding(self, enc, x, y, K, use_mean=False):
        mu_qz, log_sig_qz = enc(x, y)
        if use_mean:
            return mu_qz.unsqueeze(0), self.log_gaussian_prob(mu_qz.unsqueeze(0), mu_qz.unsqueeze(0), log_sig_qz.unsqueeze(0))
        ph = torch.zeros([K]+list(mu_qz.shape))
        norm_sample = torch.normal(ph).to(device)
        samples = mu_qz+torch.unsqueeze(log_sig_qz.exp(), dim=0)*norm_sample
        logq = self.log_gaussian_prob(samples, mu_qz.unsqueeze(0), log_sig_qz.unsqueeze(0))
        return samples, logq
    
    def bayes_classifier(self, lowerbound, x, enc, dec, dimY,  K = 1, beta=1.0, use_mean=False):
        N = x.shape[0]
        logpxy = []
        for i in range(dimY):
            y = torch.zeros([N, dimY]).to(device)
            y[:, i] = 1
            bound = lowerbound(x, y,enc, dec, K,
                            IS=True, beta=beta, use_mean=use_mean)
            logpxy.append(torch.unsqueeze(bound, 1))
        logpxy = torch.concat(logpxy, 1)
        pyx = F.softmax(logpxy,dim=1)
        return pyx


    def train_model(self, model_name, data_loader, optimizer, num_epochs=30, variational_beta=1.0):
        if model_name == 'GBZ':
            self.train_gbz(data_loader, optimizer, num_epochs, variational_beta)
        elif model_name == 'DBX':
            self.train_dbx(data_loader, optimizer, num_epochs, variational_beta)
        else:
            raise ValueError('model type not recognised')    
        
    def train_gbz(self, data_loader, optimizer, num_epochs=30, variational_beta=1.0):
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
                # print("trues: ", y[:20])
                # print("preds: ", torch.max(y_pred, 1)[1][:20])
                label_loss = F.cross_entropy(y_pred, y, reduction='sum')  # Use y_pred directly
                kldivergence = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
                loss = recon_loss + label_loss + variational_beta * kldivergence

                loss.backward()  # Compute the gradients
                optimizer.step()  # Update the model parameters

            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {loss.item()}")
            
    def train_dbx(self, data_loader, optimizer, num_epochs=30, variational_beta=1.0, K=100):
        pass
    
    def predict(self, model_name, x, K=8, batch_size=10):
      """
      Predict the class probabilities for the input x using importance sampling.
      """
      if model_name == 'GBZ':
          return self.predict_GBZ(x, K)
      elif model_name == 'DBX':
          return self.predict_DBX(x, K)
      else:
          raise ValueError('model type not recognised')

    def predict_GBZ(self, x, K=10):
        """
        Predict the class probabilities for the input x using importance sampling.
        """
        return self.bayes_classifier(self.lowerbound_F, x, self.encoder, self.decoder, self.num_labels, K=K, beta=1.0, use_mean=True)
      
    def predict_DBX(self, x, K=10, batch_size=10):
        pass





