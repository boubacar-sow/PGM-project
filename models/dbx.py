from typing import Tuple 
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 10
latent_dims = 32
hidden_channels = 12
hidden_width = 64


class Decoder(nn.Module):
    def __init__(self,latent_dim: int, num_labels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.fc = nn.Linear(hidden_channels * 2 * 7 * 7, hidden_width)
        self.fc_mu = nn.Linear(
            in_features=hidden_width,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=hidden_width,
            out_features=latent_dim,
        )
        self.fc_y = nn.Sequential(nn.Linear(latent_dim, hidden_width),
                                  nn.ReLU(),
                                  nn.Linear(hidden_width,num_labels))
        

        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h = h.view(h.size(0), -1)  # Flatten the tensor
        h = self.activation(self.fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, self.fc_y(z)
