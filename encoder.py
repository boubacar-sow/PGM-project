from typing import Tuple 
import torch
import torch.nn as nn
import torch.nn.functional as F
 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, num_labels: int) -> None:
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

        self.fc_mu = nn.Linear(
            in_features=hidden_channels * 2 * 7 * 7 + num_labels,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=hidden_channels * 2 * 7 * 7 + num_labels,
            out_features=latent_dim,
        )

        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h = h.view(h.size(0), -1)  # Flatten the tensor
        h = torch.cat(
            (h, y), dim=1
        )  # Now you can concatenate h and y along dimension 1
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
