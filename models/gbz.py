from typing import Mapping, Union, Optional
from pathlib import Path
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os
import pickle
from tqdm.notebook import tqdm
import pandas as pd
from typing import Callable, Optional, Tuple
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 10
latent_dims = 32
hidden_channels = 12
hidden_width = 64

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

        self.fc = nn.Linear(hidden_channels * 2 * 7 * 7 + num_labels, hidden_width)

        self.fc_mu = nn.Linear(
            in_features=hidden_width,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=hidden_width,
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
        )
        h = self.activation(self.fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, num_labels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        # MLP for p(y|z)
        self.fc_py_z = nn.Linear(latent_dim, 500)
        self.fc_py_z1 = nn.Linear(500, num_labels)

        # MLP for p(x|z)
        self.fc_px_z = nn.Linear(latent_dim, hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv1 = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the MLP to get the distribution over labels
        y = self.activation(self.fc_py_z(z))
        y = self.fc_py_z1(y)

        h = self.activation(self.fc_px_z(z))
        h = h.view(h.size(0), self.hidden_channels * 2, 7, 7)  # Reshape the tensor

        # Use the rest of the decoder to get the reconstructed image
        h = self.activation(self.conv2(h))
        x_recon = torch.sigmoid(self.conv1(h))

        return x_recon, y