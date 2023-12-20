
from typing import Tuple 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Decoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, num_labels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        # MLP for p(y|z)
        self.fc_py_z = nn.Linear(latent_dim, num_labels)

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
        y = F.softmax(self.fc_py_z(z), dim=1)

        # Use the MLP to get the initial tensor for image reconstruction
        h = self.activation(self.fc_px_z(z))
        h = h.view(h.size(0), self.hidden_channels * 2, 7, 7)  # Reshape the tensor

        # Use the rest of the decoder to get the reconstructed image
        h = self.activation(self.conv2(h))
        x_recon = torch.sigmoid(self.conv1(h))

        return x_recon, y