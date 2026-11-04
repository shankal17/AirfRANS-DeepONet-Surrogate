"""DeepONet for surface quantities.

The trunk gets random-free Fourier features of the stretched chord coordinate u so we can
represent the sharp leading-edge suction peak and the jump at the trailing edge.
"""
from __future__ import annotations

import math
import torch

from torch import nn

def mlp(num_input_features, hidden_layer_width, num_hidden_layers, num_output_channels):
    """Fully connected network with GELU activations and a linear output layer.

    Args:
        num_input_features (int): Num input features.
        hidden_layer_width (int): Hidden layer width.
        num_hidden_layers (int): Num hidden layers.
        num_output_channels (int): Num output features.

    Returns:
        nn.Sequential: The network.
    """

    layers, in_features = [], num_input_features
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(in_features, hidden_layer_width), nn.GELU()]
        in_features = hidden_layer_width
    layers.append(nn.Linear(in_features, num_output_channels))

    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Branch-trunk operator network with per-channel basis coefficients.

    Args:
        branch_dim: Branch input features Db.
        trunk_dim: Trunk input features Dt, before the Fourier features are appended.
        num_output_channels: Num output channels.
        hidden_layer_width: Hidden layer width of both MLPs.
        num_hidden_layers: Num of hidden layers in both MLPs.
        n_basis: Num of basis functions per output channel.
        n_fourier: Num of fourier frequencies (pi, 2 pi, ...) of the first trunk column.

    Attributes:
        branch (nn.Sequential): Maps conditions to num_output_channels * n_basis coefficients.
        trunk (nn.Sequential): Maps embedded points to num_output_channels * n_basis basis values.
        bias (nn.Parameter): Per-channel output bias, shape (num_output_channels,).
    """

    def __init__(self, branch_dim: int, trunk_dim: int, num_output_channels: int = 3,
                 hidden_layer_width: int = 256, num_hidden_layers: int = 4, n_basis: int = 128, n_fourier: int = 16):
        super().__init__()
        self.num_output_channels, self.n_basis, self.n_fourier, self.trunk_dim = num_output_channels, n_basis, n_fourier, trunk_dim
        self.register_buffer("freqs", math.pi * torch.arange(1, n_fourier + 1, dtype=torch.float32))
        self.branch = mlp(branch_dim, hidden_layer_width, num_hidden_layers, num_output_channels * n_basis)
        self.trunk = mlp(trunk_dim + 2 * n_fourier, hidden_layer_width, num_hidden_layers, num_output_channels * n_basis)
        self.bias = nn.Parameter(torch.zeros(num_output_channels))

    def trunk_embed(self, trunk):
        """Append sin and cos Fourier features of the first trunk column (u in [-1, 1]).

        Args:
            trunk (Tensor): Trunk inputs, shape (..., Dt).

        Returns:
            Tensor: Embedded inputs, shape (..., Dt + 2 * n_fourier).
        """

        phase = trunk[..., :1] * self.freqs  # first trunk channel is u in [-1, 1]

        return torch.cat([trunk, torch.sin(phase), torch.cos(phase)], -1)

    def forward(self, branch, trunk):
        """Evaluate the operator at every trunk point of every case.

        Args:
            branch (Tensor): Normalised branch inputs, shape (B, Db).
            trunk (Tensor): Trunk inputs, shape (B, N, Dt).

        Returns:
            Tensor: Prediction in normalised target units, shape (B, N, num_output_channels).
        """

        branch_coeffs = self.branch(branch).view(*branch.shape[:-1], 1, self.num_output_channels, self.n_basis)
        trunk_basis = self.trunk(self.trunk_embed(trunk)).view(*trunk.shape[:-1], self.num_output_channels, self.n_basis)
        
        return (branch_coeffs * trunk_basis).sum(-1) / math.sqrt(self.n_basis) + self.bias
