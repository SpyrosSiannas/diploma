import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


class PointCloudEncoder(nn.Module):
    def __init__(self, point_dim=3, hidden_dim=64, latent_dim=32):
        super().__init__()
        # Encoder architecture
        # input_dim = 3 (x, y, z)
        # 1d Conv filter with hidden_dim = 64

        # fully connected layer with latent_dim = 32
        # second fully connected layer with latent_dim = 32

        self.conv = nn.Conv1d(in_channels=point_dim,
                              out_channels=hidden_dim, kernel_size=1)
        self.fc1 = nn.Linear(1, latent_dim)
        self.fc2 = nn.Linear(1, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.max(x, 1, keepdim=True)[0]
        mean = self.fc1(x)
        logvar = self.fc2(x)
        return mean, logvar
