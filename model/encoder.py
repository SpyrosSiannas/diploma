import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, latent_dim=64, num_points=2048):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.conv1 = nn.Conv1d(in_channels, hidden_dim,
                               kernel_size=1)
        self.pool1 = nn.MaxPool1d(num_points, None)

        self.fc1 = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, self.hidden_dim)
        mean = torch.relu(self.fc1(x))
        log_var = torch.relu(self.fc2(x))
        return mean, log_var
