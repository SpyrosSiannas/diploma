import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base.encoder_base import PointCloudEncoder


class VanillaEncoder(PointCloudEncoder):
    def __init__(self, in_channels=3, hidden_dim=254, latent_dim=128, num_points=2048):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.conv1 = nn.Conv1d(in_channels, hidden_dim//2,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.pool1 = nn.MaxPool1d(num_points, None)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(latent_dim)
        self.bn6 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = x.view(-1, self.hidden_dim)
        x = torch.relu(self.bn4(self.fc1(x)))
        mean = self.fc2(x)
        log_var = self.fc3(x)
        return mean, log_var
