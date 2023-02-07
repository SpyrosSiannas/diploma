import torch
import torch.nn as nn


class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=3):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
