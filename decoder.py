import torch
import torch.nn as nn


class PointCloudDecoder(nn.Module):
    def __init__(self, output_dim=3, hidden_dim=128, latent_dim=64, num_points=2048):
        super(PointCloudDecoder, self).__init__()
        self.output_size = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.fc1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim*4)
        self.fc2 = nn.Linear(in_features=hidden_dim*4,
                             out_features=num_points*output_dim)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, self.num_points, self.output_size)
        return x
