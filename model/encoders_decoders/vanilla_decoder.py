import torch
import torch.nn as nn

from model.base.decoder_base import PointCloudDecoder


class VanillaDecoder(PointCloudDecoder):
    def __init__(self, output_dim=3, hidden_dim=254, latent_dim=128, num_points=2048):
        super(PointCloudDecoder, self).__init__()
        self.output_size = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.fc1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim*4)
        self.fc2 = nn.Linear(in_features=hidden_dim*4,
                             out_features=hidden_dim*8)
        self.fc3 = nn.Linear(in_features=hidden_dim*8,
                             out_features=output_dim*num_points)

        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = nn.BatchNorm1d(hidden_dim*8)

    def forward(self, z):
        x = torch.relu(self.bn1(self.fc1(z)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, self.num_points, self.output_size)
        return x
