import torch.nn as nn
from torch_geometric.nn import NNConv, global_max_pool
import torch


class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PointCloudEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.conv1 = NNConv(128, 256, self.nn)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.fc1(x))
        x = self.conv1(x, edge_index)
        x = torch.relu(self.fc2(x))
        x = global_max_pool(x, batch)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
