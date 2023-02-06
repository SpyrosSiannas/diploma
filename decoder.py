import torch.nn as nn
import torch.nn.functional as F
import torch


class PointCloudDecoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, k=20):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels),
        )
        self.conv = NNConv(hidden_channels, hidden_channels,
                           self.nn, aggr='mean', root_weight=False)
        self.k = k

    def forward(self, z, edge_index):
        z = z.view(-1, self.k * self.conv.out_channels)
        z = self.nn(z)
        z = self.conv(z, edge_index)
        return z
