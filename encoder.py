import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointConv, global_max_pool


class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(PointCloudEncoder, self).__init__()

        self.conv = PointConv(
            in_channels, hidden_channels, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels//2)
        self.fc2 = nn.Linear(hidden_channels//2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = global_max_pool(x, batch=None)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
