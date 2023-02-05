import torch.nn as nn
import torch.nn.functional as F


class PointCloudDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PointCloudDecoder, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
