from torch import nn


class PointCloudEncoder(nn.Module):
    def __init__(self):
        super(PointCloudEncoder, self).__init__()

    def forward(self, data):
        assert False, "Not implemented in a base class"
