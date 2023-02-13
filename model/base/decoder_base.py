from torch import nn


class PointCloudDecoder(nn.Module):
    def __init__(self):
        super(PointCloudDecoder, self).__init__()

    def forward(self, latent_vec):
        assert False, "Not implemented in a base class"
