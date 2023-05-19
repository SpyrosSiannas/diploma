from torch import nn

from models.base.decoder_base import PointCloudDecoder
from models.base.encoder_base import PointCloudEncoder


class AE(nn.Module):
    def __init__(self, encoder: PointCloudEncoder, decoder: PointCloudDecoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        raise AssertionError("Not implemented in a base class")
