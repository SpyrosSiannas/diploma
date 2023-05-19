from torch import nn

from models.base.decoder_base import PointCloudDecoder
from models.base.encoder_base import PointCloudEncoder


class VAE(nn.Module):
    def __init__(self, encoder: PointCloudEncoder, decoder: PointCloudDecoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        raise AssertionError("Not implemented in a base class")

    def reparameterize(self, mu, logvar):
        raise AssertionError("Not implemented in a base class")
