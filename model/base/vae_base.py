from torch import nn
from model.base.decoder_base import PointCloudDecoder
from model.base.encoder_base import PointCloudEncoder


class VAE(nn.Module):
    def __init__(self, encoder: PointCloudEncoder, decoder: PointCloudDecoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        assert False, "Not implemented in a base class"

    def reparameterize(self, mu, logvar):
        assert False, "Not implemented in a base class"
