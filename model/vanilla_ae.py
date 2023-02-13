import torch.nn as nn


class VanillaAutoEncoder():
    def __init__(self, encoder, decoder):
        super(VanillaAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        latent_vec = self.encoder(data)
        x_recon = self.decoder(latent_vec)
        return latent_vec, x_recon
