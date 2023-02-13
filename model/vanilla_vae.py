import torch
import torch.nn as nn

from model.base.vae_base import VAE
from model.encoders_decoders.vanilla_decoder import VanillaDecoder
from model.encoders_decoders.vanilla_encoder import VanillaEncoder


class VanillaVAE(VAE):
    def __init__(self):
        super().__init__(VanillaEncoder(), VanillaDecoder())

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
