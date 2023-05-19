import torch
import torch.nn as nn

from models.base.decoder_base import PointCloudDecoder
from models.base.encoder_base import PointCloudEncoder
from models.base.vae_base import VAE


class EncoderNet(PointCloudEncoder):
    def __init__(
        self, in_channels=3, hidden_dim=2 * 254, latent_dim=128, num_points=2048
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.pool1 = nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(latent_dim)
        self.bn6 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        # out : (batch_size, hidden_dim, num_points)
        x = self.pool1(x)
        # out: (batch_size, hidden_dim, 1)
        x = x.view(-1, self.hidden_dim)
        # out: (batch_size, hidden_dim)
        x = torch.relu(self.bn4(self.fc1(x)))
        mean = self.fc2(x)
        log_var = self.fc3(x)
        return mean, log_var


class DecoderNet(PointCloudDecoder):
    def __init__(self, output_dim=3, hidden_dim=254, latent_dim=128, num_points=2048):
        super(PointCloudDecoder, self).__init__()
        self.output_size = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_points = num_points

        self.fc1 = nn.Linear(in_features=latent_dim, out_features=hidden_dim * 4)
        self.fc2 = nn.Linear(in_features=hidden_dim * 4, out_features=hidden_dim * 8)
        self.fc3 = nn.Linear(
            in_features=hidden_dim * 8, out_features=output_dim * num_points
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 8)

    def forward(self, z):
        x = torch.relu(self.bn1(self.fc1(z)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, self.num_points, self.output_size)
        return x


class VanillaVAE(VAE):
    def __init__(self):
        super().__init__(EncoderNet(), DecoderNet())

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
