from vae import VAE
from decoder import PointCloudDecoder
from encoder import PointCloudEncoder
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet
from torch.nn import functional as F
import torch


def loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def main() -> None:
    # Load the ModelNet10 dataset
    dataset = ModelNet(root='dataset', name='10', train=True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the point cloud encoder and decoder networks
    in_channels = 3
    hidden_channels = 64
    latent_channels = 32
    out_channels = 3
    encoder = PointCloudEncoder(in_channels, hidden_channels, latent_channels)
    decoder = PointCloudDecoder(latent_channels, hidden_channels, out_channels)

    model = VAE(latent_channels, encoder, decoder)
    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = ModelNet(root='dataset', name='10', train=True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data.x)
            loss = loss_fn(recon_batch, data.x, mu, logvar)
            loss.backward()


if __name__ == "__main__":
    main()
