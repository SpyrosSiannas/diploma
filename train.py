from vae import VAE
from decoder import PointCloudDecoder
from encoder import PointCloudEncoder
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch.nn import functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    in_channels = 3
    latent_channels = 32
    out_channels = 3
    encoder = PointCloudEncoder().to(device)
    decoder = PointCloudDecoder().to(device)

    # Define the loss function and optimizer
    dataset = ModelNet(root='dataset', name='10', train=True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_epochs = 1000

    model = VAE(encoder, decoder).to(device)  # initialize the model
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for data in data_loader:
            # move the input data to the GPU
            pos = data.pos.transpose(1, 0).to(device)
            optimizer.zero_grad()  # reset the gradients
            # forward pass through the model
            recon_x, mu, log_var = model(pos)
            recon_loss = F.mse_loss(recon_x, pos)  # reconstruction loss
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss  # overall loss
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameters
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch +
              1, num_epochs, loss.item()))


if __name__ == "__main__":
    main()
