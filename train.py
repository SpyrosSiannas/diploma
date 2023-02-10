from model.vae import VAE
from model.decoder import PointCloudDecoder
from model.encoder import PointCloudEncoder
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
import torch
import torch_geometric.transforms as T
from utils.network_cfg import NNConfig as cfg
from utils.train_config import TrainConfig
from utils.loss import MSE_KLD
from utils.trainloop import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loader(batch_size=32, num_workers=4, shuffle=True):

    dataset = ModelNet(root='dataset', name='10', train=True, pre_transform=T.NormalizeScale(),
                       transform=T.SamplePoints(cfg.num_points))
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return data_loader


def main() -> None:
    # Initialize the point cloud encoder and decoder networks

    encoder = PointCloudEncoder().to(device)
    decoder = PointCloudDecoder().to(device)
    model = VAE(encoder, decoder).to(device)  # initialize the model
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))

    train_config = TrainConfig(
        model=model, optimizer=optimizer, train_loader=get_data_loader(), device=device, loss_fn=MSE_KLD)

    trainer = Trainer(train_config)
    trainer.train(cfg.num_epochs)


if __name__ == "__main__":
    main()
