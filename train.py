from model.vae import VAE
from model.decoder import PointCloudDecoder
from model.encoder import PointCloudEncoder
import torch.optim as optim
import torch
from utils.configs import TrainConfig
from utils.loss import MSE_KLD
from utils.trainloop import Trainer
from utils.dataset import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    encoder = PointCloudEncoder().to(device)
    decoder = PointCloudDecoder().to(device)
    model = VAE(encoder, decoder).to(device)  # initialize the model
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    train_config = TrainConfig(
        model=model, optimizer=optimizer, train_loader=get_data_loader(), device=device, loss_fn=MSE_KLD)

    trainer = Trainer(train_config)
    trainer.train(train_config.nn_cfg.num_epochs)
    trainer.release()


if __name__ == "__main__":
    main()
