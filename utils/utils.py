import torch
import torch.optim as optim

from model.vanilla_vae import VanillaVAE
from utils.configs import TrainConfig
from utils.dataset import get_data_loader
from utils.loss import custom_loss


def get_default_config() -> TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaVAE().to(device)  # initialize the model
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    train_config = TrainConfig(
        model=model,
        optimizer=optimizer,
        train_loader=get_data_loader(),
        device=device,
        loss_fn=custom_loss,
    )
    return train_config
