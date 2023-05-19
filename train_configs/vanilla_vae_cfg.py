import torch
from torch import optim

from models.vanilla.model_cfg import NNConfig
from models.vanilla.vanilla_vae import VanillaVAE
from scripts.data.modelnet_dataloader import get_data_loader
from scripts.loss import custom_loss
from train_configs.train_config import TrainConfig


def get_config() -> TrainConfig:
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
        nn_cfg= NNConfig(),
    )
    return train_config