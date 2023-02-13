from model.vanilla_vae import VanillaVAE
import torch.optim as optim
import torch
from utils.configs import TrainConfig
from utils.loss import MSE_KLD
from utils.dataset import get_data_loader


def get_default_config() -> TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaVAE().to(device)  # initialize the model
    # initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    train_config = TrainConfig(
        model=model, optimizer=optimizer, train_loader=get_data_loader(), device=device, loss_fn=MSE_KLD)
    return train_config
