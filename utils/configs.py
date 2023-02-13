
from dataclasses import dataclass


@dataclass
class NNConfig():
    point_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 64
    num_points: int = 2048
    num_epochs: int = 500


@dataclass
class TrainConfig():
    model: object = None
    optimizer: object = None
    train_loader: object = None
    device: object = None
    nn_cfg: NNConfig = NNConfig()
    loss_fn: object = None
    num_epochs: int = 500
