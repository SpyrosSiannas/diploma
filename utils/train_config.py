
from dataclasses import dataclass

from utils.network_cfg import NNConfig


@dataclass
class TrainConfig():
    model: object = None
    optimizer: object = None
    train_loader: object = None
    device: object = None
    nn_cfg: NNConfig = NNConfig()
    loss_fn: object = None
