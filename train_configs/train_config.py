from dataclasses import dataclass

@dataclass
class TrainConfig:
    model: object = None
    optimizer: object = None
    train_loader: object = None
    device: object = None
    nn_cfg  = None
    loss_fn: object = None
    num_epochs: int = 500