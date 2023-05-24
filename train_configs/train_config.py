from dataclasses import dataclass
from pathlib import Path
@dataclass
class TrainConfig:
    model: object = None
    optimizer: object = None
    train_loader: object = None
    val_loader: object = None
    device: object = None
    nn_cfg  = None
    loss_fn: object = None
    num_epochs: int = 50
    checkpoints_dir: Path = Path("checkpoints")