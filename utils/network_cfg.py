
from dataclasses import dataclass


@dataclass
class NNConfig():
    in_channels: int = 3
    out_channels: int = 3
    hidden_dim: int = 64
    latent_dim: int = 32
    num_points: int = 2048
