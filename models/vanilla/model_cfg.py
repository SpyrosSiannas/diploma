from dataclasses import dataclass

@dataclass
class NNConfig:
    point_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 64
    num_points: int = 2048
    num_epochs: int = 500