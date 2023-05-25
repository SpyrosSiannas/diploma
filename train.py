import argparse
from pathlib import Path

import torch

from scripts.trainloop import Trainer

from train_configs.gpccv2_cfg import get_config

def parse_args():
    argparser = argparse.ArgumentParser(description="Train an ML model")

    argparser.add_argument(
        "--load_checkpoint",
        type=lambda x: Path(x).expanduser(),
        help="Path to checkpoint to load",
    )

    return argparser.parse_args()


def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    # train_config = get_default_config()
    torch.cuda.memory.set_per_process_memory_fraction(1.0)
    args = parse_args()
    train_config = get_config()
    if args.load_checkpoint:
        train_config.model.load_state_dict(torch.load(args.load_checkpoint))
    trainer = Trainer(train_config)
    trainer.train()
    trainer.release()


if __name__ == "__main__":
    main()
