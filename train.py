from scripts.trainloop import Trainer
# from utils.configs import get_default_config
from train_configs.gpccv2_cfg import get_config
import argparse
from pathlib import Path
import torch

def parse_args():
    argparser = argparse.ArgumentParser(description="Train an ML model")

    argparser.add_argument("--load_checkpoint", type=lambda x: Path(x).expanduser(), help="Path to checkpoint to load")

    return argparser.parse_args()

def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    #train_config = get_default_config()
    args = parse_args()
    train_config = get_config()
    if args.load_checkpoint:
        train_config.model.load_state_dict(torch.load(args.load_checkpoint))
    trainer = Trainer(train_config)
    trainer.validate()
    trainer.release()


if __name__ == "__main__":
    main()
