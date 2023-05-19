from scripts.trainloop import Trainer
# from utils.configs import get_default_config
from train_configs.gpccv2_cfg import get_config


def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    #train_config = get_default_config()
    train_config = get_config()
    trainer = Trainer(train_config)
    trainer.train()
    trainer.release()


if __name__ == "__main__":
    main()
