from utils.trainloop import Trainer
from utils.utils import get_default_config


def main() -> None:
    # Initialize the point cloud encoder and decoder networks
    train_config = get_default_config()
    trainer = Trainer(train_config)
    trainer.train()
    trainer.release()


if __name__ == "__main__":
    main()
