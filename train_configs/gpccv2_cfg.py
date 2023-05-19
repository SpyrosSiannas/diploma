from train_configs.train_config import TrainConfig
import torch
import torch.optim as optim
from models.interframe_model.interframe_ae import InterFrameAE
from scripts.data.jpeg_pleno_dataloader import make_jpeg_pleno_loader
from scripts.loss import get_gpccv2_loss
import glob

def get_config() -> TrainConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterFrameAE().to(device)
    optimizer = optim.Adam(
        model.parameters(), weight_decay=1e-4, lr=8e-4, betas=(0.9, 0.999)
    )
    # Wrong! Must change
    filelist = glob.glob("dataset/jpeg-pleno/**/Ply/*.ply")
    if len(filelist) == 0:
        raise Exception("No files found in the dataset")
    dataset_list = sorted(filelist[:750])
    train_config = TrainConfig(
        model=model,
        optimizer=optimizer,
        train_loader=make_jpeg_pleno_loader(dataset_list),
        device=device,
        loss_fn=get_gpccv2_loss,
    )
    return train_config
