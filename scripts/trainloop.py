import os
from datetime import datetime

import MinkowskiEngine as ME
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data.batch import Batch
from tqdm import tqdm


class Trainer:
    def __init__(self, train_config, cleanup_logs=True):
        self.model = train_config.model
        self.optimizer = train_config.optimizer
        self.train_loader = train_config.train_loader
        self.device = train_config.device
        self.nn_cfg = train_config.nn_cfg
        self.num_epochs = train_config.num_epochs
        self.loss_fn = train_config.loss_fn
        self.writer = SummaryWriter("logs")
        self.checkpoints_path = train_config.checkpoints_dir
        self.gradient_accumulations = 4
        if not self.checkpoints_path.exists():
            os.makedirs(self.checkpoints_path)
        if cleanup_logs:
            os.system("rm -rf logs/*")

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.num_epochs):
            self.__epoch(epoch)

    def __epoch(self, epoch: int) -> None:
        batch_idx = 0
        loss = 0
        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            for data in tepoch:
                batch_idx += 1
                loss = self.__trainloop(data)
                (loss / self.gradient_accumulations).backward()
                if (
                    batch_idx % self.gradient_accumulations == 0
                    or batch_idx == len(self.train_loader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    torch.cuda.empty_cache()
                tepoch.set_postfix(loss=loss.item())
            now = datetime.now()
            torch.save(
                self.model.state_dict(),
                self.checkpoints_path
                / f"{self.model.__class__.__name__}_epoch_{epoch + 1}_{now.strftime('%Y%m%d%H%M')}.pth",
            )
                # self.writer.add_mesh(
                #     "original",
                #     original[2, :, :].unsqueeze(0).detach(),
                #     global_step=epoch + 1,
                # )
                # self.writer.add_mesh(
                #     "point_cloud",
                #     recon_x.transpose(2, 1)[2, :, :].unsqueeze(0).detach(),
                #     global_step=epoch + 1,
                # )

    def __trainloop(self, data: Batch):
        self.optimizer.zero_grad()
        #        pos_reshaped = data.pos.reshape(-1, self.nn_cfg.num_points, 3).to(self.device)
        # forward pass through the model
        x = ME.SparseTensor(coordinates=data[0], features=data[1], device=self.device)
        model_out = self.model(x, training=True)
        # compute the loss
        loss = self.loss_fn(model_out, x)
        return loss

    def release(self):
        self.writer.close()
