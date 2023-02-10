from tqdm import tqdm
from utils.loss import MSE_KLD
from torch_geometric.data.batch import Batch


class Trainer():
    def __init__(self, train_config):
        self.model = train_config.model
        self.optimizer = train_config.optimizer
        self.train_loader = train_config.train_loader
        self.device = train_config.device
        self.nn_cfg = train_config.nn_cfg

    def train(self, num_epochs: int) -> None:
        self.model.train()
        for epoch in range(num_epochs):
            self.__epoch(epoch)

    def __epoch(self, epoch: int) -> None:
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                loss = self.__trainloop(data)
                tepoch.set_postfix(loss=loss)

    def __trainloop(self, data: Batch) -> any:
        self.optimizer.zero_grad()
        # move the input data to the GPU
        pos_tr = data.pos.transpose(1, 0).to(self.device)
        pos_reshaped = pos_tr.reshape(-1,
                                      3, self.nn_cfg.num_points).to(self.device)
        # forward pass through the model
        recon_x, mu, log_var = self.model(pos_reshaped)
        # compute the loss
        loss = MSE_KLD(
            recon_x, pos_tr.reshape(-1, self.nn_cfg.num_points, 3), mu, log_var)
        loss.backward()  # compute the gradients
        self.optimizer.step()  # update the parameters
        return loss.item()
