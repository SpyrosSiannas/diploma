from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from utils.configs import NNConfig as cfg


def get_data_loader(batch_size=32, num_workers=4, shuffle=True):

    dataset = ModelNet(root='dataset', name='10', train=True, pre_transform=T.NormalizeScale(),
                       transform=T.SamplePoints(cfg.num_points))
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return data_loader
