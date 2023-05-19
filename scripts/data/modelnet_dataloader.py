import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader


def filter_sofa(example):
    return example["y"] == 7

def get_data_loader(batch_size=32, num_workers=4, shuffle=True):
    dataset = ModelNet(
        root="dataset",
        name="10",
        pre_filter=filter_sofa,
        train=True,
        pre_transform=T.NormalizeScale(),
        transform=T.SamplePoints(500),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader
