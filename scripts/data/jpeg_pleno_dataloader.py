import MinkowskiEngine as ME
import numpy as np
import torch

from scripts.utils import read_ply_ascii_geo


class JPEGPlenoDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coordinates, features = self.cache[idx]
        else:
            if filedir.endswith(".ply"):
                coordinates = read_ply_ascii_geo(filedir)
            else:
                raise Exception(f"Unsupported file format: {filedir}")
            features = np.expand_dims(np.ones(coordinates.shape[0]), 1).astype("int")
            self.cache[idx] = (coordinates, features)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if (
                cache_percent > 0
                and cache_percent % 10 == 0
                and cache_percent != self.last_cache_percent
            ):
                self.last_cache_percent = cache_percent

        features = features.astype("float32")
        return (coordinates, features)


def collate_pointcloud(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError("No data in the batch")
    coords, feats = list(zip(*list_data, strict=False))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)
    return coords_batch, feats_batch


def make_jpeg_pleno_loader(
    file_list,
    batch_size=1,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_pointcloud,
):
    dataset = JPEGPlenoDataset(file_list)
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": False,
        "drop_last": False,
        "shuffle": shuffle,
    }
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader
