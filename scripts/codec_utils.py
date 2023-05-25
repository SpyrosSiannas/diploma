import os
import random
import string
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from models.interframe_model.entropy_bottleneck import EntropyBottleneck
from scripts.gpcc import gpcc_decode, gpcc_encode
from scripts.utils import array2vector, read_ply_ascii_geo, write_ply_ascii_geo


def sort_spare_tensor(sparse_tensor):
    """Sort points in sparse tensor according to their coordinates."""
    indices_sort = np.argsort(
        array2vector(sparse_tensor.C.cpu(), sparse_tensor.C.cpu().max() + 1)
    )
    sparse_tensor_sort = ME.SparseTensor(
        features=sparse_tensor.F[indices_sort],
        coordinates=sparse_tensor.C[indices_sort],
        tensor_stride=sparse_tensor.tensor_stride[0],
        device=sparse_tensor.device,
    )

    return sparse_tensor_sort


def load_sparse_tensor(filedir, device):
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    feats = torch.ones((len(coords), 1)).float()
    # coords, feats = ME.utils.sparse_quantize(coordinates=coords, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device=device
    )

    return x


class FeatureCoder:
    """Class that uses a learned entropy model to compress and decompress
    the feature tensor of a point cloud in a lossy manner.
    """

    def __init__(
        self,
        entropy_model: EntropyBottleneck,
    ) -> None:
        self.entropy_model = entropy_model.cpu()

    def encode(self, features, out_path, filename):
        strings, min_v, max_v = self.entropy_model.compress(features.cpu())
        shape = features.shape
        with open(out_path / (filename + "_Feats.bin"), "wb") as f:
            f.write(strings)
        with open(out_path / (filename + "_Header.bin"), "wb") as f:
            f.write(np.array(shape, dtype=np.int32).tobytes())
            f.write(np.array(len(min_v), dtype=np.int8).tobytes())
            f.write(np.array(min_v, dtype=np.float32).tobytes())
            f.write(np.array(max_v, dtype=np.float32).tobytes())
        return

    def decode(self, in_path, filename):
        with open(in_path / (filename + "_Header.bin"), "rb") as f:
            shape = np.frombuffer(f.read(4 * 2), dtype=np.int32)
            min_v_len = np.frombuffer(f.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(f.read(4 * min_v_len), dtype=np.float32)[0]
            max_v = np.frombuffer(f.read(4 * min_v_len), dtype=np.float32)[0]
        with open(in_path / (filename + "_Feats.bin"), "rb") as f:
            strings = f.read()
        features = self.entropy_model.decompress(
            strings, min_v, max_v, shape, shape[-1]
        )
        return features


class CoordinateCoder:
    """Class that uses the GPCC model to compress and decompress the coordinates
    of a point cloud in a lossy manner.
    """

    @staticmethod
    def encode(coordinates, out_path, filename: str):
        temp_file = out_path / (
            "".join(random.choices(string.ascii_lowercase, k=7)) + ".ply"
        )
        output_file = out_path / (filename + ".bin")
        coords = coordinates.numpy().astype("int")

        write_ply_ascii_geo(temp_file, coords)
        gpcc_encode(str(temp_file), str(output_file))

        os.remove(temp_file)

    @staticmethod
    def decode(in_path, filename: str):
        input_file = in_path / (filename + ".bin")
        temp_file = in_path / (filename.split(".")[0] + ".ply")

        gpcc_decode(str(input_file), str(temp_file))
        coords = read_ply_ascii_geo(temp_file)
        os.remove(temp_file)
        return coords


class InterframeCodec:
    r"""Class that takes an interframe AutoEncoder as input and
    performs compression and decompression on point clouds using
    a feature coder and a coordinates coder..shape[0])+'\n').
    """

    def __init__(self, model):
        self.model = model
        self.feature_coder = FeatureCoder(model.entropy_bottleneck)
        self.coordinate_coder = CoordinateCoder()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    # TODO:
    # 1. Why do we keep numpoints
    # 2. Why do we divide by "tensor_stride" what is it?
    def encode(self, x, out_dir, filename):
        output_path = Path(out_dir).expanduser()
        y_ext = self.model.encoder(x)
        y = sort_spare_tensor(y_ext[-1])
        num_points = [len(ground_truth) for ground_truth in y_ext[:-1] + [x]]
        with open(output_path / (filename + "_num_points.bin"), "wb") as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder.encode(y.F, output_path, filename)
        self.coordinate_coder.encode(
            (y.C // y.tensor_stride[0]).detach().cpu()[:, 1:],
            output_path,
            filename,
        )

    @torch.no_grad()
    def decode(self, in_dir, filename, rho=1):
        input_path = Path(in_dir).expanduser()
        y_C = self.coordinate_coder.decode(input_path, filename)
        # TODO: Why concat??
        y_C = torch.cat(
            (torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=-1
        )

        indices_sort = np.argsort(array2vector(y_C, y_C.max() + 1))

        # Decoded coordinates
        y_C = y_C[indices_sort]

        # Decoded features
        y_F = self.feature_coder.decode(input_path, filename)
        y = ME.SparseTensor(
            features=y_F, coordinates=y_C * 8, tensor_stride=8, device=self.device
        )

        # Decode labels
        with open(input_path / (filename + "_num_points.bin"), "rb") as f:
            num_points = np.frombuffer(f.read(4 * 3), dtype=np.int32).tolist()
            num_points[0] = int(rho * num_points[0])
            num_points = [[num] for num in num_points]

        _, out = self.model.decoder(
            y, num_points, ground_truth_list=[None] * 3, training=False
        )

        return out
