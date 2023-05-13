import os
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from model.interframe_model import EntropyBottleneck
from utils.gpcc import gpcc_decode, gpcc_encode
from utils.utils import read_ply_ascii_geo, write_ply_ascii_geo


class FeatureCoder:
    """Class that uses a learned entropy model to compress and decompress
    the feature tensor of a point cloud in a lossy manner.
    """

    def __init__(
        self,
        entropy_model: EntropyBottleneck,
        input_path: str = ".",
        output_path: str = ".",
    ) -> None:
        self.entropy_model = entropy_model.cpu()
        self.output_path = Path(output_path)
        self.input_path = Path(input_path)

    def encode(self, features, filename):
        strings, min_v, max_v = self.entropy_model.compress(features.cpu())
        shape = features.shape
        with open(self.input_path / filename + "_Feats.bin", "wb") as f:
            f.write(strings)
        with open(self.input_path / filename + "_Header.bin", "wb") as f:
            f.write(np.array(shape, dtype=np.int32).tobytes())
            f.write(np.array(len(min_v), dtype=np.int8).tobytes())
            f.write(np.array(min_v, dtype=np.float32).tobytes())
            f.write(np.array(max_v, dtype=np.float32).tobytes())
        return

    def decode(self, filename):
        with open(self.output_path / filename + "_Header.bin", "rb") as f:
            shape = np.frombuffer(f.read(4 * 2), dtype=np.int32)
            min_v_len = np.frombuffer(f.read(1), dtype=np.int8)
            min_v = np.frombuffer(f.read(4 * min_v_len), dtype=np.float32)[0]
            max_v = np.frombuffer(f.read(4 * min_v_len), dtype=np.float32)[0]
        with open(self.output_path / filename + "_Feats.bin", "rb") as f:
            strings = f.read()

        features = self.entropy_model.decompress(
            strings, min_v, max_v, shape, shape[-1]
        )
        return features


class CoordinateCoder:
    """Class that uses the GPCC model to compress and decompress the coordinates
    of a point cloud in a lossy manner.
    """

    def __init__(self, input_path: str, output_path: str):
        """Construct a CoordinateCoder object and validates the input and output paths.

        Args:
        ----
        input_path (str): Input path for the point cloud to be decoded
        output_path (str): Output path for the point cloud to be encoded

        Raises:
        ------
        Exception: Invalid input or output path.
        """
        self.input_path = Path(input_path).expanduser()
        self.output_path = Path(output_path).expanduser()

        if not self.input_path.exists() or not self.output_path.exists():
            raise Exception("Invalid input or output path.")

    def encode(self, coordinates, filename: str):
        temp_file = self.output_path / (filename + ".ply")
        output_file = self.output_path / (filename + ".bin")
        coords = coordinates.numpy().astype("int")

        write_ply_ascii_geo(temp_file, coords)
        gpcc_encode(str(temp_file), str(output_file))

        os.remove(temp_file)
        pass

    def decode(self, filename: str):
        input_file = self.input_path / (filename)
        temp_file = self.input_path / (filename.split(".")[0] + ".ply")

        gpcc_decode(str(input_file), str(temp_file))
        coords = read_ply_ascii_geo(temp_file)
        os.remove(input_file)
        return coords


class InterframeCodec:
    r"""Class that takes an interframe AutoEncoder as input and
    performs compression and decompression on point clouds using
    a feature coder and a coordinates coder..shape[0])+'\n').
    """

    def __init__(self, model, input_path, output_path):
        self.model = model
        self.feature_coder = FeatureCoder(
            model.entropy_bottleneck, input_path, output_path
        )
        self.coordinate_coder = CoordinateCoder(input_path, output_path)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.input_path.exists() or not self.output_path.exists():
            raise Exception("Invalid input or output path.")

    @torch.no_grad()
    # TODO:
    # 1. Why do we keep numpoints
    # 2. Why do we divide by "tensor_stride" what is it?
    def encode(self, x, filename):
        y = self.model.encoder()
        # TODO: Implement sparse tensor sorting for utils!
        num_points = [len(ground_truth) for ground_truth in y[1:] + [x]]
        with open(self.input_path / filename + "_num_points.bin", "wb") as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder.encode(y.F, self.output_path / filename)
        self.coordinate_coder.encode(
            (y.C // y.tensor_stride[0]).detach().cpu()[:, 1:],
            self.output_path / filename,
        )

    @torch.no_grad()
    def decode(self, filename, rho=1):
        y_C = self.coordinate_coder.decode(self.input_path / filename)
        # TODO: Why concat??
        y_C = torch.cat(
            (torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=1
        )
        # TODO: indices_sort = np.argsort(array2vector(y_C, y_C.max()+1))

        indices_sort = np.array((y_C[:, 0] * y_C.max() + y_C[:, 1]).argsort())
        # Decoded coordinates
        y_C = y_C[indices_sort]

        # Decoded features
        y_F = self.feature_coder.decode(self.input_path / filename)
        y = ME.SparseTensor(
            features=y_F, coordinates=y_C * 8, tensor_stride=8, device=self.device
        )

        # Decode labels
        with open(self.input_path / filename + "_num_points.bin", "rb") as f:
            num_points = np.frombuffer(f.read(4 * 3), dtype=np.int32).tolist()
            num_points[-1] = int(rho * num_points[-1])
            num_points = [[num] for num in num_points]

        _, out = self.model.decoder(
            y, num_points, ground_truth=[None] * 3, training=False
        )
