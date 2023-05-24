import argparse
from pathlib import Path

import torch
import open3d as o3d

from models.interframe_model.interframe_ae import InterFrameAE
from scripts.codec_utils import InterframeCodec, load_sparse_tensor
from scripts.utils import write_ply_ascii_geo


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Compress and decompress point clouds using an interframe AutoEncoder"
    )

    argparser.add_argument(
        "input_dir",
        type=lambda x: Path(x).expanduser(),
        help="Path to input point cloud",
    )
    argparser.add_argument("filename", type=str, help="Filename of input point cloud")
    argparser.add_argument(
        "--output_dir",
        type=lambda x: Path(x).expanduser(),
        help="Path to output point cloud",
        default=".",
    )
    argparser.add_argument(
        "--load_checkpoint",
        type=lambda x: Path(x).expanduser(),
        help="Path to checkpoint to load",
    )

    return argparser.parse_args()


def display_pcs(input_path, output_path, filename) -> None:
    original_pc = o3d.io.read_point_cloud(str(input_path / filename))
    compressed_pc = o3d.io.read_point_cloud(str(output_path / filename))
    o3d.visualization.draw_geometries([original_pc])
    o3d.visualization.draw_geometries([compressed_pc])


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterFrameAE().to(device)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint))
    codec = InterframeCodec(model)
    filename = args.filename.split(".")[0]
    filepath = args.input_dir / args.filename
    inp = load_sparse_tensor(filepath, device=device)
    codec.encode(inp, args.output_dir, filename)
    out = codec.decode(args.output_dir, filename)
    write_ply_ascii_geo(args.output_dir / (filename + ".ply"), out.C.detach().cpu().numpy()[:,1:])
    display_pcs(args.input_dir, args.output_dir, args.filename)
