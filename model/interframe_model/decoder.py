import MinkowskiEngine as ME
import torch
import torch.nn
from inception_residual import InceptionResidualBlock
from modelutils import isin, istopk, make_block


class InterframeDecoder(torch.nn.Module):
    """the decoding network with upsampling."""

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [8, 64, 32, 16]
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.block0 = make_block(
            nn_module=InceptionResidualBlock, num_layers=3, channels=channels[1]
        )

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.block1 = make_block(
            nn_module=InceptionResidualBlock, num_layers=3, channels=channels[2]
        )

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.block2 = make_block(
            nn_module=InceptionResidualBlock, num_layers=3, channels=channels[3]
        )

        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth=None, training=False):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert ground_truth is not None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, nums_list, ground_truth_list, training=True):
        #
        out = self.relu(self.conv0(self.relu(self.up0(x))))
        out = self.block0(out)
        out_cls_0 = self.conv0_cls(out)
        out = self.prune_voxel(
            out, out_cls_0, nums_list[0], ground_truth_list[0], training
        )
        #
        out = self.relu(self.conv1(self.relu(self.up1(out))))
        out = self.block1(out)
        out_cls_1 = self.conv1_cls(out)
        out = self.prune_voxel(
            out, out_cls_1, nums_list[1], ground_truth_list[1], training
        )
        #
        out = self.relu(self.conv2(self.relu(self.up2(out))))
        out = self.block2(out)
        out_cls_2 = self.conv2_cls(out)
        out = self.prune_voxel(
            out, out_cls_2, nums_list[2], ground_truth_list[2], training
        )

        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]

        return out_cls_list, out


if __name__ == "__main__":
    model = InterframeDecoder().cuda()
    # create random sparse tensor input with ME

    coords = torch.rand(1024, 4).cuda()
    feats = torch.randn(1024, 1).cuda()
    input = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device="cuda:0",
    )

    out0, out1, out2 = model(input)
    print(model)
