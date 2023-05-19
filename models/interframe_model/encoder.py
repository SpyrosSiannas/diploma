import MinkowskiEngine as ME
import torch
import torch.nn as nn
from .inception_residual import InceptionResidualBlock
from .modelutils import make_block


class InterframeEncoder(nn.Module):
    def __init__(self, channels=None) -> None:
        super(InterframeEncoder, self).__init__()
        if channels is None:
            # Default channels
            channels = [1, 16, 32, 64, 32, 8]

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.block0 = make_block(
            InceptionResidualBlock, num_layers=3, channels=channels[2]
        )

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.block1 = make_block(
            InceptionResidualBlock, num_layers=3, channels=channels[3]
        )

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        )
        self.block2 = make_block(
            InceptionResidualBlock, num_layers=3, channels=channels[4]
        )

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)
        out2 = self.conv3(out2)

        return [out0, out1, out2]


if __name__ == "__main__":
    model = InterframeEncoder().cuda()
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
