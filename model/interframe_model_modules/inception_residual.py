import MinkowskiEngine as ME
import torch.nn as nn


class InceptionResidualBlock(nn.Module):
    """More efficient feature aggregation
    see: https://arxiv.org/pdf/2011.03799.pdf.
    """

    def __init__(self, num_chans) -> None:
        super(InceptionResidualBlock, self).__init__()

        self.chain2_conv0 = ME.MinkowskiConvolution(
            in_channels=num_chans,
            out_channels=num_chans // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.chain2_conv1 = ME.MinkowskiConvolution(
            in_channels=num_chans // 4,
            out_channels=num_chans // 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.chain1_conv0 = ME.MinkowskiConvolution(
            in_channels=num_chans,
            out_channels=num_chans // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.chain1_conv1 = ME.MinkowskiConvolution(
            in_channels=num_chans // 4,
            out_channels=num_chans // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )
        self.chain1_conv2 = ME.MinkowskiConvolution(
            in_channels=num_chans // 4,
            out_channels=num_chans // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3,
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        chain1_out = self.relu(self.chain2_conv1(self.relu(self.chain2_conv0(x))))
        chain2_out = self.relu(
            self.chain1_conv2(
                self.relu(self.chain1_conv1(self.relu(self.chain1_conv0(x))))
            )
        )
        return ME.cat(chain2_out, chain1_out) + x


if __name__ == "__main__":
    model = InceptionResidualBlock(1024)
    print(model)
