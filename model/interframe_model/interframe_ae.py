import MinkowskiEngine as ME
import torch.nn as nn
from decoder import InterframeDecoder
from encoder import InterframeEncoder
from entropy_bottleneck import EntropyBottleneck


class InterFrameAE(nn.Module):
    def __init__(self) -> None:
        super(InterFrameAE, self).__init__()
        self.encoder = InterframeEncoder()
        self.decoder = InterframeDecoder()
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(
            data.F, quantize_mode=quantize_mode
        )
        data_Q = ME.SparseTensor(
            features=data_F,
            coordinate_map_key=data.coordinate_map_key,
            coordinate_manager=data.coordinate_manager,
            device=data.device,
        )

        return data_Q, likelihood

    def forward(self, x, training=False):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x]
        nums_list = [
            [len(C) for C in ground_truth.decomposed_coordinates]
            for ground_truth in ground_truth_list
        ]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(
            y, quantize_mode="noise" if training else "symbols"
        )

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {
            "out": out,
            "out_cls_list": out_cls_list,
            "prior": y_q,
            "likelihood": likelihood,
            "ground_truth_list": ground_truth_list,
        }


if __name__ == "__main__":
    model = InterFrameAE().cuda()
    print(model)
