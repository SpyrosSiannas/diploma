import numpy as np
import torch


def make_block(nn_module, num_layers: int, channels: int):
    block = []
    for _ in range(num_layers):
        block.append(nn_module(channels))
    return torch.nn.Sequential(*block)


def array2vector(array, step):
    """Ravel 2D array with multi-channel to one 1D vector by sum each channel with different step."""
    array, step = array.long().cpu(), step.long().cpu()
    vector = sum([array[:, i] * (step**i) for i in range(array.shape[-1])])

    return vector


def isin(data, ground_truth):
    """Generate a mask of the same length as `data` that is True where an element of `data` is in `ground_truth`.

    Args:
    ----
    data : tensor of shape [N, D]
    ground_truth : tensor of shape [N, D]

    Returns:
    -------
    torch.Tensor : boolean vector of shape [N]
    """
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)


def istopk(data, nums, rho=1.0):
    """Return a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.

    Args:
    ----
    data: SparseTensor of shape [N, D]
    nums: list of shape [batch_size]
    rho: float


    Returns:
    -------
    mask: boolean vector of shape [N]
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums, strict=False):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(
            data.F[row_indices].squeeze().detach().cpu(), k
        )  # must CPU.
        mask[row_indices[indices]] = True

    return mask.bool().to(data.device)
