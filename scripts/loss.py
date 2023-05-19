import torch
from torch.nn import functional as F

from models.interframe_model.modelutils import isin

def get_gpccv2_loss(model_out, ground_truth):
    # Fixme: Remove dict, find better way (pandas?) :P
    bce = 0
    for out_cls, ground_truth in zip(model_out['out_cls_list'], model_out['ground_truth_list'], strict=False):
        curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
        bce += curr_bce
    bpp = get_bits(model_out['likelihood'])/float(ground_truth.__len__())
    return bce + bpp


def get_bce(data, groud_truth):
    criterion = torch.nn.BCEWithLogitsLoss()
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    return sum_bce


def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))
    return bits


def MSE_KLD(recon_x, original, mean, log_var):
    MSE = F.mse_loss(recon_x, original)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return MSE + KLD


def chamfer_distance(recon_x, original):
    dist1 = torch.cdist(recon_x, original)
    dist2 = torch.cdist(original, recon_x)
    # Calcualte Chamfer Loss
    return (
        torch.mean(torch.min(dist1, dim=1)[0]) + torch.mean(torch.min(dist2, dim=1)[0])
    ) / 2


def custom_loss(recon_x, original, mean, log_var):
    return MSE_KLD(recon_x, original, mean, log_var) + chamfer_distance(
        recon_x, original
    )
