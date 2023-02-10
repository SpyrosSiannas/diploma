import torch
from torch.nn import functional as F


def MSE_KLD(recon_x, original, mean, log_var):
    MSE = F.mse_loss(recon_x, original)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return MSE + KLD
