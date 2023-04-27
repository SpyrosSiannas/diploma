import torch
from torch.nn import functional as F


def MSE_KLD(recon_x, original, mean, log_var):
    MSE = F.mse_loss(recon_x, original)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (MSE + KLD)


def chamfer_distance(recon_x, original):
    dist1 = torch.cdist(recon_x, original)
    dist2 = torch.cdist(original, recon_x)
    # Calcualte Chamfer Loss
    return (torch.mean(torch.min(dist1, dim=1)[0]) + torch.mean(torch.min(dist2, dim=1)[0])) / 2


def custom_loss(recon_x, original, mean, log_var):
    return MSE_KLD(recon_x, original, mean, log_var) + chamfer_distance(recon_x, original)
