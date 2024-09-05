import torch


def random_spatial_shuffle(x):
    i = torch.randperm(x.shape[2])
    j = torch.randperm(x.shape[3])
    x = x[:, :, i]
    x = x[:, :, :, j]
    return x

