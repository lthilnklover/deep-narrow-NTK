import torch
import os
import argparse
import numpy as np
import random


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_grad(model, x, device):
    model = model.to(device)
    x = x.to(device)
    return torch.autograd.grad(model(x), model.parameters())


def get_kernel(grad_x, grad_y):
    assert len(grad_x) == len(grad_y)
    return sum([torch.sum(torch.mul(grad_x[i], grad_y[i])) for i in range(len(grad_x))])


def round_to_nearest_int(x):
    assert len(x.shape) == 1
    for i in range(len(x)):
        x[i] = round(x[i].item())
    return x.type(torch.uint8)
