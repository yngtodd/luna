from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import rpc

from luna.rpc import _remote_method
from luna.nn.halo import Halo1d


class ConvBlock1d(nn.Module):
    """ Simple 1D convolution """

    def __init__(self):
        super(ConvBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3)
        self.conv2 = nn.Conv1d(64, 128, 3)

    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))


class HaloConv1d(nn.Module):
    """ Halo => ConvBlock """

    def __init__(self, halo_size, num_workers, rank):
        super(HaloConv1d, self).__init__()
        self.halo = Halo1d(halo_size, num_workers, rank)
        self.conv = ConvBlock1d()

    def forward(self, x_rrefs):
        print(f'x_rrefs: {x_rrefs}')
        x = self.halo(x_rrefs)
        print(f'Input conv: {x.shape}')
        x = self.conv(x)
        return x

