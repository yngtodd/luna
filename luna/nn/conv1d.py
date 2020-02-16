from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import rpc

from luna.rpc import _remote_method


def max_element(x):
    """ Helper function to apply max pool on RRefs """
    return F.adaptive_max_pool1d(
        x.local_value(), 1, return_indices=False)


class ConvBlock1d(nn.Module):
    """ Simple 1D convolution """

    def __init__(self):
        super(ConvBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3)
        self.conv2 = nn.Conv1d(64, 128, 3)

    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))


class DomainDecompConv(nn.Module):
    """ 1D convolution via domain decomposition """

    def __init__(self, model, workers, num_labels):
        super(DomainDecompConv, self).__init__()
        # initialize our workers
        self.models = [rpc.remote(w, deepcopy, args=(model,)) for w in workers]
        self.fc = nn.Linear(128, num_labels)

    def forward(self, x_refs):
        x_refs = [_remote_method(model.forward, x_refs) for model in self.models]
        maxes = [rpc.rpc_async(x.owner(), max_element, args=[x]) for x in x_refs]
        maxes = [m.wait() for m in maxes]
        maxes, _ = torch.max(torch.cat(maxs, dim=-1), dim=-1)
        return self.fc(maxes)
