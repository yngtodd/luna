import torch
import torch.nn as nn
from torch.distributed import rpc

from luna.nn.conv1d import HaloConv1d
from luna.rpc import _parameter_rrefs, _remote_method


def max_element(x):
    """ Helper function to apply max pool on RRefs """
    return F.adaptive_max_pool1d(
        x.local_value(), 1, return_indices=False)


class DomainDecomp1d(nn.Module):
    """ 1D convolution via domain decomposition """

    def __init__(self, halo_size, num_workers, rank, num_labels):
        super(DomainDecomp1d, self).__init__()

        workers = [f'worker{w}' for w in range(num_workers)]

        self.halo_conv_rrefs = [
            rpc.remote(w, HaloConv1d, args=(halo_size, num_workers, rank))
            for w in workers
        ]

        self.fc = nn.Linear(128, num_labels)

    def forward(self, x_refs):
        x_refs = [_remote_method(HaloConv1d.forward, conv, x_refs) for conv in self.halo_conv_rrefs]
        xs = [x.to_here() for x in x_refs]
        print(f'Xs: {xs}')
        maxes = [rpc.rpc_async(x.owner(), max_element, args=[x]) for x in x_refs]
        maxes = [m.wait() for m in maxes]
        maxes, _ = torch.max(torch.cat(maxs, dim=-1), dim=-1)
        return self.fc(maxes)

    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of each subdomain convolution
        for halo_conv in self.halo_conv_rrefs:
            remote_params.extend(_remote_method(_parameter_rrefs, halo_conv))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.fc))
        return remote_params
