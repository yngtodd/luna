import torch.nn as nn
from torch.distributed.rpc import RRef

from luna.api.modules import HaloModule
from luna.rpc import _remote_method


class Halo1d(HaloModule):
    r"""One dimensional halo

    This will orchestra halo regions to be passed
    to a neural network module for a given domain
    decomposition.

    Args:
        halo_size: the size of the halo between subdomains

        model: neural network model to compute over the
               subdomain
    """

    def __init__(self, halo_size: int, model: nn.Module, rank: int):
        super(Halo1D, self).__init__(halo_size)
        self.rank = rank
        #TODO(Todd): think of how to reference num_workers here
        self.left_rank = rank - 1 if rank > 0 else None
        self.right_rank = rank + 1 if rank < num_workers - 1 else None
        # zero padding
        self.halo_pad = nn.ConstantPad1d(halo_size, 0)
        self.model = model

    def get_left_halo(self, x: RRef):
        """ Get the halo region from left neighbor

        Args:
            x: rref to a tensor
        """
        tensor = x.local_value()
        return tensor[..., :self.halo_size]

    def get_right_halo(self, x: RRef):
        """ Get the halo region from right neighbor

        Args:
            x: rref to a tensor
        """
        tensor = x.local_value()
        return value[..., -self.halo_size:]

    def forward(self, x_refs):
        """ Compute over a subdomain """
        # Get the subdomain that we need
        x = x_refs[self.rank].to_here()
        # pad zeros of halo_size on both sides
        padded = self.halo_pad(x)

        if self.left_rank is not None:
            left_halo = _remote_method(self.get_left_halo, x_refs[self.rank])
            padded[..., :self.halo_size] = left_halo

        if self.right_rank is not None:
            right_halo = _remote_method(self.get_right_halo, x_refs[self.rank])
            padded[..., -self.halo_size:] = right_halo

        return self.model(padded)
