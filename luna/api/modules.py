from collections import OrderedDict
import torch.nn as nn


class HaloModule(nn.Module):
    r"""Base class for all halo modules.

    Halo modules should subclass this class.
    """

    __constants__ = ['halo_size']

    def __init__(self, halo_size: int):
        super(HaloModule, self).__init__()
        self.halo_size = halo_size

        self._rrefs = []
        self._neighbors = OrderedDict()
