from collections import OrderedDict
import torch.nn as nn


class HaloModule(nn.Module):
    r"""Base class for all halo modules.

    Halo modules should subclass this class.
    """

    def __init__(self):
        super(HaloModule, self).__init__()
        self._rrefs = []
        self._neighbors = OrderedDict()
