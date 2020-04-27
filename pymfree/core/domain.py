# domain.py
# part of pymfree

"""
This module does the following...
"""

import torch
from pymfree.core.norm import Norm


class Domain(object):
    """
    This is a docstring.
    """

    def __init__(self):
        self._domain = torch.rand(10, 3)
        self._support = torch.rand(10, 3)
        self._support_function = torch.rand(10, 1)
        self._rbf_params = torch.empty()
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


class DomainFunction(Norm):
    r"""

    """
    def __init__(self):
        pass
