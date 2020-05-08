""" domain.py part of pymfree/core

This module implements a central PyMfree entity, which is a domain. This domain
is defined by a number of n-dim support nodes and a norm on that domain.
"""
import torch
import numpy
from pymfree.core.function import Norm
from pymfree.core.function import L2Norm
from pymfree.core.function import DomainFunction


class Domain(object):
    """
    This is a docstring.
    """

    def __init__(self, coordinates, values, norm=L2Norm, query='self', dim=2):
        if isinstance(coordinates, list):
            # self.dim = dim
            pass
        elif isinstance(coordinates, numpy.ndarray):
            self.dim = None
            pass
        elif isinstance(coordinates, torch.Tensor):
            self.dim = None
            pass
        else:
            raise TypeError(
                "PyMfree Domain: Input coordinates must \
                        be torch tensor, numpy array or list.")
        self.node_coordinates = None

        if isinstance(values, DomainFunction):
            pass
        elif isinstance(values, list):
            pass
        elif isinstance(values, numpy.ndarray):
            pass
        elif isinstance(values, torch.Tensor):
            pass
        else:
            raise TypeError(
                "PyMfree Domain: Input values must come from DomainFunction \
                        a torch tensor, numpy array or list.")
        self.node_values = None

        if not isinstance(norm, Norm):
            raise TypeError("PyMfree Domain: Need a proper Norm on Domain.")
        self.norm = norm

        if query == 'self':
            self.query = self.node_coordinates
        elif isinstance(query, list):
            pass
        elif isinstance(query, numpy.ndarray):
            pass
        elif isinstance(query, torch.Tensor):
            raise TypeError("PyMfree Domain: Query coordinates must the domain \
                    a torch tensor, numpy array or list.")
        self.query = None

    def __len__(self):
        return len(self.node_coordinates)

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        return print(self)

    def __call__(self):
        pass

    def __getitem__(self):
        pass

    @property
    def shape(self):
        return {'nodes': len(self), 'dim': self.dim}

    @property
    def dim(self):
        return self.node_coordinates.shape[1]
