""" domain.py part of pymfree/core

This module implements a central PyMfree entity, which is a domain. This domain
is defined by a number of n-dim support nodes and a norm on that domain.
"""
import torch
import faiss
import numpy as np
from pymfree.core.function import Norm
from pymfree.core.function import L1Norm
from pymfree.core.function import L2Norm
from pymfree.core.function import LInfNorm
from pymfree.core.function import DomainFunction
from pymfree.util.utils import scale_params


class Domain(object):
    """
    This is a docstring.
    """

    def __init__(self, coordinates, values, norm=L2Norm, k=32, dim=2):
        if isinstance(coordinates, list):
            self.node_coordinates = torch.tensor(coordinates)[
                :-(len(coordinates) % dim)].reshape(-1, dim)
        elif isinstance(coordinates, np.ndarray):
            if len(coordinates.shape) == 1:
                self.node_coordinates = torch.tensor(coordinates)[
                    :-(len(coordinates) % dim)].reshape(-1, dim)
            elif len(coordinates.shape) == 2:
                self.node_coordinates = torch.tensor(coordinates)
            else:
                raise TypeError(
                    "PyMfree Domain: Input array dimensions invalid")
        elif isinstance(coordinates, torch.Tensor):
            if len(coordinates.shape) == 1:
                self.node_coordinates = coordinates[
                    :-(len(coordinates) % dim)].reshape(-1, dim)
            elif len(coordinates.shape) == 2:
                self.node_coordinates = coordinates
            else:
                raise TypeError(
                    "PyMfree Domain: Input array dimensions invalid")
            pass
        else:
            raise TypeError(
                "PyMfree Domain: Input coordinates must \
                        be torch tensor, numpy array or list.")

        if isinstance(values, DomainFunction):
            self.node_values = values(self.node_coordinates)
        elif isinstance(values, list):
            if len(values) < len(self):
                raise TypeError(
                    "PyMfree Domain: Node value list too short.")
            else:
                self.node_values = torch.tensor(values[:len(self)])
        elif isinstance(values, np.ndarray):
            if len(values) < len(self) or len(values.shape != 1):
                raise TypeError(
                    "PyMfree Domain: Node value array invalud.")
            else:
                self.node_values = torch.tensor(values[:len(self)])
        elif isinstance(values, torch.Tensor):
            if len(values) < len(self) or len(values.shape != 1):
                raise TypeError(
                    "PyMfree Domain: Node value array invalud.")
            else:
                self.node_values = values[:len(self)]
        else:
            raise TypeError(
                "PyMfree Domain: Input values must come from DomainFunction \
                        a torch tensor, numpy array or list.")
        self.node_values = None

        if not isinstance(norm, Norm):
            raise TypeError("PyMfree Domain: Need a proper Norm on Domain.")
        self.norm = norm

        if isinstance(self.norm, L1Norm):
            self.index = faiss.IndexFlat(self.dim, faiss.METRIC_L1)
        elif isinstance(self.norm, L2Norm):
            self.index = faiss.IndexFlatL2(self.dim)
        elif isinstance(self.norm, LInfNorm):
            self.index = faiss.IndexFlat(self.dim, faiss.METRIC_Linf)
        else:
            self.index = None

        if isinstance(k, int):
            self.k = k
        else:
            raise TypeError("PyMfree Domain: k must be an integer.")

        self.query = None
        self.neighbour_map = None
        self.neighbour_distances = None
        self.__counter = 0
        self.scaler = {}
        self.k = torch.tensor([k])
        self.shift = torch.tensor([0.])
        self.scale = torch.tensor([1.])

    def __len__(self):
        return len(self.node_coordinates)

    def __iter__(self):
        return self

    def __next__(self):
        if self.__counter > len(self)-1:
            self.__counter = 0
            raise StopIteration
        else:
            self.__counter += 1
            return self.node_coordinates[
                self.__counter-1], self.node_values[self.__counter-1]

    def __str__(self):
        one = str(self.__class__.__name__)
        one = one[one.rfind('.')+1:one.rfind('\'')]
        two = str(
            len(self) + "Nodes; \t" + str(self.dim) + "spatial dimensions.")
        three = str(self.index.__class__.__name__) + "performs fast nn search."
        if self.query_ready:
            four = str(len(self.query)) + " point query established."
        else:
            four = "No query in Domain."
        if self.scaler is None:
            five = "No coordinate rescaling."
        else:
            five = "Coordinates rescaled into " \
                + str(self.scaler.get_params()['feature_range']) + " domain."
        print(one+"\n"+two+"\n"+three+"\n"+four+"\n"+five)

    def __repr__(self):
        return print(self)

    def __call__(self, x):
        if len(self.scaler) != 0:
            pass
        pass

    def __getitem__(self):
        pass

    def rescale_coordinates(self, min_in=-1., max_in=1.):
        if not isinstance(min, float) or not isinstance(max, float):
            raise TypeError("PyMfree Domain: domain bounds must be float.")
        self.scale['scale'], self.scale['shift'] = scale_params(
            self.coordinates, min_in, max_in)
        self.scale['new_unit'] = max_in - min_in
        self.coordinates *= self.scale['scale']
        self.coordinates += self.scale['shift']

    def to(self, device=torch.device('cpu')):
        pass

    @property
    def shape(self):
        if self.query_ready:
            helper = len(self.query_ready)
        else:
            helper = 0
        return {'nodes': len(self), 'dim': self.dim, 'query': helper}

    @property
    def dim(self):
        return self.node_coordinates.shape[1]

    @property
    def query_ready(self):
        return isinstance(self.query, torch.Tensor)
