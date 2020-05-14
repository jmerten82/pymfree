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
from pymfree.core.function import L2SquaredNorm
from pymfree.core.function import LInfNorm
from pymfree.core.function import DomainFunction
from pymfree.util.utils import scale_params
from pymfree.util.utils import any_norm_nn_search


class Domain(object):
    """
    This is a docstring.
    """

    def __init__(
            self, coordinates, values,
            norm=L2Norm, k=32, dim=2, device=torch.device('cpu')):
        if isinstance(coordinates, list):
            self.node_coordinates = torch.tensor(coordinates, device=device)[
                :-(len(coordinates) % dim)].reshape(-1, dim)
        elif isinstance(coordinates, np.ndarray):
            if len(coordinates.shape) == 1:
                self.node_coordinates = torch.tensor(
                    coordinates, device=device)[
                    :-(len(coordinates) % dim)].reshape(-1, dim)
            elif len(coordinates.shape) == 2:
                self.node_coordinates = torch.tensor(
                    coordinates, device=device)
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
            self.node_coordinates.to(device)
        else:
            raise TypeError(
                "PyMfree Domain: Input coordinates must \
                        be torch tensor, numpy array or list.")

        if isinstance(values, DomainFunction):
            # TO BE SEEN WHAT HAPPENS IF THIS IS CALLED ON CUDA DEVICE
            self.node_values = values(self.node_coordinates)
        elif isinstance(values, list):
            if len(values) < len(self):
                raise TypeError(
                    "PyMfree Domain: Node value list too short.")
            else:
                self.node_values = torch.tensor(
                    values[:len(self)], device=device)
        elif isinstance(values, np.ndarray):
            if len(values) < len(self) or len(values.shape != 1):
                raise TypeError(
                    "PyMfree Domain: Node value array invalud.")
            else:
                self.node_values = torch.tensor(
                    values[:len(self)], device=device)
        elif isinstance(values, torch.Tensor):
            if len(values) < len(self) or len(values.shape != 1):
                raise TypeError(
                    "PyMfree Domain: Node value array invalud.")
            else:
                self.node_values = values[:len(self)]
            self.node_values.tol(device)
        else:
            raise TypeError(
                "PyMfree Domain: Input values must come from DomainFunction \
                        a torch tensor, numpy array or list.")

        if not isinstance(norm, Norm):
            raise TypeError("PyMfree Domain: Need a proper Norm on Domain.")
        self.norm = norm

        if isinstance(self.norm, L1Norm):
            self.index = faiss.IndexFlat(self.dim, faiss.METRIC_L1)
        elif isinstance(self.norm, L2Norm):
            self.index = faiss.IndexFlatL2(self.dim)
        elif isinstance(self.norm, L2SquaredNorm):
            self.index = faiss.IndexFlatL2(self.dim)
        elif isinstance(self.norm, LInfNorm):
            self.index = faiss.IndexFlat(self.dim, faiss.METRIC_Linf)
        else:
            self.index = None

        if isinstance(k, int):
            self.k = k
        else:
            raise TypeError("PyMfree Domain: k must be an integer.")

        if self.index is not None:
            self.index.add(
                self.node_coordinates.to(torch.device('cpu')).numpy())

        self.query = None
        self.neighbour_map = None
        self.neighbour_distances = None
        self.__counter = 0
        self.k = torch.tensor([k], device=device)
        self.scale = {}
        self.scale['shift'] = torch.tensor([0.], device=device)
        self.scale['scale'] = torch.tensor([1.], device=device)
        self.scale['unit'] = torch.tensor([1.], device=device)
        self.scale['rescaled'] = False

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

        five = "Coordinates info ---: "
        + "Unit: " + str(self.scale['unit'].item())
        +  " Scale: " + str(self.scale['scale'].item())
        +  " Shift: " + str(self.scale['shift'].item())
        print(one+"\n"+two+"\n"+three+"\n"+four+"\n"+five)

    def __repr__(self):
        return print(self)

    def __call__(self, x):
        if isinstance(x, list):
            self.query = torch.tensor(x, device=self.device)[
                :-(len(x) % self.dim)].reshape(-1, self.dim)
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                self.query = torch.tensor(
                    x, device=self.device)[
                    :-(len(x) % self.dim)].reshape(-1, self.dim)
            elif len(x.shape) == 2:
                self.query = torch.tensor(
                    x, device=self.device)
            else:
                raise TypeError(
                    "PyMfree Domain: Query input array dimensions invalid")
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                self.query = x[
                    :-(len(x) % self.dim)].reshape(-1, self.dim)
            elif len(x.shape) == 2:
                self.query = x
            else:
                raise TypeError(
                    "PyMfree Domain: Query array dimensions invalid")
            self.query.to(self.device)
        else:
            raise TypeError(
                "PyMfree Domain: Input query array must \
                        be torch tensor, numpy array or list.")

        if self.scale['rescaled']:
            self.query *= self.scale['scale']
            self.query += self.scale['shift']

        if self.index is not None:
            self.neighbour_distances, self.neighbour_map = self.index.search(
                self.query.numpy(), self.k)
            self.neighbour_distances = torch.tensor(
                self.neighbour_distances, device=self.device)
            self.neighbour_map = torch.tensor(
                self.neighbour_map, device=self.device)
        else:
            self.neighbour_distances, self.neighbour_map = any_norm_nn_search(
                self.node_coordinates, self.query, self.norm, self.k)

        return self.neighbour_distances, self.neighbour_map

    def __getitem__(self, index):
        if not isinstance(index, int) or index > len(self)-1:
            raise IndexError("PyMfree Domain: Index out of bounds.")

    def rescale_coordinates(self, min_in=-1., max_in=1.):
        if not isinstance(min, float) or not isinstance(max, float):
            raise TypeError("PyMfree Domain: domain bounds must be float.")
        self.scale['scale'], self.scale['shift'] = scale_params(
            self.coordinates, min_in, max_in)
        self.scale['unit'] = torch.tensor(
            max_in - min_in, device=self.device)
        self.coordinates *= self.scale['scale']
        self.coordinates += self.scale['shift']
        self.scale['rescale'] = True

    def to(self, device=torch.device('cpu')):
        if not isinstance(device, torch.device):
            raise TypeError("PyMfree Domain: Device must be torch device.")

        self.node_coordinates = self.node_coordinates.to(device)
        self.node_values = self.node_values.to(device)
        if self.query is not None:
            self.query = self.query.to(device)
        if self.neighbour_map is not None:
            self.neighbour_map = self.neighbour_map.to(device)
        if self.neighbour_distances is not None:
            self.neighbour_distances = self.neighbour_distances.to(device)
        self.k = self.k.to(device)
        self.scale['shift'] = self.scale['shift'].to(device)
        self.scale['scale'] = self.scale['scale'].to(device)
        self.scale['unit'] = self.scale['unit'].to(device)

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

    @property
    def device(self):
        return self.node_coordinates.device
