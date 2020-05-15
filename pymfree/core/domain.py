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
from pymfree.util.utils import check_pymfree_type
from pymfree.util.utils import scale_params


class Domain(object):
    r""" A mesh-free domain representation.

    Implements means to save support coordinates in n-dimensions, together with
    a function residing on the domain. Establishes norms and nearest neighbours
    searches on the domain, together with coordinate queries.

    Parameters
    ----------
    coordinates : list, numpy.ndarray or torch.Tensor
        The support coordinates of the domain. If provides as a list of floats,
        also the spatial dimensionality of the domain must be provided
        (see dim). If torch tensor or numpy array are provided as PyMfree
        coordinates, the dimensioanlity is directly inferred,
        otherwise it must also be provided.
    values : list, numpy.ndarray, torch.Tensor
             or pymfree.core.function.DomainFunction
        A function with values on the support points. If provided as numpy
        array or torch tensor, the input must be scalar. Also, the length of
        the function values must be equal or longer than the number of support
        coordinates. Instead of an input vector with values, also a
        DomainFunction can be provided, which is evaluated at the support
        coordinates.
    norm : pymfree.core.function.Norm, optional
        The norm to calculate distances on the Domain. Default is L2Norm
    k : int, optional
        The number of nearest neighbours that are searched within support given
        a coordinates query.
    dim : int, optional
        If scalars are provided at construction for the support coordinates,
        the dimension provided here is used to infer the number of spatial
        components of the input. E.g. a scalar of 10 input numbers and dim set
        to 3 would result in 3 3-d support coordinates. Defaults to 2.
    device : torch.device, optional
        The device the domain shall be created on. This means that all data
        vectors are stored on this device. Default is torch.device('cpu) and
        it can be changed later via Domain.to(device)

    Attributes
    ----------
    All of the following live on a specifc device which can be changed via
    Domain.to(device).

    node_coordinates : torch.Tensor
        This is where the domains support points are stored. The format of this
        is PyMfree coordinate.
    node_values : torch.Tensor
        A PyMfree scalar that holds function values at the support coordinates.
    query : torch.Tensor
        A PyMfree coordinate which stores the query coordinates the user
        requested via the call operator. This is None at construction and
        created after the call operator.
    neighbour_map : torch.Tensor
        The index map relating to query search. E.g. if n coordinates are
        queried and k is set to m, the dimension of this object would be
        (n, 1, m). This is None at construction and created after
        the call operator.
    neighbour_distances :torch.Tensor
        The distances to the support nodes relating to query search.
        E.g. if n coordinates are queried and k is set to m,
        the dimension of this object would be (n, 1, m). This is None at
        construction and created after the call operator.
    k : torch.Tensor
        The number of nearest neigbours for query searches.
    scale : dict
        If rescale_coordinates is called this holds the transformation
        parameters as a dict of torch tensors. Contains keys unit, shift and
        scale.

    Finally, this is the index which is used to calculate nearest neighbours
    fast.

    index : faiss.FlatIndex or None
        In most cases this is a faiss flat index, depending which Norm
        is used on the Domain. If a Norm is used that is unknown to faiss, the
        index is None and an internal, slower routines is used to calculate
        neighbour_map and distances.

    Raises
    ------
    TypeError
        If coordinates are provided as numpy array or torch tensor but are not
        scalar or coordinates.
    TypeError
        If coordinates are not provided as list, numpy array or torch tensor.
    TypeError
        If the values are provided as array but are either too short
        (< coordinates) or not a scalar.
    TypeError
        If values are not a list, numpy array, torch tensor or DomainFunction.
    TypeError
        If norm is not a pymfree.core.function.Norm.
    TypeError
        If k is not an integer.

    Notes
    -----
    Thinking about including faiss GPU index in the future. Currently this
    dodes not support all Norms.

    See also
    --------
    pymfree.core.function.DomainFunction
        One way of providing function values on the Domain.
    pymfree.core.domain.Domain()
        The call operator which initiates a coordinate query. Here the value
        k becomes relevant.
    pymfree.core.domain.Domain.to(device)
        Changes the device where the domain data is stored.
    pymfree.core.domain.Domain.rescale_coordinates()
        Shifts and scales the domain coordinates into a fixed window.
    pymfree.core.function.Norm.L2Norm
        The default Norm for the Domain.

    References
    ----------
    [1] [Faiss](https://github.com/facebookresearch/faiss)
        Facebook AI Research Similarity Search

    Examples
    --------
    The following constructs a domain with one 4-d coordinate at (1.,1.,1.,1.)
    and sets this coordinate to 42.
    >>> from pymfree.core.domain import Domain
    >>> my_domain = Domain([1., 1., 1., 1., 666.], [42., 66., 3.14], dim=4)

    And this constructs a 3-d domain with 10000 random coordinates, uses the
    L1Norm on the mesh, but sets values at the support nodes to the L2-norm
    from the origin.
    >>> import numpy as np
    >>> from pymfree.core.domain import Domain
    >>> from pymfree.core.function import L1Norm
    >>> from pymfree.core.function import DomainFunction
    >>> from pymfree.util.functional import l2
    >>> l2func = DomainFunction(l2)
    >>> my_domain = Domain(np.random.rand(1000,3), l2func, L1Norm)
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
                    "PyMfree Domain: Node value array invalid.")
            else:
                self.node_values = torch.tensor(
                    values[:len(self)], device=device)
        elif isinstance(values, torch.Tensor):
            if len(values) < len(self) or len(values.shape != 1):
                raise TypeError(
                    "PyMfree Domain: Node value array invalid.")
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
        r""" The length of the domain.

        Just the number of support points.

        Returns
        -------
        int
            Calls len(self.node_coordinates)
        """

        return len(self.node_coordinates)

    def __iter__(self):
        return self

    def __next__(self):
        r""" The iterator functionality.

         Moves through the support coordinates and returns them while
        incrementing an internal counter. Counter is reset as soon as the
        last coordinate is reached.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Single support coordinate with value while being iterated.
        """
        if self.__counter > len(self)-1:
            self.__counter = 0
            raise StopIteration
        else:
            self.__counter += 1
            return self.node_coordinates[
                self.__counter-1], self.node_values[self.__counter-1]

    def __str__(self):
        r""" String representation of the class.
        Shows information on the nodes and dimensons, as well as
        query information and evetual rescaling.

        Returns
        -------
        str
            Essential class information.
        """
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
        r""" Representation of the class.
        Shows information on the nodes and dimensons, as well as
        query information and evetual rescaling.

        Returns
        -------
        stdout
            Essential class information.
        """

        return print(self)

    def __call__(self, x):
        r""" Call operator for the class, sets a coordinate query.

        This is the main function for the Domain, which lets the user set
        a coordinate query. If domain is rescaled via rescale_coordinates,
        transformation is applied to query coordinates. x -> x*scale +shift.

        Parameters
        ----------
        x : list, numpy.ndarray or torch.Tensor
            The coordinates to be queried. If given as a scalar, the spatial
            dimensionality is set to the one of the Domain (see example).

        Returns
        -------
        neighbour_distances, neighbour_map : torch.Tensor, torch.Tensor
            The distances and indeces of the k nearest neighbours in the
            support domain for the query. k is set at construction.

        Raises
        ------
        TypeError
            If query is numpy array or torch tensor
            but not a PyMfree coordinate.
        TypeError
            If query is not a numpy array, torch tensor or list.

        See also
        --------
        pymfree.core.domain.Domain.rescale_coordinates()
        Shifts and scales the domain coordinates into a fixed window.

        Examples
        --------
        The following queries two points on a 3-d Domain:
        >>> import numpy as np
        >>> from pymfree.core.domain import Domain
        >>> my_domain = Domain(np.random.rand(100,3), np.random.rand(100))
        >>> my_domain([1., 2., 3., 4., 5., 6.])
        and the output will be index map and distances of shape (2, 1, 32),
        since the domain has k=32 by default. The 1 in the middle is PyMfree
        convention since the output is not a coordinate.
        """
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
            if isinstance(self.index, faiss.IndexFlatL2):
                self.neighbour_distances = torch.sqrt(self.neighbour_distances)
        else:
            self.neighbour_distances, self.neighbour_map = any_norm_nn_search(
                self.node_coordinates, self.query, self.norm, self.k)
        self.neighbour_distances.unsqueeze(1)
        self.neighbour_map.unsqueeze(1)

        return self.neighbour_distances, self.neighbour_map

    def __getitem__(self, index):
        r""" The square bracket opertator

        Returns the support coordinate at a given index position.

        Parameters
        ----------
        index : int
            The index position of the coordinate to be returned.

        Returns
        -------
        torch.Tensor
            The coorindate at the queried index.

        Raises
        ------
        IndexError
            If index out of bounds.
        """

        if not isinstance(index, int) or index > len(self)-1:
            raise IndexError("PyMfree Domain: Index out of bounds.")
        return self.node_coordinates[index]

    def rescale_coordinates(self, min_in=-1., max_in=1.):
        r""" Rescales support coordinates into given window.

        This routines loops through all spatial dimensions and finds the one
        which has the largest difference between the largest and smallest
        coorindate component. Then it rescales this distances into the interval
        max_in - min_in and shifts the minimum to min_in.

        Parameters
        ----------
        min_in : int, optional
            The new minimum location for the longest spatial dimension.
            Defaults to -1.
        max_in : int, optional
            The new maximum location for the longest spatial dimension.
            Defaults to -1.

        Returns
        -------
        No direct output, but Domain attributr self.scale is set accordingly.

        Raises
        ------
        TypeError
            If min_in or max_in are not given as floats.
        """

        if not isinstance(min_in, float) or not isinstance(max_in, float):
            raise TypeError("PyMfree Domain: domain bounds must be float.")
        self.scale['scale'], self.scale['shift'] = scale_params(
            self.coordinates, min_in, max_in)
        self.scale['unit'] = torch.tensor(
            max_in - min_in, device=self.device)
        self.coordinates *= self.scale['scale']
        self.coordinates += self.scale['shift']
        self.scale['rescale'] = True

    def to(self, device=torch.device('cpu')):
        r""" Transfers Domain to a given device.

        All data attributes are moved to provided device.

        Parameters
        ----------
        device : torch.device, optional
            The target torch device. Default is torch.device('cpu').

        Raises
        ------
        TypeError
            If device is not a torch device.
        """

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
        r""" Returns base Domain properties.

        The basic dimensionalities of the Domain.

        Returns
        -------
        dict
            Entries are 'nodes', number of support nodes. 'dim', the number
            of spatial dimensions and 'query', the number of query nodes.
        """
        if self.query_ready:
            helper = len(self.query_ready)
        else:
            helper = 0
        return {'nodes': len(self), 'dim': self.dim, 'query': helper}

    @property
    def dim(self):
        r""" The spatial dimension of the domain.

        Directly derived from the shape of the support coordinates.

        Returns
        -------
        int
            The spatial dimension of the support node coordinates.
        """
        return self.node_coordinates.shape[1]

    @property
    def query_ready(self):
        r""" Flag if query coordinates have been set.

        Returns
        -------
        bool
            True if query attribute exists, False otherwise.
        """
        return isinstance(self.query, torch.Tensor)

    @property
    def device(self):
        r""" The device the Domain resides on.

        Most data arrays have to be assigned to a specific device and
        this routines returns that device.

        Returns
        -------
        torch.device
            The Domain device, as inferred from the device of the support
            coordinate vector.
        """
        return self.node_coordinates.device


def any_norm_nn_search(support, query, norm, k=32):
    r""" Nearest neighbour search for arbitrary norm.

    Since not all norms are implemented in e.g. faiss, this offers a way
    of brute-force calculating nearest neighbours with any norm. Please be
    adviced that this can be orders of magnitude slower than a smart index
    search, even if that is flat.

    Parameters
    ----------
    support : torch.Tensor
        The support coordinates for the search. Must be PyMfree coordinate.
    query : torch.Tensor
        The query coordinates. Must be PyMfree coordinates
    norm : pymfree.core.function.Norm
        The PyMfree norm to be used. This is not recommended for standard
        norms auch L2, L1 or Linf, since those are implemented by default in
        many faster routines such as faiss or sklearn.neighbors.KdTree.
    k : int, optional
        The number of nearest neighbours to be searched. Defaults to 32.

    Returns
    -------
    distances, indices : torch.Tensor, torch.Tensor
        The distances and index postions of the k nearest neighbours for each
        query point. Output shape is hence (len(query), k).

    Raises
    ------
    TypeError
        If support or query are not PyMfree coordinates.
    TypeError
        If norm is not a PyMfree Norm.
    TypeError
        If k is not given as int.

    References
    ----------
    [1] [faiss](https://github.com/facebookresearch/faiss)
    [2] [Scikit-learn KdTree](
        https://scikit-learn.org/stable/modules/generated/\
            sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree)
    """
    if not check_pymfree_type(support)['coordinate']:
        raise TypeError("PyMfree nn search: Support must be coordinates")
    if not check_pymfree_type(query)['coordinate']:
        raise TypeError("PyMfree nn search: Query must be coordinates")
    if not isinstance(norm, Norm):
        raise TypeError(
            "PyMfree nn search: Norm must be PyMfree norm.")
    if not isinstance(k, int):
        raise TypeError("PyMfree nn search: k must be integer.")

    n = len(query)
    device = support.device
    result = torch.zeros(n, k, dtype=torch.float32, device=device)
    indices = torch.zeros(n, k, dtype=torch.int64, device=device)
    for i, element in enumerate(query):
        current = torch.sub(support, element)
        current = norm.no_checks(current)
        dists, numbers = torch.sort(current)
        result[i, :] = dists[: k]
        indices[i, :] = numbers[: k]

    return result, indices
