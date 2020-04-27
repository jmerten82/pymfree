""" norm.py part of pymfree/core

This module implements PyMFree norms, which transform coordinates into
scalars.
"""

import torch
import numpy
from pymfree.core.functional import l1
from pymfree.core.functional import l2
from pymfree.core.functional import linf


class Norm(object):
    r""" A Norm, converting a coordinate into a scalar.

    This class implements Norms, which are central elements in PyMFree since
    they convert coordinates to scalars while they obey the properties of a
    Norm (see References). They do not carry derivatives and have no
    parameters.

    Parameters
    ----------
    F : callable function
        The functional form for the metric. Must be a function that takes
        two torch tensors or numpy arrays of coordinates and retuns a batch
        of Scalars. This is tested at construction.

    numpy : bool, optional
        Falg indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.

    Raises
    ------
        TypeError:
            If F is  not callable.
        TypeError:
            If a test-call of F feeding two (10,3) tensors does not deliver
            tensor.
        RuntimeError:
            If a test-call of F feeding two (10,3) tensors does not deliver
            tensor of length 10.
        RuntimeError:
            If a test-call of F feeding two (10,3) tensors does not deliver
            scalar of length 10.

    See also
    --------
    pymfree.core.functional
        For function implementations.

    References
    ----------
    [1] [Wikipedia on Norms](https://en.wikipedia.org/wiki/Norm_(mathematics))

    """
    def __init__(self, F, numpy=False):
        if not callable(F):
            raise TypeError(
                "PyMFree Norm: Function not callable")
        x1 = torch.rand(10, 3)
        x2 = torch.rand(10, 3)
        y = F(x1-x2)
        if not isinstance(y, torch.Tensor):
            raise TypeError(
                "PyMFree Norm: Function does not deliver torch.tensor.")
        if len(y.shape) != 1:
            raise RuntimeError(
                "PyMFree Norm: Function does not deliver scalars.")
        if y.shape[0] != 10:
            raise RuntimeError(
                "PyMFree Norm: Function delivers wrong sample number.")

        self.F = F

        self.array_out = numpy

    def __call__(self, x1, x2=None):
        r""" The call operator for the norm.

        Calculates the norm for two batched coordinates. The functional form
        was defined at construction of the norm.

        Parameters
        ----------
        x1 : torch.tensor or numpy.ndarray
            One set of coordinates. If given as numpy, conversion to torch
            follows.
        x2 : torch.tensor or numpy.ndarray, optional
            A second set of coordinates. If given as numpy, conversion to torch
            follows. , by default None

        Returns
        -------
        torch.tensor
            Calculates F(x1) or F(x1-x2), where F was defined at construction.
        numpy.ndarray
            If flag numpy is set at construction, output will be made as
            numpy array.

        Raises
        ------
        TypeError
            If x1 is not torch.Tensor or numpy.ndarray.
        TypeError
            If x2 is not None and not torch.tTensor or numpy.ndarray.
        TypeError
            If x1 is not of form (n,d), where n is the number of samples and
            d the spatial dimensionality.
        TypeError
            If x2 is not None and not of form (n,d),
            where n is the number of samples and d the spatial dimensionality.
        r"""

        if not isinstance(x1, torch.Tensor):
            if not isinstance(x1, numpy.ndarray):
                raise TypeError(
                    "PyMFree Norm: Need torch tensor or numpy array.")
            else:
                x1 = torch.tensor(x1)

        if len(x1.shape) != 2:
            raise TypeError("PyMFree Norm: Need torch Tensor of shape (n,d).")
        if x2 is not None:
            if not isinstance(x2, torch.Tensor):
                if not isinstance(x2, numpy.ndarray):
                    raise TypeError(
                        "PyMFree Norm: Need torch tensor or numpy array.")
                else:
                    x2 = torch.tensor(x2)

            if len(x2.shape) != 2:
                raise TypeError(
                    "PyMFree Norm: Need torch Tensor of shape (n,d).")
            if self.array_out:
                return self.F(x1-x2).numpy()
            else:
                return self.F(x1-x2)
        else:
            if self.array_out:
                return self.F(x1).numpy()
            else:
                return self.F(x1)

    def NoChecksPair(self, x1, x2):
        r""" A very bare-bones implementation of the pair Norm calculation.

        This deliverd the same as the ()-operator on coordinate
        pairs but without any checks and conversions. x1 and x2 must be
        torch tensors of adequate shape (n, d), where n is the number of
        samples and d the spatial dimensionality.

        Parameters
        ----------
        x1 : torch.Tensor
            Of shape (n, d) a batch of coordinates.
        x2 : [type]
            Of shape (n, d) a batch of coordinates.

        Returns
        -------
        torch.Tensor
            Calculates F(x1-x2)
        r"""

        return self.F(x1-x2)

    def NoChecksSingle(self, x):
        r""" A very bare-bones implementation of the Norm calculation.

        This deliverd the same as the ()-operator on batch of coordinates
        but without any checks and conversions. x must be torch tensor
        of adequate shape (n, d), where n is the number of samples and d the
        spatial dimensionality.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n, d) a batch of coordinates.

        Returns
        -------
        torch.Tensor
            Calculates F(x)
        r"""
        return self.F(x)


class L2Norm(Norm):
    r""" A fast-lane implementation of the l2-norm.

    This inherits from Norm and delivers the l2-norm by simply calling
    the general constructor with fixed function l2.

    See also
    --------
    pymfree.core.norm.Norm

    References
    ----------
    [1] [L2-Norm on Wolfram](https://mathworld.wolfram.com/L2-Norm.html)
    r"""
    def __init__(self):
        super().__init__(l2)


class L1Norm(Norm):
    r""" A fast-lane implementation of the l1-norm.

    This inherits from Norm and delivers the l1-norm by simply calling
    the general constructor with fixed function l1.

    See also
    --------
    pymfree.core.norm.Norm

    References
    ----------
    [1] [L1-Norm on Wolfram](https://mathworld.wolfram.com/L1-Norm.html)
    r"""
    def __init__(self):
        super().__init__(l1)


class LInfNorm(Norm):
    r""" A fast-lane implementation of the $l\infty$-norm.

    This inherits from Norm and delivers the $l\infty$-norm by simply calling
    the general constructor with fixed function linf.

    See also
    --------
    pymfree.core.norm.Norm

    References
    ----------
    [1] [L\infty-Norm on Wolfram]
        (https://mathworld.wolfram.com/L-Infinity-Norm.html)
    r"""
    def __init__(self):
        super().__init__(linf)
