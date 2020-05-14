""" function.py part of pymfree/core

This module implements PyMFree functions. There are domain functions, which
transform coordinates into scalars, out of which we have Norms which carry no
parameters and DomainScalars which can carry parameters. Finally there are
RadialBasisFunctions, which convert scalars into scalars and can carry
parameters.
"""

import torch
import numpy
from pymfree.util.functional import check_functional
from pymfree.util.functional import l1
from pymfree.util.functional import l2
from pymfree.util.functional import linf
from pymfree.util.functional import l2l2


class Norm(object):
    r""" A Norm, converting a coordinate into a scalar.

    This class implements Norms, which are central elements in PyMFree since
    they convert coordinates to scalars while they obey the properties of a
    Norm (see References). They do not carry derivatives and have no
    parameters.

    Parameters
    ----------
    F : callable function
        The functional form for the metric. Must pass check_functional().

    numpy : bool, optional
        Flag indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.

    Raises
    ------
        various
            check_functional is carried out. See below.

    See also
    --------
    pymfree.util.functional
        For function implementations.
    pymfree.util.functional.check_functional
        Checks if the provided functional fulfills all requirements. Here this
        called with not requesting an input scalar but the output must be
        scalar.

    References
    ----------
    [1] [Wikipedia on Norms](https://en.wikipedia.org/wiki/Norm_(mathematics))

    """
    def __init__(self, F, numpy=False):

        self.F = check_functional(F)
        self.array_out = numpy

    def __str__(self):
        r""" String representation of the class.

        Shows class name and the functional form underlying the norm.

        Returns
        -------
        str
            Representation string.
        """
        one = str(self.__class__.__name__)
        one = one[one.rfind('.')+1:one.rfind('\'')]
        two = str(self.F).split(' ')[1]
        return str(one + "\n" + "Functional form: " + two)

    def __repr__(self):
        r""" Prints string representation.
        """

        print(self)

    def __call__(self, x1, x2=None):
        r""" The call operator for the norm.

        Calculates the norm for two batched coordinates. The functional form
        was defined at construction of the norm.

        Parameters
        ----------
        x1 : torch.Tensor or numpy.ndarray
            One set of coordinates. If given as numpy, conversion to torch
            follows.
        x2 : torch.Tensor or numpy.ndarray, optional
            A second set of coordinates. If given as numpy, conversion to torch
            follows. , by default None

        Returns
        -------
        torch.Tensor
            Calculates F(x1) or F(x1-x2), where F was defined at construction.
        numpy.ndarray
            If flag numpy is set at construction, output will be made as
            numpy array.

        Raises
        ------
        TypeError
            If x1 is not torch.Tensor or numpy.ndarray.
        TypeError
            If x2 is not None and not torch.Tensor or numpy.ndarray.
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

    def no_checks_pair(self, x1, x2):
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

    def no_checks(self, x):
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

    def sklearn_dist(self, x1, x2):
        x1 = torch.tensor(x1).unsqueeze(dim=0)
        x2 = torch.tensor(x2).unsqueeze(dim=0)
        return self(x1-x2).squeeze()


class L2Norm(Norm):
    r""" A fast-lane implementation of the l2-norm.

    This inherits from Norm and delivers the l2-norm by simply calling
    the general constructor with fixed function l2.

    Parameters
    ----------
    numpy : bool, optional
        Falg indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.


    See also
    --------
    pymfree.core.function.Norm

    References
    ----------
    [1] [L2-Norm on Wolfram](https://mathworld.wolfram.com/L2-Norm.html)
    r"""
    def __init__(self, numpy=False):
        super().__init__(l2, numpy)


class L2SquaredNorm(Norm):
    r""" A fast-lane implementation of the squared l2-norm.

    This inherits from L2Norm and delivers the l2-norm by simply calling
    the general constructor with fixed function l2l2.

    Parameters
    ----------
    numpy : bool, optional
        Falg indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.


    See also
    --------
    pymfree.core.function.Norm

    References
    ----------
    [1] [L2-Norm on Wolfram](https://mathworld.wolfram.com/L2-Norm.html)
    r"""
    def __init__(self, numpy=False):
        super().__init__(l2l2, numpy)


class L1Norm(Norm):
    r""" A fast-lane implementation of the l1-norm.

    This inherits from Norm and delivers the l1-norm by simply calling
    the general constructor with fixed function l1.

    Parameters
    ----------
    numpy : bool, optional
        Falg indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.

    See also
    --------
    pymfree.core.norm.Norm

    References
    ----------
    [1] [L1-Norm on Wolfram](https://mathworld.wolfram.com/L1-Norm.html)
    r"""
    def __init__(self, numpy=False):
        super().__init__(l1, numpy)


class LInfNorm(Norm):
    r""" A fast-lane implementation of the $l\infty$-norm.

    This inherits from Norm and delivers the $l\infty$-norm by simply calling
    the general constructor with fixed function linf.

    Parameters
    ----------
    numpy : bool, optional
        Falg indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.

    See also
    --------
    pymfree.core.norm.Norm

    References
    ----------
    [1] [L\infty-Norm on Wolfram]
        (https://mathworld.wolfram.com/L-Infinity-Norm.html)
    r"""
    def __init__(self, numpy=False):
        super().__init__(linf, numpy)


class RadialBasisFunction(object):
    r""" Transforms scalar into scalar.

    A radial basis function (RBF), which can carry external parameters.
    Could inherit from Norm,
    but does not in order order avoid overhead.

    Parameters
    ----------
    F : pymfree.util.functional
        The functional form of the RBF. Must pass check_functional().
    params : dict, optional
        A dict with a short parameter description as key and a function
        parameter as value. Defaults to None.
    numpy : bool, optional
        Flag indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.
    device : torch.device, optional
        The device on which the RBF exists. In practice, this means where
        the parameter tensor is stored. Can be changed with to(device)
        later on. Defaults to torch.device('cpu').

    Attributes
    ----------
    params : torch.Tensor
        A 1D torch tensor holding the class parameters. torch.tensor(0) if
        there are none.
    descr : list
        A list of strings that describe the class parameters. Empty list p[]
        if there are none.

    Raises
    ------
    various
        check_functional with scalar_in flag is carried out. See below.

    See also
    --------
    pymfree.util.functional
        For function implementations.
    pymfree.util.functional.check_functional
        Checks if the provided functional fulfills all requirements. Here this
        called requesting an input scalar and the output must be
        scalar.
    pymfree.core.function.RadialBasisFunction.to()
        Change the device of the RBF.

    References
    ----------
    [1] [Wikiepdia on radial basis functions]
            (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(
            self, F, params=None, numpy=False, device=torch.device('cpu')):
        self.F = check_functional(F, scalar_in=True)
        self.array_out = numpy
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError("PyMFree RBF: Parameters must be a dict.")
            self.descr, aux = zip(
                    *[(key, value) for key, value in params.items()])
            self.params = torch.tensor(aux, device=device)
        else:
            self.descr = []
            self.params = torch.tensor(0, device=device)

    def __str__(self):
        r""" String representation of the class.

        Shows class name and the functional form underlying the norm, followed
        by the parameter descriptions and values

        Returns
        -------
        str
            Representation string.
        """
        one = str(self.__class__.__name__)
        one = one[one.rfind('.')+1:one.rfind('\'')]
        two = str(self.F).split(' ')[1]
        three = str(self.get_params())
        out = (one + "\n" + "Functional form: " + two + "Parameters: " + three)
        return str(out)

    def __repr__(self):
        r""" Prints string representation.
        """

        print(self)

    def __call__(self, x):
        r""" RBF function call.

        Retuns the radial basis function values for the provided radii batch.

        Parameters
        ----------
        x : torch.tensor or numpy.ndarray
            The batch  of input radii. Must be a scalar.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            A batch of radial basis function values for the batch of radii.
            Output object type depends on array_out attribute.

        Raises
        ------
        TypeError
            If x is not a torch tensor or numpy array.
        TypeError
            If input is not a batch of scalars.
        """
        if not isinstance(x, torch.Tensor):
            if not isinstance(x, numpy.ndarray):
                raise TypeError(
                 "PyMFree RBF: Need torch tensor or numpy array.")
            else:
                x = torch.tensor(x, device=self.device)
        else:
            x.to(self.device)

        if len(x.shape) != 1:
            raise TypeError(
                "PyMFree RBF: Need scalar, batched torch Tensor of shape (n).")

        if self.array_out:
            return self.F(x).numpy()
        else:
            return self.F(x)

    @property
    def device(self):
        r""" Returns the device the RBF resides on.

        Queries torch device from parameter vector.

        Returns
        -------
        torch.device
        Torch device where the parameter vector is held.
        """
        return self.params.device

    def get_params(self):
        r""" Returns the class parameters and their descrptions as a dict.

        Packs the descr and params attributes into a dict and returns it.

        Returns
        -------
        dict
            {descr:params} pairs for all class parameters.
        """
        return {key: value for (key, value) in zip(self.descr, self.params)}

    def set_params(self, locs, values):
        r""" Changes the parameters of the function.

        Either single values or several parameters can be changed
        with one call.

        Parameters
        ----------
        locs : int or str, list of int or str
        The location of the parameter you want to change. Either the index in
        the parameter vector or its name stored in the description.
        variants can also be given as a list.

        Raises
        ------
        TypeError
            If format of locs and values does not match. Either one a list,
            the other a value or vice-versa.
        Warning to stdout
            If not all parameter names in str list are found.
        TypeError
            If Parameter locations are not str, int or list of strs and ints.
        TypeError
            If single provided parameter value is not float.
        """
        if isinstance(locs, list):
            if not isinstance(values, list) or len(locs) != len(values):
                raise TypeError(
                    "PyMFree RBF: Parameter values do not \
                                    fit your provided location.")
            if isinstance(locs[0], int):
                indeces = locs
            elif isinstance(locs[0], str):
                indeces = [
                    i for i, word in enumerate(self.descr) if word in locs]
                if(len(indeces) != len(locs)):
                    print("PyMFree RBF WARNING: Not all parameters in your \
                    list where found.")
            else:
                raise TypeError(
                    "PyMFree RBF: Parameter locs must be int or str.")
        else:
            if not isinstance(values, float):
                raise TypeError(
                    "PyMFree RBF: Provided Parameter value must be float.")
            if isinstance(locs, int):
                indeces = locs
            elif isinstance(locs, str):
                indeces = [
                    i for i, word in enumerate(self.descr) if word == locs]
            else:
                raise TypeError("PyMFree RBF: Parameter locations must be \
                                a list, str, or int.")

        self.params[indeces] = torch.tensor(values, device=self.device)

    def no_checks(self, x):
        r""" Raw wrap for direct self.F call.

        Parameters
        ----------
        x : torch.tensor
            The batch of input radii.

        Returns
        -------
        torch.Tensor
            A batch of radial basis function values for the batch of radii.
        """

        return self.F(x)

    def to(self, device=torch.device('cpu')):
        r""" Send RBF to a specific torch device.

        The params tensor is sent to the specfied device.

        Parameters
        ----------
        device : torch.device
            The torch device you want the RBF to live at.

        Raises
        ------
        TypeError
            If device is not of type torch.device

        Example
        -------
        rbf.to(device=torch.device('gpu:0'))
        """

        if not isinstance(device, torch.device):
            raise TypeError("PyMFree RBF: Need torch.device.")
        self.params = self.params.to(device)


class DomainFunction(RadialBasisFunction):
    r""" Transforms coordinates into scalars.

    A standard function on a Domain, which can carry external parameters.
    Inherits some basic functions from RadialBasisFunction.

    Parameters
    ----------
    F : pymfree.util.functional
        The functional form DomainFunction. Must pass check_functional().
    params : dict, optional
        A dict with a short parameter description as key and a function
        parameter as value. Defaults to None.
    numpy : bool, optional
        Flag indication if the output shall be a numpy array instead of a
        torch tensor. Default to False.
    device : torch.device, optional
        The device on which the RBF exists. In practice, this means where
        the parameter tensor is stored. Can be changed with to(device)
        later on. Defaults to torch.device('cpu').

    Attributes
    ----------
    params : torch.Tensor
        A 1D torch tensor holding the class parameters. torch.tensor(0) if
        there are none.
    descr : list
        A list of strings that describe the class parameters. Empty list p[]
        if there are none.

    Raises
    ------
        various
            check_functional is carried out. See below.
        TypeError
            If params is not None and not a dict.

    See also
    --------
    pymfree.util.functional
        For function implementations.
    pymfree.util.functional.check_functional
        Checks if the provided functional fulfills all requirements. Here this
        called with not requesting an input scalar but the output must be
        scalar.
    """

    def __init__(
            self, F, params=None, numpy=False, device=torch.device('cpu')):
        self.F = check_functional(F)
        self.array_out = numpy
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError(
                    "PyMFree DomainFunction: Parameters must be a dict.")
            self.descr, aux = zip(
                    *[(key, value) for key, value in params.items()])
            self.params = torch.tensor(aux, device=device)
        else:
            self.descr = []
            self.params = torch.tensor(0, device=device)

    def __call__(x, self):
        r""" DomainFunction call.

        Retuns the function values for the provided coordinate batch.

        Parameters
        ----------
        x : torch.tensor or numpy.ndarray
            The batch  of input coordinates. Must be a valid coordinate.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            A batch of function values for the batch of coordinates.
            Output object type depends on array_out attribute.

        Raises
        ------
        TypeError
            If x is not a torch tensor or numpy array.
        TypeError
            If input is not a batch of coordinates.
        """
        if not isinstance(x, torch.Tensor):
            if not isinstance(x, numpy.ndarray):
                raise TypeError(
                 "PyMFree DomainFunction: Need torch tensor or numpy array.")
            else:
                x = torch.tensor(x, device=self.device)
        else:
            x.to(self.device)

        if len(x.shape) != 2:
            raise TypeError(
                "PyMFree DomainFunction: Need torch Tensor of shape (n,d).")

        if self.array_out:
            return self.F(x).numpy()
        else:
            return self.F(x)
