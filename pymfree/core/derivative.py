""" derivative.py part pymfree/core

This module provides classes and helper functions that deal with
derivatives. Their functional implementation and properties.
"""

# Initial imports

import numpy as np
import torch
from pymfree.core.function import DomainFunction
from pymfree.util.utils import check_pymfree_type
from pymfree.util.polynomial import count_dim_contrib


class LinearDerivative(object):
    r""" A linear derivative operator in PyMfree.

    Can have several additive components, stored separately. A functional
    form of the derivative can be provided via a DomainFunction.

    Parameters
    ----------
    signature : str
        The description of the operator via a signature. Must be a string. The
        actual signature must be between two '$' characters. Each components
        within the signature starts with a prefactor and then the derivative
        components within round brackets. See example later on.

    function : pymfree.core.function.DomainFunction, optional
        If wanted, a functional form of a derivative can be provided, which is
        of course not a dreivative operator. Defaults to None.

    Raises
    ------
    TypeError
        If function is given and not a DomainFunction or None.

    See also
    --------
    pymfree.core.derivative.derivative_parser
        Reads derivative signatures.
    """

    def __init__(self, signature, function=None):
        self.signature, comps = derivative_parser(signature)
        if not isinstance(function, DomainFunction) and function is not None:
            raise TypeError("LinearDerivative: Implementation of\
                 derivative functional form must be DomainFunction.")
        self.F = function
        self.components = [DerivativeComponent(comp) for comp in comps]

    def __call__(self, x):
        r""" Applies the eventually provided derivative to a coordinate.

        This only makes sense if a DomainFunction was provided. Otherwise a
        0 scalar is returned.

        Parameters
        ----------
        x : torch.Tensor
            The coordinates the derivative should be applied to. Must be
            pymfree coordinate.

        Returns
        -------
        torch.Tensor
            A pymfree scalar with F(x), given a F was provided as
            DomainFunction at construction. Otherwise Zeroes are returned.

        Raises
        ------
        TypeError
            If x is not a pymfree coordinate.
        """
        if not check_pymfree_type(x)['coordinate']:
            raise TypeError("LinearDerivative: Input is not a coordinate.")
        return self.F(x)

    def __len__(self):
        r""" Length operator.

        Returns the number of derivative components.

        Returns
        -------
        int
            The length of the components vector, so the number of derivative
            components.
        """

        return len(self.components)

    def __repr__(self):
        r""" The class representation.

        Prints the class string to stdout.

        Returns
        -------
        stdout
            Returns class string.
        """
        return self.__str__()

    def __str__(self):
        r""" The string representation of the class.

        Provides information on the total derivative signature and the
        attached function form.

        Returns
        -------
        str
            The string representation of the class.
        """
        one = "Signature\n"
        one += "---------\n"
        one += self.signature + "\n\n"
        if self.F is not None:
            one += "Function\n"
            one += "--------"
            one += str(self.F.F.__name__)
        return str(one)


class DerivativeComponent(object):
    r""" A representation of derivative components.

    With component we mean a closed derivative operator of a certain order.
    E.g. the 1/2*(d^2 / dx^2) in the 2D Laplacian.

    Parameters
    ----------
    signature : str
        A valid derivative component signature. E.g. 0.5(0, 0)

    Attributes
    ----------
    factor : float
        The factor in front of the derivative. E.g. a 1/2 in 2D Laplacian.
    component_vector : numpy.ndarray
        A vector showing the derivative order of in each relevant component.
        E.g. d^2/dxdz would be [1, 0, ,1]

    See also
    --------
    pymfree.core.derivative.derivative_component_parser
        Reads the str signatures of derivative components.
    """

    def __init__(self, signature):
        self.signature, \
         self.factor, \
         self.component_vector = derivative_component_parser(signature)
        self.component_vector = torch.tensor(self.component_vector)
        self.component_vector = count_dim_contrib(
            self.component_vector.unsqueeze(0), [0, 1, 2, 3, 4])

    def __len__(self):
        r""" The order of the derivative component.
        """
        return len(self.component_vector)

    @property
    def order(self):
        return np.sum(self.component_vector[:, 1])

    @property
    def max_dim(self):
        return self.component_vector[-1, 0]

    def dim_vector(self, dim):
        dim_vector = torch.zeros(dim, dtype=torch.int64)
        for line in self.component_vector:
            dim_vector[line[0]] = torch.from_numpy(line)[1]
        return dim_vector

    def max_factor(self):
        current = self.factor
        for line in self.component_signature:
            current *= np.math.factorial(line[1])
        return current


def derivative_parser(input):
    if not isinstance(input, str):
        raise TypeError("derivative_parser: Signature must be a string.")
    if input.count('$') < 2:
        raise TypeError("derivative_parser: Input signature format invalid.")

    # Removing all white spaces
    input = input.replace(' ', '')

    # Extracting functional part, according to $-convention
    signature = input.split('$')[1]
    if signature.count('(') == 0:
        raise TypeError("derivative_parser: No derivative in signature.")
    if not signature.count('(') == signature.count(')'):
        raise TypeError(
            "derivative_parser: Bracket count in signature invalid.")

    # Extracting components, according to ()-convention
    component_signature = []
    component_signature.append(signature[0:signature.find(')')+1])
    temp = signature[signature.find(')')+1:]
    index = temp.find('(')
    while index > 0:
        component_signature.append(temp[0:temp.find(')')+1])
        temp = temp[temp.find(')')+1:]
        index = temp.find('(')

    return signature, component_signature


def derivative_component_parser(input):
    if not isinstance(input, str):
        raise TypeError(
            "derivative_component_parser: Signature must be a string.")
    if input.count('(') != 1 and input.count(')') != 1:
        raise TypeError(
         "derivative_component_parser: Input signature contains no brackets.")
    if input.count(',') == 0:
        raise TypeError(
         "derivative_component_parser: Input signature contains no colons.")
    signature = input.replace(' ', '')
    factor = float(signature[0:signature.find('(')])
    temp = signature[signature.find('(')+1:signature.find(')')]
    indices = temp.split(',')
    # Little list comprehension the end and convert
    indices = [int(element) for element in indices]

    return signature, factor, indices
