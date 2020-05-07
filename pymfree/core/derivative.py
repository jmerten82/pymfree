""" derivative.py part pymfree/core

This module provides classes and helper functions that deal with
derivatives. Their functional implementation and properties.
"""

# Initial imports

import numpy as np
import torch
from pymfree.core.norm import DomainFunction


class LinearDerivative(object):

    def __init__(self, signature, function):
        self.signature, comps = derivative_parser(signature)
        if not isinstance(function, DomainFunction):
            raise TypeError("LineaeDerivative: Implementation of\
                 derivative functional form must be DomainFunction.")
        self.F = function

# MAKE THIS A LIST COMPREHESION
        self.components = []
        for comp in comps:
            self.components.append(DerivativeComponent(comp))

    def __call__(self, x):
        return self.F(x)

    def __len__(self):

        return len(self.components)

    def __str__(self):
        print("Signature")
        print("---------")
        print(self.signature, end="\n")


class DerivativeComponent(object):

    def __init__(self, signature):
        self.signature, \
         self.factor, \
         self.component_vector = derivative_component_parser(signature)

        self.component_vector = np.array(self.component_vector)
        self.component_vector = np.stack(
            np.unique(self.component_vector, return_counts=True), axis=-1)

    def __len__(self):
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
