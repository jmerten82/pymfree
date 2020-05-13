""" polynomial.py part of pymfree/util

This module provides classes and functions that handle polynomial support
for mesh free differentiation and interpolation.
"""

# Initial imports

import torch


class PolynomialSupport(object):
    r""" ...

    ...
    """

    def __init__(self, dim, pdeg):
        if not isinstance(dim, int) or pdeg < 0:
            raise TypeError("PolynomialSupport: Dim must be int > 0.")
        if not isinstance(pdeg, int) or pdeg < 0:
            raise TypeError("PolynomialSupport: Dim must be int > 0.")
        self.polynomials = get_poynomials(dim, pdeg)

    def __len__(self):
        return len(self.polynomials)

    def shape(self):
        return {"dim": self.polynomials.shape[1],
                "pdeg": torch.max(self.polynomials).item(),
                "terms": len(self)}

    def embed_AP(self, coords, AP):
        # Careful, coords must have origin at query point
        if not isinstance(coords, torch.Tensor):
            raise TypeError(
                "PolynomialSupport: Coordinate input must be torch tensor.")
        if not isinstance(AP, torch.Tensor):
            raise TypeError(
                "PolynomialSupport: Coordinate input must be torch tensor.")
        if len(coords.shape) != 4 and len(AP.shape) != 4:
            raise TypeError(
                "PolynomialSupport: Need batches of PyMfree special objects.")
        if coords.shape[3] != self.shape['dim']:
            raise TypeError(
                "PolynomialSupport: Spatial dimensions don't match.")
        if AP.shape[2] < coords.shape[2] or AP.shape[2] < len(self):
            raise TypeError("PolynomialSupport: Given AP matrices too small.")
        AP[:, :, :coords.shape[2], len(self):] = poly_AP_contrib(
                                                    coords, self.polynomials)
        return AP

    def derivative_component(self, derivative):
        # Keep in mind that this refers to shifted coordinates to query point.
        pass


def get_poynomials(dim, pdeg):
    dims = torch.range(0, dim-1, dtype=torch.int64)
    out = torch.zeros(1, 3, dtype=torch.int64)
    for i in range(1, pdeg+1):
        out = torch.cat(
            (out, count_dim_contrib(
                torch.combinations(dims, with_replacement=True, r=i), dims)))
    return out


def count_dim_contrib(terms, dims):
    stack = []
    for dim in dims:
        stack.append((terms == dim).sum(axis=1))
    return torch.stack(stack, 1)


def poly_AP_contrib(coords, polynomials):
    out = []
    for exps in polynomials:
        out.append(torch.pow(coords, exps).prod(3, keepdim=True))
    return torch.cat(out, 3)
