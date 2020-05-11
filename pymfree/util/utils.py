r""" utils.py part of pymfree/util

This collects utility functions.
"""

# Basic imports

import torch
from pymfree.core.function import Norm


def any_norm_nn_search(support, query, norm, k=32):
    """
    Needs a docstring.
    """
    if not isinstance(support, torch.Tensor):
        raise TypeError("PyMfree nn search: Support must be a tensor.")
    if not isinstance(query, torch.Tensor):
        raise TypeError("PyMfree nn search: Query must be a tensor.")
    if len(support.shape) != 2 or len(query.shape) != 2:
        raise TypeError(
            "PyMfree nn search: Query and support must be coordinates")
    if not isinstance(norm, Norm):
        raise TypeError(
            "PyMfree nn search: Norm must be PyMfree norm.")
    if not isinstance(k, int):
        raise TypeError("PyMfree nn search: k must be integer.")

    n = len(query)
    result = torch.zeros(n, k, dtype=torch.float32)
    indices = torch.zeros(n, k, dtype=torch.int64)
    for i, element in enumerate(query):
        current = torch.sub(support, element)
        current = norm.no_checks(current)
        dists, numbers = torch.sort(current)
        result[i, :] = dists[: k]
        indices[i, :] = numbers[: k]
    return result, indices


def scale_params(coordinates, min_in=-1., max_in=1.):
    axis = torch.argmax(
        torch.max(coordinates, axis=0).values - torch.min(
            coordinates, axis=0).values).item()
    cmax = torch.max(coordinates[:, axis])
    cmin = torch.min(coordinates[:, axis])
    dist = cmax - cmin
    target_dist = max_in - min_in
    scale = target_dist/dist
    coordinates = coordinates*scale
    cmin = torch.min(coordinates[:, axis])
    shift = min_in - cmin
    return scale, shift
