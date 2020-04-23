r""" functional.py part of pymfree/core

This collects bare-bones, torch-based functions.
"""

# Basic imports

import torch


def l1(x):
    r""" L_{1} norm

    Calculates the L1 norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.tensor
        A tensor of shape(n)
    """

    return torch.sum(torch.abs(x), axis=1)


def l2(x):
    r""" L_{2} norm

    Calculates the L2 norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.tensor
        A tensor of shape(n)
    """

    return torch.sqrt(torch.sum(x*x, axis=1))


def linf(x):
    r""" L_{\infty} norm

    Calculates the max norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.tensor
        A tensor of shape(n)
    """
    return torch.max(torch.abs(x)).values
