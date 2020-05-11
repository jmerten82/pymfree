r""" functional.py part of pymfree/core

This collects bare-bones, torch-based functions.
"""

# Basic imports

import torch


def check_functional(f, scalar_in=False, scalar_out=True):
    r""" Check lowest level functions

    Checks if a pymfree functional is valid. Meaning that it accepts parameters
    and delivers batched results.

    Parameters
    ----------
    f : callable
        The function to be tested.
    scalar_in : bool, optional
        If this is set, the functional is tested with a scalar. Defaults to
        False.
    scalar_out : bool, optional
        Flag if output is checked to be a scalar. Defaults to True.

    Returns
    -------
    callable
        The tested function if passed tests.

    Raises
    ------
    TypeError
        If function does not accept two arguments.
    RuntimeError
        If exception is raised when called with coordinate batch and
        parameter vector.
    TypeError
        If no torch.Tensor is produced.
    TypeError
        If the output batch has the wrong length.
    TypeError
        If scalar flag is set and the output is not a scalar.
    r"""
    if scalar_in:
        test_x = torch.rand(10)
    else:
        test_x = torch.rand(10, 3)

    try:
        f(torch.rand(1, 1), torch.rand(1, 1))
    except Exception as e:
        if str(e).find("positional argument but 2 were given"):
            raise TypeError("check_functional: need two arguments.")
    try:
        result = f(test_x, torch.rand(5))

    except Exception:
        raise RuntimeError("check_functional: was not able to call funtional.")
    if not isinstance(result, torch.Tensor):
        raise TypeError("check_functional: No torch tensor produced.")
    if len(result) != 10:
        raise TypeError("check_functional: Output batch has wrong size.")
    if scalar_out:
        if len(result.shape) != 1:
            raise TypeError("check_functional: Output is not a scalar.")

    return f


def l1(x, params=None):
    r""" L_{1} norm

    Calculates the L1 norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.Tensor
        A tensor of shape(n)
    """

    return torch.sum(torch.abs(x), axis=1)


def l2(x, params=None):
    r""" L_{2} norm

    Calculates the L2 norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.tensor
        A tensor of shape(n)
    """

    return torch.sqrt(torch.sum(x*x, axis=1))


def linf(x, params=None):
    r""" L_{\infty} norm

    Calculates the max norm of a batch of n samples of d-dimensional tensors.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.Tensor
        A tensor of shape(n)
    """
    return torch.max(torch.abs(x)).values


def l2l2(x, params=None):
    r""" L_{2} norm

    Calculates the squared L2 norm of a batch of n samples
    of d-dimensional tensors.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape(n,d)

    Returns
    -------
    torch.tensor
        A tensor of shape(n)
    """

    return torch.sum(x*x, axis=1)


def ga(r, params):
    r""" A Gaussian radial basis function with shape parameter $\epsilon$,

    $f(x)=\exp\left(-(\epsilon r)^{2}\right)$

    Parameters
    ----------
    r : torch.Tensor
        Radii to be evaluated. Shape is (n), where n is the number of samples
    params : list
        A list  of torch tensors. Needed for this function is
        * params[0] = torch.tensor(epsilon)

    Returns
    -------
    torch.Tensor
        f(r) for all r. Shape of output is (n), if n was the number radii
        provided.
    """

    return torch.exp(-torch.pow(params[0]*r), 2)
