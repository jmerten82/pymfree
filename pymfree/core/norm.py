# norm.py
# Part of pymfree

"""
This module does the following thing...
"""

import torch


# Lowest level function definitions
def l1(x):
    """
    Calculates the l1 norm of a batch of n samples of tensor of dimension d pairs. 
    Args:
        x torch.tensor of shape (n,d)
    Returns:
        torch tensor of shape (n) calculating \Sum_{n=0}^{d-1}(|x|)
    """
    return torch.sum(torch.abs(x),axis=1)

def l2(x):
    """
    Calculates the l2 norm of a batch of n samples of tensor of dimension d pairs. 
    Args:
        x torch.tensor of shape (n,d)
    Returns:
        torch tensor of shape (n) calculating \sqrt(\Sum_{n=0}^{d-1}(x^2))
    """
    return torch.sqrt(torch.sum(x*x,axis=1))

def linf(x):
    """
    Calculates the l_{\infty} norm of a batch of n samples of tensor of dimension d pairs. 
    Args:
        x torch.tensor of shape (n,d)
    Returns:
        torch tensor of shape (n) calculating \max(|x_{n}|) for n \in (0,d=1)
    """
    return torch.max(torch.abs(x)).values

class norm(object):
    """
    This is a docstring.
    """
    def __init__(self,F):
        if not callable(F):
            raise TypeError("PyMFree Norm: Function not callable")
        x1 = torch.rand(10,3)
        x2 = torch.rand(10,3)
        y = F(x1,x2)
        if not isinstance(y,torch.tensor):
            raise TypeError("PyMFree Norm: Function does not deliver torch.tensor.")
        if len(y.shape) is not 1:
            raise RuntimeError("PyMFree Norm: Function does not deliver scalars.")
        if y.shape[0] is not 10:
            raise RuntimeError("PyMFree Norm: Function delivers wrong sample number.")

        self.F = F

    def __call__(self,x1,x2=None):
        """
        This is a docstring.
        """

        if not isinstance(x1,torch.tensor):
            raise TypeError("PyMFree Norm: Need torch Tensor.")
        if not len(x1.shape) is not 2:
            raise TypeError("PyMFree Norm: Need torch Tensor of shape (n,d).")
        if x2 is not None:
            if not isinstance(x2,torch.tensor):
                raise TypeError("PyMFree Norm: Need torch Tensor.")
            if not len(x2.shape) is not 2:
                raise TypeError("PyMFree Norm: Need torch Tensor of shape (n,d).")
            return self.F(x1-x2)
        else:
            return self.F(x1)

    def NoChecksPair(self,x1,x2):
        """
        This is a docstring.
        """

        return self.F(x1-x2)

    def NoChecksSingle(self,x):
        """
        This is a docstring.
        """
        return self.F(x)
