# norm.py
# Part of pymfree

"""
This module does the following thing...
"""

import torch


class Norm(object):
    """
    This is a docstring.
    """
    def __init__(self, F):
        if not callable(F):
            raise TypeError(
                "PyMFree Norm: Function not callable")
        x1 = torch.rand(10, 3)
        x2 = torch.rand(10, 3)
        y = F(x1, x2)
        if not isinstance(y, torch.tensor):
            raise TypeError(
                "PyMFree Norm: Function does not deliver torch.tensor.")
        if len(y.shape) != 1:
            raise RuntimeError(
                "PyMFree Norm: Function does not deliver scalars.")
        if y.shape[0] != 10:
            raise RuntimeError(
                "PyMFree Norm: Function delivers wrong sample number.")

        self.F = F

    def __call__(self, x1, x2=None):
        """
        This is a docstring.
        """

        if not isinstance(x1, torch.tensor):
            raise TypeError("PyMFree Norm: Need torch Tensor.")
        if not len(x1.shape) != 2:
            raise TypeError("PyMFree Norm: Need torch Tensor of shape (n,d).")
        if x2 is not None:
            if not isinstance(x2, torch.tensor):
                raise TypeError("PyMFree Norm: Need torch Tensor.")
            if not len(x2.shape) != 2:
                raise TypeError(
                    "PyMFree Norm: Need torch Tensor of shape (n,d).")
            return self.F(x1-x2)
        else:
            return self.F(x1)

    def NoChecksPair(self, x1, x2):
        """
        This is a docstring.
        """

        return self.F(x1-x2)

    def NoChecksSingle(self, x):
        """
        This is a docstring.
        """
        return self.F(x)


class l2Norm(Norm):
    """
    This is a docstring.
    """
    def __init__(self):
        super().__init__(l2)


class l1Norm(Norm):
    """
    This is a docstring.
    """
    def __init__(self):
        super().__init__(l1)


class linfNorm(Norm):
    """
    This is a docstring.
    """
    def __init__(self):
        super().__init__(linf)
