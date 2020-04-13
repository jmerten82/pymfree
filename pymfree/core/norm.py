# This will have to contain: l1, l2, lp, linf....

import torch

def l1(X1,X2):
    return torch.sum(torch.abs(X1-X2),axis=1)

def l2(X1,X2):
    return torch.sqrt(torch.sum(torch.pow(X1-X2,2),axis=1))

def linf(X1,X2):
    return torch.max(torch.abs(X1-X2)).values

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
        pass

    def NoChecksPair(self,x1,x2):
        return self.F(x1,x2)
