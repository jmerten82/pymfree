# rbf.py
# part of pymfree

import torch

"""
This module does the following thing...
"""


def ga(r, params):
    return torch.exp(-torch.pow(params[0]*r), 2)


def ga_dx(r, params):
    return 0


class RBF(object):
    """
    This is a docstring.
    """

    def __init__(self, function, params=None):
        if not callable(function):
            raise TypeError("PyMFree RBF: Callable function needed.")

        self.f = function
        self.description = []
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError("PyMFree RBF: Params must be given as dict")
            else:
                aux_value = []
                for key, value in params.items():
                    self.description.append(key)
                    aux_value.append(value)
                self.params = torch.tensor(aux_value)

        else:
            self.params = torch.empty(0)

        self.derivatives = {}

        # Build function check, if it produces errors with params.

    def __call__(self):
        pass

    def add_derivative(self):
        pass

    def __getitem__(self, selection):
        pass
