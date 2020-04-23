# pymfree
Mesh free operations based on PyTorch. 

This mainly implements the ideas described [here](https://www.colorado.edu/amath/bengt-fornberg-0).


## Data, index and coordinate conventions
1. Every quantity in this package shall be represented as a PyTorch tensor. 

2. Every quantity in this package shall be batched, meaning that the leading array index is always the batch/sample index.

3. Scalars shall only carry the batch index. The shape of a single scalar is hence (1,) and the shape of n scalars is (n,)

4. Coordinates shall carry two indices and store components in the second index. A single 1D coordinate hence has shape (1,1), a batch of n 3D coordinates has shape (n,3).

5. Any other quantity shall carry three or more indices, where the second
index is set to 1. For example a batch of n samples of NxM matrices shall 
have shape (n,1,N,M).

6. The leading index in a matrix represents the rows and the second index the columns 

## Functions
1. Radial Basis Functions (RBFs) work on batches of scalars and return batches of scalars. 
2. Norms (distances) work on batches of coordinates and return batches of scalars. 
3. Scalar functions are doing the same as Norms, but we won't call them that way, most importantly, they will have derivatives.

## Coding style
We will strictly following [PEP 8](https://www.python.org/dev/peps/pep-0008/) enforced by [flake8](https://flake8.pycqa.org/en/latest/) linter.

## Docstring style
Tbtk follows [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). A full example can bee seen in [example.py](https://numpydoc.readthedocs.io/en/latest/example.html#example).

## Naming-style
Classes are written in capitals and no underscores, e.g. DnATable. Class methods and functions are minor with underscores used. E.g. do_this_to_a_table(*). Names of variables shall be as descriptive as possible. Otherwise, add a single line docstring. 
 
