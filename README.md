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

 
