# pymfree
Mesh free operations based on PyTorch. 

This mainly implements the ideas described [here](https://www.colorado.edu/amath/bengt-fornberg-0).


## Data, index and coordinate conventions
1. Every quantity in this package shall be represented as a PyTorch tensor. 

2. Every quantity in this package shall be batched, meaning that the leading array index is always the batch/sample index.

3. Scalars shall only carry the batch index. The shape of a single scalar is hence (1,) and the shape of n scalars is (n,)

4. Coordinates carry their components in the second index. A single 1D coordinate hence has shape (1,1), a batch of n 3D coordinates has shape (n,3).

## Functions
1. Radial Basis Functions (RBFs) are a basic  component to this package. They work on batches of scalars. 
2. Norms are another main compoment to this package an they work on batches of coordinate pairs. Since pymree uses the excellent kdtree routines from sklearn, there must be some consistency between the sklearn norms and the package ones. 


Every vector(tensor) in this library carries at least one index. This leading index is the sample index and separates unique objects. Think about a row in a table or a single image in a collection. If the object is dimensionless  the sample index is the only index. An example would be a radius. A single radius has shape (1), a collection of n radii has shape (n). The second index entails the spatial coordinates of the object. In 3D the shape of a coordinate is (1,3) and the shape of n coordinates is (n,3).
 
