# pymfree
Mesh free operations based on PyTorch. 

This mainly implements the ideas described [here](https://www.colorado.edu/amath/bengt-fornberg-0).


## Index/Coordinate conventions
Every vector(tensor) in this library carries at least one index. This leading index is the sample index and separates unique objects. Think about a row in a table or a single image in a collection. If the object is dimensionless  the sample index is the only index. An example would be a radius. A single radius has shape (1), a collection of n radii has shape (n). The second index entails the spatial coordinates of the object. In 3D the shape of a coordinate is (1,3) and the shape of n coordinates is (n,3).