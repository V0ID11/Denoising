Use of sliding window setup to predict the class label of each pixel

- Allows the network to localize
- Training data of patches is much larger than the number of training images
- Can be quite slow doing it this way due to redundancy from overlapping patches
- Trade-off between localisation accuracy and the use of context.
- Larger patches require more max pooling layers that reduce the localization accuracy
- Whilst smaller patches only see little context
- Recent approaches propose a classifier output that "takes" into account the features from multiple layers

- Using "fully convolutional network" is the more elegant solution
- Replace pooling operators with upsampling operators
- Layers increase the resolution of the output
- Expansive path is symmetric to the contracting path, yields a u-shaped architecture
- No fully connected layers, only using the valid part of each convolution
