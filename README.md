# Overview

This library offers a collection of initialization schemes for neural networks based on recent
developments. Some parts of this library are based on [this other one][nninit] with the same name.

[nninit]: https://github.com/Kaixhin/nninit

# Supported Initialization Schemes

- A generalization of the variance-normalizing initialization schemes from [1], [2], and [3].
- The orthogonal initialization scheme, described in [4].
- Input preserving initialization schemes that are useful for pixel-labeling architectures:
   - Identity initialization for convolutional layers, described in [6].
   - Downsampling initialization for strided convolutional layers.
   - Bilinear upsamping initialization for transpose convolutional layers (known as
     `SpatialFullConvolution` in Torch -- a horrible name), described in [5].

[1]: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
"Efficient BackProp"

[2]: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
"Understanding the Difficulty of Training Deep Feedforward Neural Networks"

[3]: https://arxiv.org/abs/1502.01852
"Delving Deep Into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"

[4]: http://arxiv.org/abs/1312.6120
"Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks"

[5]: http://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
"Fully Convolutional Networks for Semantic Segmentation"

[6]: https://arxiv.org/abs/1511.07122
"Multi-Scale Context Aggregation by Dilated Convolutions"
