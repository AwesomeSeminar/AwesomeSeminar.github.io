---
layout: post
title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
mathjax: true
---

chollet 对 separable convs 的评论：

![](http://oi3xms5do.bkt.clouddn.com/fchollet_t1.png)

![](http://oi3xms5do.bkt.clouddn.com/fchollet_t2.png)

**Related Resources:**

- [MobileNetV1 (Howard et al. 2017)](https://arxiv.org/abs/1704.04861)
- [MobileNetV2 (Sandler et al. 2018)](https://arxiv.org/abs/1801.04381)

# 1 Intro

MobileNet 从实际出发，考虑 mobile and embedded applications 中深度学习模型应用效率的问题。

MobileNet is specifically tailored for mobile and resource constrained environments. It pushes SOTA for mobile tailored CV models, by significantly decreasing the number of operations and memory needed while retaining the same accuracy.

MobileNet is built primarily from depthwise separable convolutions initially introduced in [(Sifre. 2014)](https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf) and subsequently used in Inception models[(Ioffe and Szegedy. 2015)](https://arxiv.org/pdf/1502.03167.pdf) to reduce the computation in the first few layers.

![](http://oi3xms5do.bkt.clouddn.com/mobilenets.png)

# 2 Depthwise Separable Convolution

## 2.1 Two Layers

> 分而治之

The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into two separate layers:

**Layer 1 - Depthwise Convolution:** performs lightweight filtering by applying a single convolutional filter per input channel.

**Layer 2 - Pointwise Convolution:** a 1x1 convolution which is responsible for building new features through computing linear combinations of the input channels.

![](http://oi3xms5do.bkt.clouddn.com/separable_conv.png)

## 2.2 Comparison with Standard Convolution

![](http://oi3xms5do.bkt.clouddn.com/replace_conv.png)

**Standard Convolution:** filters and combines input in one step:

$$
\begin{cases}
	\text{input feature map } F: & D_F \times D_F \times M \\\\
	\text{output feature map } G: & D_F \times D_F \times N \\\\
	N \text{ convolution kernel }: & D_K \times D_K \times M \times N \\\\
	\text{computation cost :} & D_K \cdot D_K \cdot M\cdot N \cdot D_F \cdot D_F
\end{cases}
$$

**Depthwise Separable Convolution:** use two layers separately for filtering and combining:

$$
\begin{cases}
	\text{input feature map } F: & D_F \times D_F \times M \\\\
	\text{Depthwise Convoluton Cost:} & D_K \cdot D_K \cdot M \cdot D_F \cdot D_F \\\\
	\text{Pointwise Convolution Cost:} & M \cdot N \cdot D_F \cdot D_F
\end{cases}
$$

By expressing convoluiton as a two step process of filtering and combining we get a reduction in computation of:

$$
\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M\cdot N \cdot D_F \cdot D_F} = \frac 1 N + \frac {1} {D^2_K}
$$

可以看到 output feature map 的 channel 越大、卷积核 filter size 越大，用 depthwise separable convoluiton 可以节省更多的计算资源。

# 3 MobileNetV1

All layers are followed by a batch norm and ReLU nonlinearity except the final fully connected layer.

![](http://oi3xms5do.bkt.clouddn.com/mobilev1_architect.png)

```python
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).

	# Arguments
		inputs: Input tensor
		filters: dimensionality of the output space (num of output filters in the convolution).
		alpha: controls the width of the network
		kernel: conv filter
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)
```

```python
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor
        pointwise_conv_filters: the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers.
        block_id: Integer, a unique identification designating
            the block number.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
```

```python
x = _conv_block(img_input, 32, alpha, strides=(2, 2))
x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                          strides=(2, 2), block_id=2)
x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                          strides=(2, 2), block_id=4)
x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                          strides=(2, 2), block_id=6)
x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                          strides=(2, 2), block_id=12)
x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

shape = (1, 1, int(1024 * alpha))

x = layers.GlobalAveragePooling2D()(x)
x = layers.Reshape(shape, name='reshape_1')(x)
x = layers.Dropout(dropout, name='dropout')(x)
x = layers.Conv2D(classes, (1, 1),
                  padding='same',
                  name='conv_preds')(x)
x = layers.Activation('softmax', name='act_softmax')(x)
x = layers.Reshape((classes,), name='reshape_2')(x)
```

# 4 MobileNetV2

[(Sandler et al. 2018)](https://arxiv.org/abs/1801.04381) proposed MobileNetV2 which is very similar to the original MobileNet, except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet.

## 4.1 Linear Bottlenecks

对一个 n 层的网络，第 i 层输出一个 h x w x d 的 activation tensor ，可以看作 d 维 h x w 的 pixels。

> Manifold of Interest

Informally, for an input set of real images, we say that the set of layer activations forms a "manifold of interst".

每一层的 layer activations 可以被映射到一个低维子空间：

It has been long assumed that manifolds of interst in neural networks could be embedded in low-dimensional subspaces.

In other words, when we look at all inividual d-channel pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which in turn is embeddable into a low dimensional subspace.

下面解释了些 ReLU + manifold of interst + low-dimension embedding，但没看懂：

If a result of a layer transformation ReLU(Bx) has a non-zero volume S, the points mapped to interior S are obtained via a linear transformation B of the input, thus indicating that **the part of the input space corresponding to the full dimensional output, is limited to a linear transformation.** 

![](http://oi3xms5do.bkt.clouddn.com/mobile_relu.png)

文章总结了 the manifold of interst should lie in a low-dimensional subspace of the higher-dimensional activation space 的亮点性质:

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of perserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

> An empirical hint

These two insights provides with an empirical hint for optimizing existing nueral architectures:

- Assuming the manifold of interst is low-dimensional we can capture this by inserting *linear bottleneck* layers into the convolutional blocks.

实验结果证明使用 linear layer 非常重要，因为非线性会损失很多信息：

Experimental evidence suggests that using linear layers is crucial as it prevents nonlinearities from destroying too much information.

下图 (a) 是传统的 3x3 卷积，(b) 是 depthwise conv + pointwise conv 的简单 separable conv, (c) 是加了 linear bottleneck 的 separable conv, (d) 是加了 expansion 的 separable conv。

![](http://oi3xms5do.bkt.clouddn.com/mobilenetV2.png)

## 4.2 Inverted residuals

> Residual Block

[(He et al. 2015)](https://arxiv.org/abs/1512.03385) proposed a *bottleneck* design for ResNet to reduce training time. 

The residual block has 3 layers: 1x1, 3x3, and 1x1 convolutions, where the 1x1 layers are responsible for **reducing and then increasing (restoring) dimensions**, leaving the 3x3 layer a bottleneck with smaller input/output dimensions.

![](http://oi3xms5do.bkt.clouddn.com/bottleneck_he.png)

下图展示了传统 Residual block 和 Inverted residual block 的不同，在 Residual block 中是先降维后升维，而在 Inverted residual block 中是先升维再降维。所以 Residual block 是两头大、中间小，而 Inverted residual block 是两头小、中间大。

However, inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks.

![](http://oi3xms5do.bkt.clouddn.com/mobile_Residule.png)

实验证明在 bottleneck 加入 linear bottleneck 比 nonlinear bottlenck 的效果好：

![](http://oi3xms5do.bkt.clouddn.com/residual_effect.png)

Operators for Bottleneck residual block:

![](http://oi3xms5do.bkt.clouddn.com/bottleneck_t1.png)

```python
def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    """
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x
```

## 4.3 Architecture

分成 stride = 1 和 2 两种 bottleneck block，只有在 stride = 1 时加 residual。

![](http://oi3xms5do.bkt.clouddn.com/mobilev2architect.png)

![](http://oi3xms5do.bkt.clouddn.com/mobilenet12.png)

## 4.4 Experiment

实验结果证明 MNet V2 在仅损失少量准确度的条件下，大幅减少了模型参数数量。

![](http://oi3xms5do.bkt.clouddn.com/mobile_result.png)

![](http://oi3xms5do.bkt.clouddn.com/mobile_result2.png)