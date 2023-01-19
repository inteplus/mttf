"""Useful subroutines dealing with GPU devices."""

import typing as tp

from ..utils import NameScope


__all__ = ["conv2d"]


def conv2d(name_scope: NameScope, x, filters, kernel_size, **kwargs):
    """Wrapper of Keras Conv2D layer with a LayerNormalization layer.

    Parameters
    ----------
    name_scope : mt.tf.NameScope
        the name scope. For every conv2d invocation, the name scope is iterated.
    x : tensor-like
        Keras tensor or TF tensor as input
    filters : int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    **kwargs : dict
        all other keyword arguments to be passed as-is to Conv2D layer construction

    Returns
    -------
    tensor-like
        TF tensor as output
    """

    import tensorflow.keras.layers as kl

    next(name_scope)
    x = kl.LayerNormalization(name=name_scope("prenorm"))(x)
    x = kl.Conv2D(filters, kernel_size, name=name_scope("conv"), **kwargs)(x)

    return x
