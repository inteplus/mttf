# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simplified version of keras-based attention layer."""
# pylint: disable=g-classes-have-attributes

import typing as tp
import numpy as np
import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.SimpleMHA2D")
class SimpleMHA2D(tf.keras.layers.Layer):
    """SimpleMHA2D layer.

    This is a simplified version of the Keras-based MultiHeadAttention layer.

    The layer takes as input a high-dim image tensor of shape [B, H, W, KV] where B is the
    batch size, H and W are the grid resolution, and KV is the (high) number of channels. It then
    2D-convolves the tensor into 2 tensors, `key` of shape [B, H, W, N*K] and `value` of shape
    [B, H, W, N*V] where N is the number of heads, K is the key dimensionality and V is the value
    dimensionality. In the absence of V, V is set to K. Next, it reshapes `key` as [B, H*W, N, K]
    and `value` as [B, H*W, N, V]. `key` is then dot-producted with an internal query tensor of
    shape [1, 1, N, K] with broadcasting, forming a tensor of shape [B, H*W, N]. This tensor is
    softmaxed along the axis containing H*W and reshaped as [B, H*W, N, 1], and then multiplied
    with `value` and sum-reduced along the axis containing H*W, forming an output attention tensor
    of shape [B, N, V].

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
    use_bias : bool
        Whether the convolutional layers use bias vectors/matrices.
    kernel_initializer : object
        Initializer for convolutional layer kernels.
    bias_initializer : object
        Initializer for convolutional layer biases.
    kernel_regularizer : object
        Regularizer for convolutional layer kernels.
    bias_regularizer : object
        Regularizer for convolutional layer biases.

    Examples
    --------

    >>> layer = SimpleMHA2D(num_heads=3, key_dim=40, value_dim=80)
    >>> input_tensor = tf.keras.Input(shape=[8, 8, 160])
    >>> output_tensor = layer(input_tensor)
    >>> print(output_tensor.shape)
    (None, 3, 80)
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: tp.Optional[int] = None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super(SimpleMHA2D, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self._query = self.add_weight(
            name="query",
            shape=[1, 1, num_heads, key_dim],
            initializer="random_normal",
            trainable=True,
        )

        self._key_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._key_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )

        self._value_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._value_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
        )

        self._softmax = tf.keras.layers.Softmax(axis=1)

    def call(self, key_value, training=None):
        """The call function.

        Parameters
        ----------
        key_value : tensorflow.Tensor
            input `Tensor` of shape `(B, H, W, KV)`.
        training : bool
            Whether the layer should behave in training mode or in inference mode.

        Returns
        -------
        attention_output : tensorflow.Tensor
            The result of the computation, of shape `(B, N, V)`, where `N` is the number of heads
            and `V` is the value dimensionality.
        """

        bs_shape = tf.shape(key_value)[0:1]
        hw_shape = tf.reduce_prod(tf.shape(key_value)[1:3], axis=0, keepdims=True)

        #   N = `num_attention_heads`
        #   K = `key_dim`
        #   V = `value_dim`
        #   H = `image_height`
        #   W = `image_width`
        # `query` = [1, 1, N ,K]

        # `key` = [B, H*W, N, K]
        key = self._key_proj(key_value, training=training)
        key_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._key_dim]], axis=0
        )
        key = tf.reshape(key, key_shape)

        # `value` = [B, H*W, N, V]
        value = self._value_proj(key_value, training=training)
        value_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._value_dim]], axis=0
        )
        value = tf.reshape(value, value_shape)

        # `dot_prod` = [B, H*W, N]
        dot_prod = tf.reduce_sum(self._query * key, axis=-1)

        # `softmax` = [B, H*W, N, 1]
        softmax = self._softmax(dot_prod)
        softmax = tf.expand_dims(softmax, axis=-1)

        # `attention_output` = [B, N, V]
        attention_output = tf.reduce_sum(softmax * value, axis=1)

        return attention_output

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "use_bias": self._use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
        }
        base_config = super(SimpleMHA2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
