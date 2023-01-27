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

import math
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from mt import tp, tfc


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
    activation : object
        activation for the `value` convolution
    kernel_initializer : object
        Initializer for convolutional layer kernels.
    bias_initializer : object
        Initializer for convolutional layer biases.
    kernel_regularizer : object
        Regularizer for convolutional layer kernels.
    bias_regularizer : object
        Regularizer for convolutional layer biases.
    kernel_constraint: object
        Contraint function applied to the layer kernels.
    bias_constraint: object
        Contraint function applied to the layer biases.
    dropout: float
        dropout probability

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
        activation="tanh",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dropout: float = 0.2,
        **kwargs
    ):
        super(SimpleMHA2D, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._use_bias = use_bias
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._dropout = dropout

        self.tensor_query = self.add_weight(
            name="query",
            shape=[1, 1, num_heads, key_dim],
            initializer="random_normal",
            trainable=True,
        )

        self.layer_key_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._key_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        self.layer_value_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._value_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        self.layer_softmax = tf.keras.layers.Softmax(axis=1)
        if self._dropout > 0:
            self.layer_dropout = tf.keras.layers.Dropout(rate=self._dropout)

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
        key = self.layer_key_proj(key_value, training=training)
        key_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._key_dim]], axis=0
        )
        key = tf.reshape(key, key_shape)

        # `value` = [B, H*W, N, V]
        value = self.layer_value_proj(key_value, training=training)
        value_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._value_dim]], axis=0
        )
        value = tf.reshape(value, value_shape)

        # `dot_prod` = [B, H*W, N]
        dot_prod = tf.reduce_sum(self.tensor_query * key, axis=-1)

        # `softmax` = [B, H*W, N, 1]
        softmax = self.layer_softmax(dot_prod)
        if self._dropout > 0:
            softmax = self.layer_dropout(softmax, training=training)
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
            "activation": tf.keras.activations.serialize(self._activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(
                self._kernel_constraint
            ),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
            "dropout": self._dropout,
        }
        base_config = super(SimpleMHA2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.layers.MHAPool2D")
class MHAPool2D(tf.keras.layers.Layer):
    """Pooling in 2D using Keras-based self-attention.

    The layer takes as input a high-dim image tensor of shape [B, H, W, D] where B is the
    batch size, H and W are the grid resolution, and D is the (high) number of channels. First, it
    pools the tensor down to an unprojected query tensor of shape [B, H2, W2, D] using max or avg
    pooling. Second, it 2D-convolves the unprojected query tensor, the input tensor and the input
    tensor to the `query` tensor of shape [B, H2, W2, N*K], the `key` tensor of shape
    [B, H, W, N*K] and the `value` tensor of shape [B, H, W, N*V] where N is the number of heads,
    K is the key dimensionality and V is the value dimensionality. In the absence of V, V is set to
    K. Third, it divides `query` with sqrt(K). Fourth, it splits the `num_heads` dimension out of
    the last dimension from all 3 tensors. Fifth, in a single einsum op of `query` and `key`, it
    contracts K, makes an outer-product of [H2,W2] with [H,W], and runs through all B and N,
    outputing a `prod` tensor of shape [B, H2, W2, H, W, N]. Fifth, it merges H with W in both
    `prod` and `value` resulting in shapes [B, H2, W2, H*W, N] and [B, H*W, N, V] respectively.
    Sixth, `prod` is softmaxed along the H*W axis. Seventh, in another einsum op of `prod` and
    `value`, it contracts H*W while running through all other indices, outputing an `attention`
    tensor of shape [B, H2, W2, N, V]. Finally, it merges N with V and returns the result.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
    pooling : {'max', 'avg'}
        type of 2D pooling
    pool_size : int or tuple
        integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal).
        (2, 2) will halve the input in both spatial dimension. If only one integer is specified,
        the same window length will be used for both dimensions.
    use_bias : bool
        Whether the convolution layers use bias vectors/matrices.
    activation : object
        activation for the `value` convolution
    kernel_initializer : object
        Initializer for the convolutional layer kernels.
    bias_initializer : object
        Initializer for the convolutional layer biases.
    kernel_regularizer : object
        Regularizer for the convolutional layer kernels.
    bias_regularizer : object
        Regularizer for the convolutional layer biases.
    kernel_constraint: object
        Contraint function applied to the layer kernels.
    bias_constraint: object
        Contraint function applied to the layer biases.
    dropout: float
        dropout probability

    Examples
    --------

    >>> layer = MHAPool2D(num_heads=3, key_dim=40, value_dim=80)
    >>> input_tensor = tf.keras.Input(shape=[8, 8, 160])
    >>> output_tensor = layer(input_tensor)
    >>> print(output_tensor.shape)
    (None, 4, 4, 240)
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: tp.Optional[int] = None,
        pooling: str = "max",
        pool_size=(2, 2),
        use_bias: bool = True,
        activation="swish",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        dropout: float = 0.2,
        **kwargs
    ):
        super(MHAPool2D, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._pooling = pooling
        self._pool_size = pool_size
        self._use_bias = use_bias
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._dropout = dropout

        if self._pooling == "max":
            self.layer_pool = tf.keras.layers.MaxPool2D()
        elif self._pooling == "avg":
            self.layer_pool = tf.keras.layers.AveragePooling2D()
        else:
            raise tfc.ModelSyntaxError(
                "Invalid pooling string: '{}'.".format(self._pooling)
            )

        self.layer_query_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._key_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="query_proj",
        )

        self.layer_key_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._key_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            activation=None,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="key_proj",
        )

        self.layer_value_proj = tf.keras.layers.Conv2D(
            self._num_heads * self._value_dim,  # filters
            1,  # kernel_size
            use_bias=self._use_bias,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="value_proj",
        )

        self.layer_softmax = tf.keras.layers.Softmax(axis=3)
        if self._dropout > 0:
            self.layer_dropout = tf.keras.layers.Dropout(rate=self._dropout)

    def call(self, blob, training=None, return_attention_scores: bool = False):
        """The call function.

        Parameters
        ----------
        blob : tensorflow.Tensor
            input `Tensor` of shape `(B, H, W, D)`.
        training : bool
            Whether the layer should behave in training mode or in inference mode.
        return_attention_scores : bool
            Whether to return the attention scores as well.

        Returns
        -------
        attention_output : tensorflow.Tensor
            The result of the computation, of shape `(B, H2, W2, N*V)`, where `H2` and `W2`
            represent the downsampled resolution, `N` is the number of heads and `V` is the value
            dimensionality.
        attention_scores : tensorflow.Tensor
            Multi-headed attention weights, of shape `(B, H2, W2, H*W, N)`. Only available if
            `return_attention_scores` is True.
        """

        blob_shape = tf.shape(blob)
        bs_shape = blob_shape[0:1]  # [B]
        hw_shape = tf.reduce_prod(blob_shape[1:3], axis=0, keepdims=True)  # [H*W]

        #   N = `num_attention_heads`
        #   K = `key_dim`
        #   V = `value_dim`
        #   H = `image_height`
        #   W = `image_width`

        # `query` = [B, H2, W2, D]
        query = self.layer_pool(blob)

        # `query` = [B, H2, W2, N, K]
        query = self.layer_query_proj(query, training=training)
        query_head_shape = tf.shape(query)[0:3]  # [B, H2, W2]
        query_shape = tf.concat(
            [query_head_shape, [self._num_heads, self._key_dim]], axis=0
        )
        query = tf.reshape(query, query_shape)

        # `key` = [B, H*W, N, K]
        key = self.layer_key_proj(blob, training=training)
        key_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._key_dim]], axis=0
        )
        key = tf.reshape(key, key_shape)

        # `value` = [B, H*W, N, V]
        value = self.layer_value_proj(blob, training=training)
        value_shape = tf.concat(
            [bs_shape, hw_shape, [self._num_heads, self._value_dim]], axis=0
        )
        value = tf.reshape(value, value_shape)

        # `prod` = [B, H2, W2, H*W, N]
        query *= 1.0 / math.sqrt(float(self._key_dim))
        prod = tf.einsum("bhwnk,bink->bhwin", query, key)

        # `attention_scores` = [B, H2, W2, H*W, N]
        attention_scores = self.layer_softmax(prod)
        if self._dropout > 0:
            dropout = self.layer_dropout(attention_scores, training=training)
        else:
            dropout = attention_scores

        # `attention_output` = [B, H2, W2, N, V]
        attention_output = tf.einsum("bhwin,binv->bhwnv", dropout, value)

        # `output`
        output_shape = tf.concat(
            [query_head_shape, [self._num_heads * self._value_dim]], axis=0
        )
        output = tf.reshape(attention_output, output_shape)

        if return_attention_scores:
            return output, attention_scores
        return output

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "pooling": self._pooling,
            "pool_size": self._pool_size,
            "use_bias": self._use_bias,
            "activation": tf.keras.activations.serialize(self._activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(
                self._kernel_constraint
            ),
            "bias_constraint": tf.keras.constraints.serialize(self._bias_constraint),
            "dropout": self._dropout,
        }
        base_config = super(MHAPool2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
