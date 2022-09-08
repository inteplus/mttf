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
"""A simplified version keras-based attention layer."""
# pylint: disable=g-classes-have-attributes

import collections
import math
import string

import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
    <query attention dims>, num_heads, channels)`

    Args:
      rank: Rank of query, key, value tensors.
      attn_axes: List/tuple of axes, `[-1, rank)`,
        that attention will be applied to.

    Returns:
      Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


@keras_export("keras.layers.SimpleMHA2D")
class SimpleMHA2D(Layer):
    """SimpleMHA2D layer.

    This is a simplified version of the Keras-based MultiHeadAttention layer.

      - There is no matrix projection at the output. The output_shape EinsumDense
        layer has been removed.
      - The query tensor is managed internally rather as input tensor. The user has
        to specify the shape of the query tensor at layer construction time.

    Update the docstring please!

    This is an implementation of multi-headed attention as described in the paper
    "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = SimpleMHA2D(query_shape=[8, 2, 2])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

    >>> layer = SimpleMHA2D(num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Args:
      query_shape: list or tuple representing the target shape (without batch size).
        It must contain at least 2 items where the last item is `key_dim` and the
        second last item is `num_heads`.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.

    Call arguments:
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be attention output if True, or (attention_output, attention_scores) if
        False. Defaults to False.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model,
        or False (inference) if there is no parent layer.

    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension.
      attention_scores: [Optional] multi-head attention coeffients over
        attention axes.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
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
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)

        self._query = self.add_weight(
            name="query",
            shape=[1, 1, num_heads, key_dim],
            initializer="random_normal",
            trainable=True,
        )

        import tensorflow as tf

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
        import tensorflow as tf

        bs_shape = outer_shape[0:1]
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
        value = self._value_proj(value, training=training)
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
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
        }
        base_config = super(SimpleMHA2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
