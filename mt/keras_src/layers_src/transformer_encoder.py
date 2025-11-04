from .. import layers, activations, initializers


class MTTransformerEncoder(layers.Layer):
    """Transformer encoder.

    This layer was ripped from [KerasHub](https://github.com/keras-team/keras-hub/blob/master/keras_hub/src/layers/modeling/transformer_encoder.py)
    by MT. It has been adapted and simplified to work with Keras 2. It does not
    take as input attention mask or padding mask, but instead computes them
    internally.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `keras.layers.MultiHeadAttention` layer.
        dropout: float. the dropout value, shared by
            `keras.layers.MultiHeadAttention` and feedforward network.
            Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer and the intermediate dense layer  are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_hub.layers.MTTransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=(10, 64))
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(2, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True

    def build(self, inputs_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=inputs_shape,
                value=inputs_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=inputs_shape,
                value_shape=inputs_shape,
            )
        self._self_attention_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(inputs_shape)
        self._self_attention_dropout = layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = layers.Dense(
            hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(
        self,
        inputs,
        training=None,
        return_attention_scores=False,
    ):
        """Forward pass of the MTTransformerEncoder.

        Args:
            inputs: a Tensor. The input data to MTTransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.
            return_attention_scores: a boolean indicating whether the output
                should be `(attention_output, attention_scores)` if `True` or
                `attention_output` if `False`. Defaults to `False`.

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs  # Intermediate result.

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)

        if return_attention_scores:
            x, attention_scores = self._self_attention_layer(
                query=x,
                value=x,
                return_attention_scores=return_attention_scores,
                training=training,
            )
        else:
            x = self._self_attention_layer(
                query=x,
                value=x,
                training=training,
            )

        x = self._self_attention_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        if return_attention_scores:
            return x, attention_scores

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape