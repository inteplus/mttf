import tensorflow.keras as tk


from .counter import Counter


class NormedConv2D(tk.layers.Layer):
    """A block of Conv2D without activation, followed by LayerNormalization, then activation.

    This layer represents the following block:

    .. code-block:: python

       x = input_tensor
       count = tf.keras_layers.Counter()(x)
       x = tf.keras.layers.Conv2D(activation=None, use_bias=False, ...)(x)
       y = tf.keras.layers.LayerNormalization(axis=-1, scale=True, ...)(x)
       alpha = growth_rate / (1.0 + count)
       x = alpha * x + (1 - alpha) * y
       z = tf.keras.layers.Activation(...)(y)
       return z

    It operates as a Conv2D layer whose kernel responses are normalized before being activated. The
    bias vector of the convolution is omitted and replaced by the beta vector of the normalization.
    The activation of the convolution is done explicitly via an activation layer.

    Parameters
    ----------
    filters : int
        Integer, the dimensionality of the output space (i.e. the number of output filters in the
        convolution). Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    strides : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the strides of the convolution along the
        height and width. Can be a single integer to specify the same value for all spatial
        dimensions. Specifying any ``stride value != 1 is`` incompatible with specifying any
        ``dilation_rate value != 1``. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    padding : {"valid", "same"}
        'valid' means no padding. 'same' results in padding with zeros evenly to the left/right
        or up/down of the input. When ``padding="same"`` and ``strides=1``, the output has the same
        size as the input. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    data_format : {"channel_last", "channel_first", None}
        A string, one of 'channels_last' (default) or 'channels_first'. The ordering of the
        dimensions in the inputs. channels_last corresponds to inputs with shape
        ``(batch_size, height, width, channels)`` while channels_first corresponds to inputs with
        shape ``(batch_size, channels, height, width)``. It defaults to the image_data_format value
        found in your Keras config file at ``~/.keras/keras.json``. If you never set it, then it
        will be 'channels_last'. Note that the 'channels_first' format is currently not
        supported by TensorFlow on CPU. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    dilation_rate : int or tuple or list
        an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated
        convolution. Can be a single integer to specify the same value for all spatial dimensions.
        Currently, specifying any ``dilation_rate value != 1`` is incompatible with specifying any
        ``stride value != 1``. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    groups : int
        A positive integer specifying the number of groups in which the input is split along the
        channel axis. Each group is convolved separately with filters / groups filters. The output
        is the concatenation of all the groups results along the channel axis. Input channels and
        filters must both be divisible by groups. Passed as-is to
        :class:`tensorflow.keras.layers.Conv2D`.
    kernel_initializer : str or object
        Initializer for the kernel weights matrix (see keras.initializers). Defaults to
        'glorot_uniform'. Passed as-is to :class:`tensorflow.keras.layers.Conv2D`.
    kernel_regularizer : str or object
        Regularizer function applied to the kernel weights matrix (see keras.regularizers).  Passed
        as-is to :class:`tensorflow.keras.layers.Conv2D`.
    kernel_constraint : str or object
        Constraint function applied to the kernel matrix (see keras.constraints). Passed as-is to
        :class:`tensorflow.keras.layers.Conv2D`.
    epsilon : float
        Small float added to variance to avoid dividing by zero. Defaults to ``1e-3``. Passed as-is
        to :class:`tensorflow.keras.layers.LayerNormalization`.
    center : float
        If True, add offset of beta to normalized tensor. If False, beta is ignored. Defaults to
        True. Passed as-is to :class:`tensorflow.keras.layers.LayerNormalization`.
    gamma_initializer : str or object
        Initializer for the gamma weight. Defaults to ones. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    gamma_regularizer : str or object
        Optional regularizer for the gamma weight. None by default. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    gamma_constraint : str or object
        Optional constraint for the gamma weight. None by default. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    beta_initializer : str or object
        Initializer for the beta weight. Defaults to zeros. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    beta_regularizer : str or object
        Optional regularizer for the beta weight. None by default. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    beta_constraint: str or object
        Optional constraint for the beta weight. None by default. Passed as-is to
        :class:`tensorflow.keras.layers.LayerNormalization`.
    growth_rate : float
        Growth rate for switching from Conv2D output to the normed output. Defaults to 1.0.
    activation : str or object
        Activation function to use. If you don't specify anything, no activation is applied (see
        keras.activations).  Passed as-is to :class:`tensorflow.keras.layers.Activation`.

    Input shape
    -----------
    ``4+D`` tensor with shape: ``batch_shape + (channels, rows, cols)`` if
    ``data_format='channels_first'`` or ``4+D`` tensor with shape:
    ``batch_shape + (rows, cols, channels)`` if ``data_format='channels_last'``.

    Output shape
    ------------
    ``4+D`` tensor with shape: ``batch_shape + (filters, new_rows, new_cols)`` if
    ``data_format='channels_first'`` or ``4+D`` tensor with shape:
    ``batch_shape + (new_rows, new_cols, filters)`` if ``data_format='channels_last'``. rows and
    cols values might have changed due to padding.

    Please see the `layer_normalization`_ paper for more details.

    .. _layer_normalization:
       https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        epsilon=0.001,
        center=True,
        gamma_initializer="ones",
        gamma_regularizer=None,
        gamma_constraint=None,
        beta_initializer="zeros",
        beta_regularizer=None,
        beta_constraint=None,
        growth_rate: float = 1.0,
        activation=None,
        **kwargs,
    ):
        super(NormedConv2D, self).__init__(**kwargs)

        self.keys = [
            "filters",
            "kernel_size",
            "strides",
            "padding",
            "data_format",
            "dilation_rate",
            "groups",
            "kernel_initializer",
            "kernel_regularizer",
            "kernel_constraint",
            "epsilon",
            "center",
            "gamma_initializer",
            "gamma_regularizer",
            "gamma_constraint",
            "beta_initializer",
            "beta_regularizer",
            "beta_constraint",
            "growth_rate",
            "activation",
        ]

        for key in self.keys:
            setattr(self, key, locals()[key])

        self.conv2d = tk.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
        )

        self.counter = Counter()

        self.norm = tk.layers.LayerNormalization(
            axis=-1,
            epsilon=epsilon,
            scale=True,
            center=center,
            gamma_initializer=gamma_initializer,
            gamma_regularizer=gamma_regularizer,
            gamma_constraint=gamma_constraint,
            beta_initializer=beta_initializer,
            beta_regularizer=beta_regularizer,
            beta_constraint=beta_constraint,
        )

        if activation is not None:
            self.acti = tk.layers.Activation(activation)

    def call(self, x, training: bool = False):
        count = self.counter(x, training=training)
        coeff = self.growth_rate / (1.0 + count)
        y1 = self.conv2d(x, training=training)
        y2 = self.norm(y1)
        z = coeff * y1 + (1 - coeff) * y2
        if self.activation is None:
            return z
        w = self.acti(z, training=training)
        return w

    call.__doc__ = tk.layers.Layer.call.__doc__

    def get_config(self):
        config = {key: getattr(self, key) for key in self.keys}
        prefixes = ["kernel", "gamma", "beta"]
        for prefix in prefixes:
            key = prefix + "_initializer"
            config[key] = tk.initializers.serialize(config[key])
            key = prefix + "_regularizer"
            config[key] = tk.regularizers.serialize(config[key])
            key = prefix + "_constraint"
            config[key] = tk.constraints.serialize(config[key])
        base_config = super(NormedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
