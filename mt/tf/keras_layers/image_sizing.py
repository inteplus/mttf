"""Module involves upsizing and downsizing images in each axis individually using convolutions of residuals."""

import typing as tp

import tensorflow as tf


__all__ = [
    "Downsize2D",
    "Upsize2D",
    "Downsize2D_V2",
    "Upsize2D_V2",
]


class Downsize2D(tf.keras.layers.Layer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height.

    Parameters
    ----------
    input_dim : int
        the dimensionality of each input pixel
    output_dim : int
        the dimensionality of each output pixel
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    kernel_initializer : object
        Initializer for the convolutional kernels.
    bias_initializer : object
        Initializer for the convolutional biases.
    kernel_regularizer : object
        Regularizer for the convolutional kernels.
    bias_regularizer : object
        Regularizer for the convolutional biases.
    kernel_constraint: object
        Contraint function applied to the convolutional layer kernels.
    bias_constraint: object
        Contraint function applied to the convolutional layer biases.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Downsize2D, self).__init__(**kwargs)

        if output_dim <= input_dim:
            raise ValueError(
                "Input dimensionality {} is not less than output dimensionality {}.".format(
                    input_dim, output_dim
                )
            )

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._residual_dim = output_dim - input_dim
        self._kernel_size = kernel_size
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._residual_dim * 4,
            self._kernel_size,
            padding="same",
            activation="swish",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="expand",
        )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        # self.norm_layer = tf.keras.layers.BatchNormalization(
        #    axis=3,
        #    epsilon=1e-3,
        #    momentum=0.999,
        #    name="batchnorm",
        # )
        self.projection_layer = tf.keras.layers.Conv2D(
            self._residual_dim,  # filters
            self._kernel_size,  # kernel_size
            padding="same",
            activation="sigmoid",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        input_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] // 2,
                2,
                input_shape[2] // 2,
                2,
                input_shape[3],
            ],
        )
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x_avg = tf.reduce_mean(x, axis=[3, 4], keepdims=True)
        x -= x_avg  # residuals
        x_avg = x_avg[:, :, :, 0, 0, :]  # means
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] // 2,
                input_shape[2] // 2,
                input_shape[3] * 4,
            ],
        )
        x = self.prenorm1_layer(x, training=training)
        x = self.expansion_layer(x, training=training)
        # x = self.norm_layer(x, training=training)
        # x = tf.nn.swish(x)
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)
        x = tf.concat([x_avg, x], axis=3)

        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[1] % 2 != 0:
            raise ValueError("The height must be even. Got {}.".format(input_shape[1]))

        if input_shape[2] % 2 != 0:
            raise ValueError("The width must be even. Got {}.".format(input_shape[2]))

        if input_shape[3] != self._input_dim:
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    self._input_dim, input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            self._output_dim,
        )

        return output_shape

    compute_output_shape.__doc__ = tf.keras.layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(Downsize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__


class Upsize2D(tf.keras.layers.Layer):
    """Upsizing along the x-axis and the y-axis using convolutions of residuals.

    Upsizing means doubling the width and the height.

    Parameters
    ----------
    input_dim : int
        the dimensionality of each input pixel
    output_dim : int
        the dimensionality of each output pixel
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    kernel_initializer : object
        Initializer for the convolutional kernels.
    bias_initializer : object
        Initializer for the convolutional biases.
    kernel_regularizer : object
        Regularizer for the convolutional kernels.
    bias_regularizer : object
        Regularizer for the convolutional biases.
    kernel_constraint: object
        Contraint function applied to the convolutional layer kernels.
    bias_constraint: object
        Contraint function applied to the convolutional layer biases.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Upsize2D, self).__init__(**kwargs)

        if output_dim >= input_dim:
            raise ValueError(
                "Input dimensionality {} is not greater than output dimensionality {}.".format(
                    input_dim, output_dim
                )
            )

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._residual_dim = input_dim - output_dim
        self._kernel_size = kernel_size
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._residual_dim * 4,
            self._kernel_size,
            padding="same",
            activation="swish",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="expand",
        )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        # self.norm_layer = tf.keras.layers.BatchNormalization(
        #    axis=3,
        #    epsilon=1e-3,
        #    momentum=0.999,
        #    name="batchnorm",
        # )
        self.projection_layer = tf.keras.layers.Conv2D(
            self._output_dim * 4,  # filters
            self._kernel_size,  # kernel_size
            padding="same",
            activation="tanh",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        x_avg = x[:, :, :, : self._output_dim]  # means
        x_avg = x_avg[:, :, :, tf.newaxis, tf.newaxis, :]
        x = x[:, :, :, self._output_dim :]  # residuals
        x = self.prenorm1_layer(x, training=training)
        x = self.expansion_layer(x, training=training)
        # x = self.norm_layer(x, training=training)
        # x = tf.nn.swish(x)
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)
        input_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                2,
                2,
                self._output_dim,
            ],
        )
        x += x_avg
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                self._output_dim,
            ],
        )

        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[3] != self._input_dim:
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    self._input_dim, input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            self._output_dim,
        )
        return output_shape

    compute_output_shape.__doc__ = tf.keras.layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(Upsize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__


class Downsize2D_V2(tf.keras.layers.Layer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    Parameters
    ----------
    input_dim : int
        the dimensionality (number of channels) of each input pixel
    expanded_dim : float, optional
        the dimensionality of the expanded space. If not given, it is set to 4 times the input
        dimensionality.
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    kernel_initializer : object
        Initializer for the convolutional kernels.
    bias_initializer : object
        Initializer for the convolutional biases.
    kernel_regularizer : object
        Regularizer for the convolutional kernels.
    bias_regularizer : object
        Regularizer for the convolutional biases.
    kernel_constraint: object
        Contraint function applied to the convolutional layer kernels.
    bias_constraint: object
        Contraint function applied to the convolutional layer biases.
    """

    def __init__(
        self,
        input_dim: int,
        expanded_dim: tp.Optional[int] = None,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Downsize2D_V2, self).__init__(**kwargs)

        self._input_dim = input_dim
        self._expanded_dim = input_dim * 4 if expanded_dim is None else expanded_dim
        self._kernel_size = kernel_size
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._expanded_dim,
            self._kernel_size,
            padding="same",
            activation="swish",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="expand",
        )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        self.projection_layer = tf.keras.layers.Conv2D(
            self._input_dim,
            self._kernel_size,
            padding="same",
            activation="tanh",  # output range (-1., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        input_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] // 2,
                2,
                input_shape[2] // 2,
                2,
                input_shape[3],
            ],
        )
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x_avg = tf.reduce_mean(x, axis=[3, 4], keepdims=True)
        x -= x_avg  # residuals
        x_avg = x_avg[:, :, :, 0, 0, :]  # means
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] // 2,
                input_shape[2] // 2,
                input_shape[3] * 4,
            ],
        )
        x = self.prenorm1_layer(x, training=training)
        x = self.expansion_layer(x, training=training)
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)
        x = tf.concat(
            [x_avg + x, x_avg - x], axis=3
        )  # to make the channels homogeneous, although new range (-1., 2.)

        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[1] % 2 != 0:
            raise ValueError("The height must be even. Got {}.".format(input_shape[1]))

        if input_shape[2] % 2 != 0:
            raise ValueError("The width must be even. Got {}.".format(input_shape[2]))

        if input_shape[3] != self._input_dim:
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    self._input_dim, input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            self._input_dim * 2,
        )

        return output_shape

    compute_output_shape.__doc__ = tf.keras.layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "expanded_dim": self._expanded_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(Downsize2D_V2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__


class Upsize2D_V2(tf.keras.layers.Layer):
    """Upsizing along the x-axis and the y-axis using convolutions of residuals.

    Upsizing means doubling the width and the height and halving the number of channels.

    Parameters
    ----------
    input_dim : int
        the dimensionality of each input pixel. Must be even.
    expanded_dim : float, optional
        the dimensionality of the expanded space. If not given, it is set to 2 times the input
        dimensionality.
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    kernel_initializer : object
        Initializer for the convolutional kernels.
    bias_initializer : object
        Initializer for the convolutional biases.
    kernel_regularizer : object
        Regularizer for the convolutional kernels.
    bias_regularizer : object
        Regularizer for the convolutional biases.
    kernel_constraint: object
        Contraint function applied to the convolutional layer kernels.
    bias_constraint: object
        Contraint function applied to the convolutional layer biases.
    """

    def __init__(
        self,
        input_dim: int,
        expanded_dim: tp.Optional[int] = None,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Upsize2D_V2, self).__init__(**kwargs)

        if input_dim & 1 != 0:
            raise ValueError(
                "Input dimensionality must be even. Got {}.".format(input_dim)
            )

        self._input_dim = input_dim
        self._expanded_dim = input_dim * 2 if expanded_dim is None else expanded_dim
        self._kernel_size = kernel_size
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._expanded_dim,
            self._kernel_size,
            padding="same",
            activation="swish",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="expand",
        )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        self.projection_layer = tf.keras.layers.Conv2D(
            self._input_dim * 2,
            self._kernel_size,
            padding="same",
            activation="tanh",  # output range (-1., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        x_plus = x[:, :, :, : self._input_dim // 2]
        x_minus = x[:, :, :, self._input_dim // 2 :]
        x_avg = (x_plus + x_minus) * 0.5  # means, expected range (0., 1.)
        x_avg = x_avg[:, :, :, tf.newaxis, tf.newaxis, :]
        x = (x_plus - x_minus) * 0.5  # residuals, expected range (-1., 1.)
        x = self.prenorm1_layer(x, training=training)
        x = self.expansion_layer(x, training=training)
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)
        input_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                2,
                2,
                self._input_dim // 2,
            ],
        )
        x += x_avg
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                self._input_dim // 2,
            ],
        )

        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[3] != self._input_dim:
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    self._input_dim, input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            self._input_dim // 2,
        )
        return output_shape

    compute_output_shape.__doc__ = tf.keras.layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "expanded_dim": self._expanded_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(Upsize2D_V2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__
