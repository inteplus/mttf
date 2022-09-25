"""Module involves upsizing and downsizing images in each axis individually via convolution."""

import typing as tp

import tensorflow as tf


__all__ = ["Upsize2D", "Downsize2D"]


class Upsize2D(tf.keras.layers.Layer):
    """Upsizing along the x-axis and the y-axis via convolution.

    Upsizing means doubling the width and the height.

    Parameters
    ----------
    output_dim : int
        the dimensionality of each output pixel
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    activation : object
        activation for the convolution
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
        output_dim: int,
        kernel_size: tp.Union[int, tuple, list] = 1,
        activation="tanh",  # we expect the input pixel values to be lower and upper-bounded
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Upsize2D, self).__init__(**kwargs)

        self._output_dim = output_dim
        self._kernel_size = kernel_size
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._output_dim * 8,
            kernel_size=1,
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
            self._output_dim * 4,  # filters
            self._kernel_size,  # kernel_size
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x):
        x = self.prenorm1_layer(x)
        x = self.expansion_layer(x)
        x = self.prenorm2_layer(x)
        x = self.projection_layer(x)
        input_shape = tf.shape(x)
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                2,
                2,
                input_shape[3] // 4,
            ],
        )
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                input_shape[3] // 4,
            ],
        )

        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
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
            "output_dim": self._output_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(SimpleMHA2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__


class Downsize2D(tf.keras.layers.Layer):
    """Downsizing along the x-axis and the y-axis via convolution.

    Downsizing means halving the width and the height.

    Parameters
    ----------
    output_dim : int
        the dimensionality of each output pixel
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    activation : object
        activation for the convolution
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
        output_dim: int,
        kernel_size: tp.Union[int, tuple, list] = 1,
        activation="tanh",  # we expect the input pixel values to be lower and upper-bounded
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Downsize2D, self).__init__(**kwargs)

        self._output_dim = output_dim
        self._kernel_size = kernel_size
        self._activation = tf.keras.activations.get(activation)
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expansion_layer = tf.keras.layers.Conv2D(
            self._output_dim * 2,
            kernel_size=1,
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
            self._output_dim,  # filters
            self._kernel_size,  # kernel_size
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x):
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
        x = tf.reshape(
            x,
            [
                input_shape[0],
                input_shape[1] // 2,
                input_shape[2] // 2,
                input_shape[3] * 4,
            ],
        )
        x = self.prenorm1_layer(x)
        x = self.expansion_layer(x)
        x = self.prenorm2_layer(x)
        x = self.projection_layer(x)

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
            "output_dim": self._output_dim,
            "kernel_size": self._kernel_size,
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
        }
        base_config = super(SimpleMHA2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__
