"""Module involves upsizing and downsizing images in each axis individually using convolutions of residuals."""

import typing as tp

import numpy as np
import tensorflow as tf


__all__ = [
    "DUCLayer",
    "Downsize2D",
    "Upsize2D",
    "Downsize2D_V2",
    "Upsize2D_V2",
]


def mirror_all_weights(l_weights: list) -> list:
    """TBC"""

    l_newWeights = []
    for arr in l_weights:
        if arr.ndim == 1:
            new_arr = np.tile(arr, 2)
        elif arr.ndim == 4:
            zero_arr = np.zeros_like(arr, dtype=arr.dtype)
            x = np.stack([arr, zero_arr, zero_arr, arr], axis=-1)
            x = x.reshape(arr.shape + (2, 2))
            x = np.transpose(x, [0, 1, 4, 2, 5, 3])
            new_arr = x.reshape(
                (arr.shape[0], arr.shape[1], arr.shape[2] << 1, arr.shape[3] << 1)
            )
        else:
            raise NotImplementedError("SOTA exceeded!")
        l_newWeights.append(new_arr)

    return l_newWeights


class DUCLayer(tf.keras.layers.Layer):
    """TBC

    Parameters
    ----------
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
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):

        super(DUCLayer, self).__init__(**kwargs)

        self._kernel_size = kernel_size
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    def get_config(self):
        config = {
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
        base_config = super(DUCLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__

    def get_mirrored_weights(self):
        return mirror_all_weights(self.get_weights())


class Upsize2D(DUCLayer):
    """Upsizing along the x-axis and the y-axis using convolutions of residuals.

    Upsizing means doubling the width and the height and halving the number of channels.

    Input at each grid cell is a pair of `(avg, res)` images at resolution `(H,W,C)`. The pair is
    transformed to `4*expansion_factor` hidden images and then 4 residual images
    `(res1, res2, res3, res4)`. Then, `avg` is added to the 4 residual images, forming at each cell
    a 2x2 block of images `(avg+res1, avg+res2, avg+res3, avg+res4)`. Finally, the new blocks
    across the whole tensor form a new grid, doubling the height and width. Note that each
    `avg+resK` image serves as a pair of average and residual images in the higher resolution.

    Parameters
    ----------
    input_dim : int
        the dimensionality of each input pixel. Must be even.
    expansion_factor : int
        the coefficient defining the number of hidden images per cell needed.
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
        expansion_factor: int = 2,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Upsize2D, self).__init__(
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        if input_dim & 1 != 0:
            raise ValueError(
                "Input dimensionality must be even. Got {}.".format(input_dim)
            )

        self._input_dim = input_dim
        self._expansion_factor = expansion_factor

        if self._expansion_factor > 1:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expansion_layer = tf.keras.layers.Conv2D(
                self._input_dim * 2 * expansion_factor,
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
            activation="tanh",  # (-1., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        x_avg = x[:, :, :, : self._input_dim // 2]

        if self._expansion_factor > 1:  # expand
            x = self.prenorm1_layer(x, training=training)
            x = self.expansion_layer(x, training=training)

        # project
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)

        # reshape
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

        # add average
        x += x_avg[:, :, :, tf.newaxis, tf.newaxis, :]

        # make a new grid
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

    call.__doc__ = DUCLayer.call.__doc__

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

    compute_output_shape.__doc__ = DUCLayer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "expansion_factor": self._expansion_factor,
        }
        base_config = super(Upsize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__


class Downsize2D(DUCLayer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Parameters
    ----------
    input_dim : int
        the dimensionality (number of channels) of each input pixel
    expansion_factor : int
        the coefficient defining the number of hidden images per cell needed.
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
        expansion_factor: int = 2,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Downsize2D, self).__init__(
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self._input_dim = input_dim
        self._expansion_factor = expansion_factor

        if self._expansion_factor > 1:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expansion_layer = tf.keras.layers.Conv2D(
                self._input_dim * 4 * self._expansion_factor,
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
            activation="sigmoid",  # (0., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        # reshape
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

        # extract average
        x_avg = tf.reduce_mean(x, axis=[2, 4], keepdims=True)
        x -= x_avg  # residuals
        x_avg = x_avg[:, :, 0, :, 0, :]  # means

        # make a new grid
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

        x = tf.concat([x_avg, x], axis=3)

        if self._expansion_factor > 1:  # expand
            x = self.prenorm1_layer(x, training=training)
            x = self.expansion_layer(x, training=training)

        # project
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)

        # form output
        x = tf.concat([x_avg, x], axis=3)

        return x

    call.__doc__ = DUCLayer.call.__doc__

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

    compute_output_shape.__doc__ = DUCLayer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "input_dim": self._input_dim,
            "expansion_factor": self._expansion_factor,
        }
        base_config = super(Downsize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__


# ----- new -----


class Downsize2D_V2(DUCLayer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
    expansion_factor : int
        the coefficient defining the number of hidden images per cell needed.
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
    projection_uses_bias : bool
        whether or not the projection convolution layer uses a bias vector
    """

    def __init__(
        self,
        img_dim: int,
        res_dim: int,
        expansion_factor: int = 2,
        kernel_size: tp.Union[int, tuple, list] = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        projection_uses_bias: bool = True,
        **kwargs
    ):
        super(Downsize2D_V2, self).__init__(
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self._img_dim = img_dim
        self._res_dim = res_dim
        self._expansion_factor = expansion_factor
        self._projection_uses_bias = projection_uses_bias

        if self._expansion_factor > 1:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expansion_layer = tf.keras.layers.Conv2D(
                (self._img_dim + self._res_dim) * 4 * self._expansion_factor,
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
            self._img_dim + self._res_dim * 2,
            1,
            padding="same",
            activation="sigmoid",  # (0., 1.)
            use_bias=self._projection_uses_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        # reshape
        I = self._img_dim
        R = self._res_dim
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1] // 2
        W = input_shape[2] // 2
        x = tf.reshape(x, [B, H, 2, W, 2, I + R])

        # extract average over the image dimensions
        x_avg = tf.reduce_mean(x[:, :, :, :, :, :I], axis=[2, 4], keepdims=True)
        zeros = tf.zeros([B, H, 1, W, 1, R])
        x -= tf.concat([x_avg, zeros], axis=5)  # residuals
        x_avg = x_avg[:, :, 0, :, 0, :]  # means

        # make a new grid
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H, W, (I + R) * 4])

        x = tf.concat([x_avg, x], axis=3)

        if self._expansion_factor > 1:  # expand
            x = self.prenorm1_layer(x, training=training)
            x = self.expansion_layer(x, training=training)

        # project
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)

        # form output
        x = tf.concat([x_avg, x], axis=3)

        return x

    call.__doc__ = DUCLayer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[1] % 2 != 0:
            raise ValueError("The height must be even. Got {}.".format(input_shape[1]))

        if input_shape[2] % 2 != 0:
            raise ValueError("The width must be even. Got {}.".format(input_shape[2]))

        if input_shape[3] != self._img_dim + self._res_dim:
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    self._img_dim + self._res_dim, input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            (self._img_dim + self._res_dim) * 2,
        )

        return output_shape

    compute_output_shape.__doc__ = DUCLayer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "img_dim": self._img_dim,
            "res_dim": self._res_dim,
            "expansion_factor": self._expansion_factor,
            "projection_uses_bias": self._projection_uses_bias,
        }
        base_config = super(Downsize2D_V2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__


class Upsize2D_V2(DUCLayer):
    """Upsizing along the x-axis and the y-axis using convolutions of residuals.

    Upsizing means doubling the width and the height and halving the number of channels.

    Input at each grid cell is a pair of `(avg, res)` images at resolution `(H,W,C)`. The pair is
    transformed to `4*expansion_factor` hidden images and then 4 residual images
    `(res1, res2, res3, res4)`. Then, `avg` is added to the 4 residual images, forming at each cell
    a 2x2 block of images `(avg+res1, avg+res2, avg+res3, avg+res4)`. Finally, the new blocks
    across the whole tensor form a new grid, doubling the height and width. Note that each
    `avg+resK` image serves as a pair of average and residual images in the higher resolution.

    Input dimensionality consists of image dimensionality and residual dimensionality. It must be
    even.

    Parameters
    ----------
    img_dim : int
        the image dimensionality.
    res_dim : int
        the residual dimensionality.
    expansion_factor : int
        the coefficient defining the number of hidden images per cell needed.
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
        img_dim: int,
        res_dim: int,
        expansion_factor: int = 2,
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Upsize2D_V2, self).__init__(
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        input_dim = img_dim + res_dim
        if input_dim & 1 != 0:
            raise ValueError(
                "Image dimensionality must be even. Got {}.".format(input_dim)
            )

        self._img_dim = img_dim
        self._res_dim = res_dim
        self._expansion_factor = expansion_factor

        if self._expansion_factor > 1:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expansion_layer = tf.keras.layers.Conv2D(
                (self._img_dim + self._res_dim) * 2 * expansion_factor,
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
            (self._img_dim + self._res_dim) * 2,
            self._kernel_size if self._expansion_factor <= 1 else 1,
            padding="same",
            activation="tanh",  # (-1., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project",
        )

    def call(self, x, training: bool = False):
        I = self._img_dim
        R = (self._res_dim - self._img_dim) // 2
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]

        x_avg = x[:, :, :, :I]

        if self._expansion_factor > 1:  # expand
            x = self.prenorm1_layer(x, training=training)
            x = self.expansion_layer(x, training=training)

        # project
        x = self.prenorm2_layer(x, training=training)
        x = self.projection_layer(x, training=training)

        # reshape
        x = tf.reshape(x, [B, H, W, 2, 2, I + R])

        # add average
        zeros = tf.zeros([B, H, W, R])
        x_avg = tf.concat([x_avg, zeros], axis=3)  # expanded average
        x += x_avg[:, :, :, tf.newaxis, tf.newaxis, :]

        # make a new grid
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H * 2, W * 2, I + R])

        return x

    call.__doc__ = DUCLayer.call.__doc__

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "Expected input shape to be (B, H, W, C). Got: {}.".format(input_shape)
            )

        if input_shape[3] != (self._img_dim + self._res_dim):
            raise ValueError(
                "The input dim must be {}. Got {}.".format(
                    (self._img_dim + self._res_dim), input_shape[3]
                )
            )

        output_shape = (
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            (self._img_dim + self._res_dim) // 2,
        )
        return output_shape

    compute_output_shape.__doc__ = DUCLayer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "img_dim": self._img_dim,
            "res_dim": self._res_dim,
            "expansion_factor": self._expansion_factor,
        }
        base_config = super(Upsize2D_V2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__
