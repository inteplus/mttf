"""Module involves upsizing and downsizing images in each axis individually using convolutions of residuals."""

import tensorflow as tf

from mt import tp, np


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
    """Base layer for all DUC layer implementations.

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


# ----- v2 -----


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


# ----- v3 -----


class Downsize2D_V3(DUCLayer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Downsize2D_V3, self).__init__(
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

        if res_dim > 0:
            if res_dim > img_dim:
                self.prenorm1_layer = tf.keras.layers.LayerNormalization(
                    name="prenorm1"
                )
                self.expand1_layer = tf.keras.layers.Conv2D(
                    img_dim * 2 + res_dim * 4,
                    self._kernel_size,
                    padding="same",
                    activation="swish",
                    kernel_initializer=self._kernel_initializer,
                    bias_initializer=self._bias_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    kernel_constraint=self._kernel_constraint,
                    bias_constraint=self._bias_constraint,
                    name="expand1",
                )
            RR = (img_dim + res_dim * 3 + 1) // 2
            self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
            self.project1_layer = tf.keras.layers.Conv2D(
                RR,
                1,
                padding="same",
                activation="swish",
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="project1",
            )
            self.prenorm3_layer = tf.keras.layers.LayerNormalization(name="prenorm3")
            self.expand2_layer = tf.keras.layers.Conv2D(
                img_dim * 2 + RR * 4,
                self._kernel_size,
                padding="same",
                activation="swish",
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="expand2",
            )
        self.prenorm4_layer = tf.keras.layers.LayerNormalization(name="prenorm4")
        self.project2_layer = tf.keras.layers.Conv2D(
            img_dim + res_dim * 2,
            1,
            padding="same",
            activation="sigmoid",  # (0., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project2",
        )

    def call(self, x, training: bool = False):
        # shape
        I = self._img_dim
        R = self._res_dim
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1] // 2
        W = input_shape[2] // 2

        # merge pairs of consecutive pixels in each row
        # target R = (I + 3R)/2 if R > 0 else I
        x = tf.reshape(x, [B, H * 2, W, 2, I + R])
        xl = x[:, :, :, 0, :I]
        xr = x[:, :, :, 1, :I]
        x_avg = (xl + xr) * 0.5  # shape = [B, H * 2, W, I]
        x_res = xl - xr  # shape = [B, H * 2, W, I]
        if R > 0:
            x = tf.concat([x_avg, x_res, x[:, :, :, 0, I:], x[:, :, :, 1, I:]], axis=3)
            if R > I:
                x = self.prenorm1_layer(x, training=training)
                x = self.expand1_layer(
                    x, training=training
                )  # shape = [B, H * 2, W, I * 2 + R * 4]
            x = self.prenorm2_layer(x, training=training)
            x = self.project1_layer(x, training=training)  # shape = [B, H*2, W, RR]
            RR = (I + R * 3 + 1) // 2
        else:
            x = x_res
            RR = I
        x_avg = tf.reshape(x_avg, [B, H, 2, W, I])
        x = tf.reshape(x, [B, H, 2, W, RR])

        # merge pairs of consecutive pixels in each column
        xt = x_avg[:, :, 0, :, :]
        xb = x_avg[:, :, 1, :, :]
        x_avg = (xt + xb) * 0.5  # shape = [B, H, W, I]
        x_res = xt - xb  # shape = [B, H, W, I]
        x = tf.concat([x_avg, x_res, x[:, :, 0, :, :], x[:, :, 1, :, :]], axis=3)
        if R > 0:
            x = self.prenorm3_layer(x, training=training)
            x = self.expand2_layer(x, training=training)  # shape = [B, H, W, I*2+RR*4]
        x = self.prenorm4_layer(x, training=training)
        x = self.project2_layer(x, training=training)  # shape = [B, H, W, I + 2 * R]
        x = tf.concat([x_avg, x], axis=3)  # shape = [B, H, W, 2 * (I + R)]

        # output
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
        }
        base_config = super(Downsize2D_V3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__


Downsize2D_V4 = Downsize2D_V3  # to be removed in future


# ----- v5 ------


class DUCLayerV5(DUCLayer):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(DUCLayerV5, self).__init__(
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.I = img_dim
        self.R = res_dim
        if res_dim == 0:
            self.RX = img_dim
            self.RY = img_dim
        else:
            self.RX = (img_dim + res_dim * 3 + 1) // 2
            self.RY = img_dim + res_dim * 2

    def get_config(self):
        config = {
            "img_dim": self.I,
            "res_dim": self.R,
        }
        base_config = super(DUCLayerV5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = DUCLayer.get_config.__doc__


class DownsizeX2D(DUCLayerV5):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(DownsizeX2D, self).__init__(
            img_dim,
            res_dim,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        if res_dim > 0:
            if res_dim > img_dim:
                self.prenorm1_layer = tf.keras.layers.LayerNormalization(
                    name="prenorm1"
                )
                self.expand1_layer = tf.keras.layers.Conv2D(
                    (self.I + self.R) * 4,
                    self._kernel_size,
                    padding="same",
                    activation="swish",
                    kernel_initializer=self._kernel_initializer,
                    bias_initializer=self._bias_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    kernel_constraint=self._kernel_constraint,
                    bias_constraint=self._bias_constraint,
                    name="expand1",
                )
            self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
            self.project1_layer = tf.keras.layers.Conv2D(
                self.RX,
                1,
                padding="same",
                activation="sigmoid",  # (0., 1.)
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="project1",
            )

    def call(self, x, training: bool = False):
        # shape
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1]
        W = input_shape[2] // 2

        # merge pairs of consecutive pixels in each row
        if self.R > 0:
            x = tf.reshape(x, [B, H, W, 2, self.I + self.R])
            xl = x[:, :, :, 0, :]
            xr = x[:, :, :, 1, :]
            x_avg = (xl + xr) * 0.5
            x_res = xl - xr
            x = tf.concat([x_avg, x_res], axis=3)
            if self.R > self.I:
                x = self.prenorm1_layer(x, training=training)
                x = self.expand1_layer(x, training=training)  # shape = [B,H,W,(I+R)*4]
            x = self.prenorm2_layer(x, training=training)
            x = self.project1_layer(x, training=training)  # shape = [B, H, W, RX]
            x_avg = x_avg[:, :, :, : self.I]
            x = tf.concat([x_avg, x], axis=3)  # shape = [B, H, W, I + RX]
        else:
            x = tf.reshape(x, [B, H, W, self.I * 2])

        # output
        return x

    call.__doc__ = DUCLayerV5.call.__doc__


class UpsizeX2D(DUCLayerV5):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(UpsizeX2D, self).__init__(
            img_dim,
            res_dim,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        if res_dim > 0:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expand1_layer = tf.keras.layers.Conv2D(
                (self.I + self.R) * 4,
                self._kernel_size,
                padding="same",
                activation="swish",
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="expand1",
            )
            self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
            self.project1_layer = tf.keras.layers.Conv2D(
                self.R,
                1,
                padding="same",
                activation="sigmoid",  # (0., 1.)
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="project1",
            )
            self.project2_layer = tf.keras.layers.Conv2D(
                self.I + self.R,
                1,
                padding="same",
                activation="tanh",  # (-1., 1.)
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="project2",
            )

    def call(self, x, training: bool = False):
        # shape
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]

        # split pairs of consecutive pixels in each row
        if self.R > 0:
            x_avg = x[:, :, :, : self.I]
            x = self.prenorm1_layer(x, training=training)
            x = self.expand1_layer(x, training=training)  # shape = [B, H, W, (I+RX)*2]
            x = self.prenorm2_layer(x, training=training)
            x1 = self.project1_layer(x, training=training)  # shape = [B, H, W, R]
            x = self.project2_layer(x, training=training)  # shape = [B, H, W, I + R]
            x_avg = tf.concat([x_avg, x1], axis=3)
            x = tf.concat([x_avg + x, x_avg - x], axis=3)
            x = tf.reshape(x, [B, H, W * 2, self.I + self.R])
        else:
            x = tf.reshape(x, [B, H, W * 2, self.I])

        # output
        return x

    call.__doc__ = DUCLayerV5.call.__doc__


class DownsizeY2D(DUCLayerV5):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(DownsizeY2D, self).__init__(
            img_dim,
            res_dim,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        if self.R > 0:
            self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
            self.expand1_layer = tf.keras.layers.Conv2D(
                (self.I + self.RX) * 4,
                self._kernel_size,
                padding="same",
                activation="swish",
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                name="expand1",
            )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        self.project1_layer = tf.keras.layers.Conv2D(
            self.RY,
            1,
            padding="same",
            activation="sigmoid",  # (0., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project1",
        )

    def call(self, x, training: bool = False):
        # shape
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1] // 2
        W = input_shape[2]

        # merge pairs of consecutive pixels in each column
        x = tf.reshape(x, [B, H, 2, W, self.I + self.RX])
        xt = x[:, :, 0, :, :]
        xb = x[:, :, 1, :, :]
        x_avg = (xt + xb) * 0.5
        x_res = xt - xb
        x = tf.concat([x_avg, x_res], axis=3)
        if self.R > 0:
            x = self.prenorm1_layer(x, training=training)
            x = self.expand1_layer(x, training=training)  # shape = [B, H, W, I*2+RX*4]
        x = self.prenorm2_layer(x, training=training)
        x = self.project1_layer(x, training=training)  # shape = [B, H, W, RY]

        # output
        x_avg = x_avg[:, :, :, : self.I]
        x = tf.concat([x_avg, x], axis=3)  # shape = [B, H, W, I + RY]
        return x

    call.__doc__ = DUCLayerV5.call.__doc__


class UpsizeY2D(DUCLayerV5):
    """Downsizing along the x-axis and the y-axis using convolutions of residuals.

    Downsizing means halving the width and the height and doubling the number of channels.

    TBC

    This layer is supposed to be nearly an inverse of the Upsize2D layer.

    Input dimensionality consists of image dimensionality and residual dimensionality.

    Parameters
    ----------
    img_dim : int
        the image dimensionality
    res_dim : int
        the residual dimensionality
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
        kernel_size: tp.Union[int, tuple, list] = 3,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(UpsizeY2D, self).__init__(
            img_dim,
            res_dim,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.prenorm1_layer = tf.keras.layers.LayerNormalization(name="prenorm1")
        self.expand1_layer = tf.keras.layers.Conv2D(
            (self.I + self.RX) * 4,
            self._kernel_size,
            padding="same",
            activation="swish",
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="expand1",
        )
        self.prenorm2_layer = tf.keras.layers.LayerNormalization(name="prenorm2")
        self.project1_layer = tf.keras.layers.Conv2D(
            self.RX,
            1,
            padding="same",
            activation="sigmoid",  # (0., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project1",
        )
        self.project2_layer = tf.keras.layers.Conv2D(
            self.I + self.RX,
            1,
            padding="same",
            activation="tanh",  # (-1., 1.)
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name="project2",
        )

    def call(self, x, training: bool = False):
        # shape
        input_shape = tf.shape(x)
        B = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]

        # split pairs of consecutive pixels in each column
        x_avg = x[:, :, :, : self.I]
        x = self.prenorm1_layer(x, training=training)
        x = self.expand1_layer(x, training=training)  # shape = [B, H, W, (I + RY) * 2]
        x = self.prenorm2_layer(x, training=training)
        x1 = self.project1_layer(x, training=training)  # shape = [B, H, W, RX]
        x = self.project2_layer(x, training=training)  # shape = [B, H, W, I + RX]
        x_avg = tf.concat([x_avg, x1], axis=3)

        # output
        x = tf.concat([x_avg + x, x_avg - x], axis=3)
        x = tf.reshape(x, [B, H, W, 2, self.I + self.RX])
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4])
        x = tf.reshape(x, [B, H * 2, W, self.I + self.RX])
        return x

    call.__doc__ = DUCLayerV5.call.__doc__
