from .. import layers


class Identical(layers.Layer):
    """An identical layer, mainly for renaming purposes."""

    def call(self, x):
        return x

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
