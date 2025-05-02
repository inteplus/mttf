import tensorflow.keras.layers as _kl


class Identical(_kl.Layer):
    """An identical layer, mainly for renaming purposes."""

    def call(self, x):
        return x

    call.__doc__ = _kl.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = _kl.Layer.compute_output_shape.__doc__
