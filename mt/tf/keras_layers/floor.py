import tensorflow as tf


@tf.custom_gradient
def floor(x):
    def grad(upstream):  # identity
        return upstream

    return tf.math.floor(x), grad


class Floor(tf.keras.layers.Layer):
    """TensorFlow floor but gradient is identity."""

    def call(self, x):
        return floor(x)

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = tf.keras.layers.Layer.compute_output_shape.__doc__
