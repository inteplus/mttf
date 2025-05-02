import tensorflow as tf
from .. import layers, initializers


class Counter(layers.Layer):
    """A layer that counts from 0 during training and does nothing during inference."""

    def build(self, input_shape):
        initializer = initializers.Constant(value=0.0)
        self.counter = self.add_weight(
            name="counter", shape=(1,), initializer=initializer
        )
        self.incrementor = tf.constant([1.0])

    def call(self, x, training: bool = False):
        if training:
            self.counter.assign_add(self.incrementor)
        y = tf.reshape(x, [-1])[:1]
        y = tf.stop_gradient(y) * 0.0
        return self.counter + y

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return (1,)

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
