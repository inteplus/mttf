import tensorflow as tf

from mt import tp


class VarianceRegularizer(tf.keras.layers.Layer):
    """A regularizer on the variance of the input tensor.

    Negative rate for making the variance larger. Positive rate for making the variance smaller.
    """

    def __init__(self, rate=1e-2, l_axes: list = [-1], **kwargs):
        super(VarianceRegularizer, self).__init__(**kwargs)
        self.rate = rate
        self.l_axes = l_axes

    def call(self, x):
        mean = tf.reduce_mean(x, axis=self.l_axes, keepdims=True)
        err = x - mean
        esq = err * err
        var = tf.reduce_mean(esq, axis=self.l_axes)
        sum_var = tf.reduce_sum(var)
        self.add_loss(self.rate * sum_var)
        return x

    call.__doc__ = tf.keras.layers.Layer.call.__doc__

    def get_config(self):
        config = {
            "rate": self.rate,
            "l_axes": self.l_axes,
        }
        base_config = super(VarianceRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = tf.keras.layers.Layer.get_config.__doc__
