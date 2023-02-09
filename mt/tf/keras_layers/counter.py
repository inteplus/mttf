import tensorflow as tf
import tensorflow.keras as tk


class Counter(tk.layers.Layer):
    """A layer that counts from 0 during training and does nothing during inference."""

    def build(self, input_shape):
        initializer = tk.initializers.Constant(value=0.0)
        self.counter = self.add_weight(shape=(1,), initializer=initializer)
        self.incrementor = tf.constant([1.0])

    def call(self, x, training: bool = False):
        if training:
            self.counter.assign_add(self.incrementor)
        y = tf.reshape(x, [-1])[:1]
        y = tf.stop_gradient(y) * 0.0
        return self.counter + y

    call.__doc__ = tk.layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return (1,)

    compute_output_shape.__doc__ = tk.layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        config = {
            "init_val": self.init_val,
            "inc": self.inc,
        }
        base_config = super(Counter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
