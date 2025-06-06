from .. import constraints


class CenterAround(keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self, ref_value: float = 0.0):
        self.ref_value = ref_value

    def __call__(self, w):
        import tensorflow as tf

        mean = tf.reduce_mean(w)
        return w - (mean - self.ref_value)

    def get_config(self):
        return {"ref_value": self.ref_value}
