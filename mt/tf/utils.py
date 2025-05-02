"""Useful subroutines dealing with GPU devices."""

from mt import tp, np, tfc

__api__ = ["gpus_in_tf_format", "as_floatx", "sigmoid", "asigmoid"]


NameScope = tfc.NameScope  # for backward compatibility


def gpus_in_tf_format(gpus):
    """Converts a gpu list or a gpu count into a list of GPUs in TF format."""

    if isinstance(gpus, int):
        gpus = range(gpus)
    return ["/GPU:{}".format(x) for x in gpus]


def as_floatx(x):
    """Ensures that a tensor is of dtype floatx."""

    import tensorflow as tf

    if not np.issubdtype(x.dtype.as_numpy_dtype, np.floating):
        x = tf.cast(x, tf.keras.backend.floatx())
    return x


def sigmoid(x):
    """Stable sigmoid, taken from tfp."""

    import tensorflow as tf

    x = tf.convert_to_tensor(x)
    cutoff = -20 if x.dtype == tf.float64 else -9
    return tf.where(x < cutoff, tf.exp(x), tf.math.sigmoid(x))


def asigmoid(y):
    """Inverse of sigmoid, taken from tfp."""

    import tensorflow as tf

    return tf.math.log(y) - tf.math.log1p(-y)
