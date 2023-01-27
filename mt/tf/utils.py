"""Useful subroutines dealing with GPU devices."""

from mt import tp, np

__all__ = ["gpus_in_tf_format", "as_floatx", "sigmoid", "asigmoid", "NameScope"]


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


class NameScope:
    """An iterator that generates name scope prefixes, mostly for Keras layers.

    Parameters
    ----------
    name : str
        the name of the scope
    parent_scope : NameScope, optional
        the parent name scope

    Methods
    -------
    __call__
        Gets the full name of a base name, with prefix generated from the name scope.

    Examples
    --------
    >>> from mt import tf
    >>> name_scope = tf.NameScope("myblock")
    >>> name_scope("a")
    'myblock_0/a'
    >>> name_scope("b")
    'myblock_0/b'
    >>> next(name_scope)
    >>> name_scope("c")
    'myblock_1/c'
    >>> child_scope = tf.NameScope("childblock", parent_scope=name_scope)
    >>> child_scope("d")
    'myblock_1/childblock_0/d'

    """

    def __init__(self, name: str, parent_scope=None):
        self.name = name
        self.parent_scope = parent_scope
        self.__iter__()

    def __iter__(self):
        self._cnt = 0
        self._prefix = self.name + "_0"

    def __next__(self):
        self._cnt += 1
        self._prefix = "{}_{}".format(self.name, self._cnt)

    def prefix(self, full: bool = False):
        """Returns the current prefix, with or without any parent prefix."""

        if full and isinstance(self.parent_scope, NameScope):
            return "{}/{}".format(self.parent_scope.prefix(), self._prefix)

        return self._prefix

    def __call__(self, base_name: str):
        return "{}/{}".format(self.prefix(True), base_name)
