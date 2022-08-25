'''Useful subroutines dealing with GPU devices.'''

__all__ = ['gpus_in_tf_format', 'as_floatx']


def gpus_in_tf_format(gpus):
    '''Converts a gpu list or a gpu count into a list of GPUs in TF format.'''

    if isinstance(gpus, int):
        gpus = range(gpus)
    return ['/GPU:{}'.format(x) for x in gpus]


def as_floatx(x):
    '''Ensures that a tensor is of dtype floatx.'''

    from mt import np
    import tensorflow as tf

    if not np.issubdtype(x.dtype.as_numpy_dtype, np.floating):
        x = tf.cast(x, tf.keras.backend.floatx())
    return x
