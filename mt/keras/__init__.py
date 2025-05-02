"""MT package that represents the working Keras 2 from the system."""

from packaging.version import Version
import tensorflow as tf

tf_ver = Version(tf.__version__)
if tf_ver >= Version("2.16"):
    try:
        import tf_keras
    except:
        raise ImportError(
            f"mt.keras can only work with Keras 2. You have TF version {tf_ver}. Please install tf_keras."
        )
    from tf_keras import *

    __version__ = tf_keras.__version__
    __source__ = "tf_keras"
else:
    try:
        import keras

        kr_ver = Version(keras.__version__)
    except ImportError:
        kr_ver = None
    if kr_ver is None or kr_ver >= Version("3.0"):
        __version__ = tf.__version__
        __source__ = "tensorflow.python"
        from tensorflow.keras import *
    else:
        __version__ = keras.__version__
        __source__ = "keras"
        from keras import *
