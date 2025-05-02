"""Determines the working Keras 2 from the system to be used by mt.keras."""

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
    keras_version = tf_keras.__version__
    keras_source = "tf_keras"
else:
    try:
        import keras

        kr_ver = Version(keras.__version__)
    except ImportError:
        kr_ver = None
    if kr_ver is None or kr_ver >= Version("3.0"):
        keras_version = tf.__version__
        keras_source = "tensorflow.python"
    else:
        keras_version = keras.__version__
        keras_source = "keras"
