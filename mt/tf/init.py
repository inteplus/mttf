"""Initialises TensorFlow, monkey-patching if necessary."""

from packaging.version import Version

__all__ = ["init"]


def init():
    """Initialises tensorflow, monkey-patching if necessary."""

    import tensorflow
    import sys

    tf_ver = Version(tensorflow.__version__)

    if tf_ver < Version("2.10"):
        raise ImportError(
            f"The minimum TF version that mttf supports is 2.10. Your TF is {tf_ver}. "
            "Please upgrade."
        )

    return tensorflow


init()
