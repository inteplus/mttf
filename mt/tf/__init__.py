from .version import version as __version__

try:
    import tensorflow as _tf
except ImportError:
    raise ImportError("You need to install tensorflow.")
