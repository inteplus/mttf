from .. import tf
from tensorflow_probability import *
from tensorflow_probability import __version__

from packaging import version as V

if V.parse(__version__) <= V.parse("0.17.0"):
    from .real_nvp import real_nvp_default_template

    bijectors.real_nvp.real_nvp_default_template = real_nvp_default_template
    bijectors.real_nvp_default_template = real_nvp_default_template
