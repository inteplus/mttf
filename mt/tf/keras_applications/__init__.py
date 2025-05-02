from mt.logg import logger

logger.warn_module_move("mt.tf.keras_applications", "mt.keras.applications")

from mt.keras.applications import *
