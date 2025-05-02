from .base import keras_version, keras_source

if keras_source == "tf_keras":
    from tf_keras import *
elif keras_source == "keras":
    from keras import *
elif keras_source == "tensorflow.keras":
    from tensorflow.keras import *
else:
    raise ImportError(f"Unknown value '{keras_source}' for variable 'keras_source'.")
