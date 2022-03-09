'''Initialises TensorFlow, monkey-patching if necessary.'''

from packaging import version

__all__ = ['init']

def init():
    '''Initialises tensorflow, monkey-patching if necessary.'''

    import tensorflow
    import sys

    tf_ver = version.parse(tensorflow.__version__)

    if tf_ver < version.parse('2.5'):
        # monkey-patch mobilenet_v3
        from .keras_applications import mobilenet_v3
        if tf_ver < version.parse('2.4'):
            setattr(tensorflow.python.keras.applications, 'mobilenet_v3', mobilenet_v3)
        setattr(tensorflow.keras.applications, 'mobilenet_v3', mobilenet_v3)
        setattr(tensorflow.keras.applications, 'MobileNetV3Small', mobilenet_v3.MobileNetV3Small)
        setattr(tensorflow.keras.applications, 'MobileNetV3Large', mobilenet_v3.MobileNetV3Large)
        sys.modules['tensorflow.python.keras.applications.mobilenet_v3'] = mobilenet_v3

        # monkey-patch CosineDecay
        from .keras_optimizers import lr_extra
        setattr(tensorflow.keras.optimizers.schedules, 'CosineDecay', lr_extra.CosineDecay)
        sys.modules['tensorflow.python.keras.optimizers.lr_extra'] = lr_extra
 
    if tf_ver < version.parse('2.5'):
        import h5py as _h5
        if _h5.__version__.startswith('3.'): # hack because h5py>=3.0.0 behaves differently from h5py<3.0.0
            from .keras_engine import hdf5_format
            tensorflow.python.keras.saving.hdf5_format = hdf5_format
            tensorflow.python.keras.engine.training.hdf5_format = hdf5_format
            sys.modules['tensorflow.python.keras.saving.hdf5_format'] = hdf5_format

    if tf_ver < version.parse('2.8'):
        # monkey-patch efficientnet_v2
        from .keras_applications import efficientnet_v2
        if tf_ver < version.parse('2.4'):
            sys.modules['tensorflow.python.keras.applications.efficientnet_v2'] = efficientnet_v2
            setattr(tensorflow.python.keras.applications, 'effiicientnet_v2', efficientnet_v2)
        sys.modules['tensorflow.keras.applications.efficientnet_v2'] = efficientnet_v2
        if tf_ver >= version.parse('2.7'):
            sys.modules['keras.applications.efficientnet_v2'] = efficientnet_v2
            import keras
            setattr(keras.applications, 'effiicientnet_v2', efficientnet_v2)
        setattr(tensorflow.keras.applications, 'efficientnet_v2', efficientnet_v2)
        setattr(tensorflow.keras.applications, 'EfficientNetV2B0', efficientnet_v2.EfficientNetV2B0)
        setattr(tensorflow.keras.applications, 'EfficientNetV2B1', efficientnet_v2.EfficientNetV2B1)
        setattr(tensorflow.keras.applications, 'EfficientNetV2B2', efficientnet_v2.EfficientNetV2B2)
        setattr(tensorflow.keras.applications, 'EfficientNetV2B3', efficientnet_v2.EfficientNetV2B3)
        setattr(tensorflow.keras.applications, 'EfficientNetV2S', efficientnet_v2.EfficientNetV2S)
        setattr(tensorflow.keras.applications, 'EfficientNetV2M', efficientnet_v2.EfficientNetV2M)
        setattr(tensorflow.keras.applications, 'EfficientNetV2L', efficientnet_v2.EfficientNetV2L)
    
    return tensorflow

init()
