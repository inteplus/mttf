'''Initialises TensorFlow, monkey-patching if necessary.'''

__all__ = ['init']

def init():
    '''Initialises tensorflow, monkey-patching if necessary.'''

    import tensorflow
    import sys

    if tensorflow.__version__.startswith('2.') and tensorflow.__version__ < '2.5':
        # monkey-patch mobilenet_v3
        from .keras_applications import mobilenet_v3
        if tensorflow.__version__ < '2.4':
            setattr(tensorflow.python.keras.applications, 'mobilenet_v3', mobilenet_v3)
        setattr(tensorflow.keras.applications, 'mobilenet_v3', mobilenet_v3)
        setattr(tensorflow.keras.applications, 'MobileNetV3Small', mobilenet_v3.MobileNetV3Small)
        setattr(tensorflow.keras.applications, 'MobileNetV3Large', mobilenet_v3.MobileNetV3Large)
        sys.modules['tensorflow.python.keras.applications.mobilenet_v3'] = mobilenet_v3

        # monkey-patch CosineDecay
        from .keras_optimizers import lr_extra
        setattr(tensorflow.keras.optimizers.schedules, 'CosineDecay', lr_extra.CosineDecay)
        sys.modules['tensorflow.python.keras.optimizers.lr_extra'] = lr_extra
 
    if tensorflow.__version__.startswith('2.') and tensorflow.__version__ < '2.5':
        import h5py as _h5
        if _h5.__version__.startswith('3.'): # hack because h5py>=3.0.0 behaves differently from h5py<3.0.0
            from .keras_engine import hdf5_format
            tensorflow.python.keras.saving.hdf5_format = hdf5_format
            tensorflow.python.keras.engine.training.hdf5_format = hdf5_format
            sys.modules['tensorflow.python.keras.saving.hdf5_format'] = hdf5_format
    
    return tensorflow

init()
