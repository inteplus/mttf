'''Imports TensorFlow, monkey-patching if necessary.'''

__all__ = ['import_tf']

def import_tf():
    '''Imports tensorflow, monkey-patching if necessary.'''

    import tensorflow

    if tensorflow.__version__.startswith('2.'):
        import h5py as _h5
        if _h5.__version__.startswith('3.'): # hack because h5py>=3.0.0 behaves differently from h5py<3.0.0
            from .keras_engine import hdf5_format
            tensorflow.python.keras.saving.hdf5_format = hdf5_format
            tensorflow.python.keras.engine.training.hdf5_format = hdf5_format
    
    return tensorflow
