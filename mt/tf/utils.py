'''Useful subroutines dealing with GPU devices.'''

__all__ = ['gpus_in_tf_format']

def gpus_in_tf_format(gpus):
    '''Converts a gpu list or a gpu count into a list of GPUs in TF format.'''
    if isinstance(gpus, int):
        gpus = range(gpus)
    return ['/GPU:{}'.format(x) for x in gpus]
