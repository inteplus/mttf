from .. import ops, backend


class SoftBend(ops.Operation):
    """Soft bend activation function.

    Function: `|x|^alpha * tanh(x)`, bending the linear activation a bit.

    If alpha is less than 1, it acts as a soft squash.
    If alpha is greater than 1, it acts as a soft explode.
    If alpha is 1, it acts as the linear activation function.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def call(self, x):
        from tensorflow.math import pow, abs, tanh

        return pow(abs(x), self.alpha) * tanh(x)

    def compute_output_spec(self, x):
        return backend.KerasTensor(x.shape, x.dtype)
