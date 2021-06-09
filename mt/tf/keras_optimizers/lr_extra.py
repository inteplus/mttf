'''Additional optimizer-related classes.'''

import abc
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class CosineDecay(LearningRateSchedule): # available in TF 2.5
  """A LearningRateSchedule that uses a cosine decay schedule.

  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
      initial_learning_rate, decay_steps)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      alpha=0.0,
      name=None):
    """Applies cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    """
    super(CosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "CosineDecay"):
      initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      completed_fraction = global_step_recomp / decay_steps
      cosine_decayed = 0.5 * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))

      decayed = (1 - self.alpha) * cosine_decayed + self.alpha
      return math_ops.multiply(initial_learning_rate, decayed)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "alpha": self.alpha,
        "name": self.name
    }

