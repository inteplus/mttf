# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real NVP bijector.

An adaptation made by MT for tfp <= 0.17.0.
"""

from mt import tf

from tensorflow_probability.python.internal import tensorshape_util


__all__ = ["real_nvp_default_template"]


def real_nvp_default_template(
    hidden_layers,
    shift_only=False,
    activation="relu",
    name=None,
    *args,  # pylint: disable=keyword-arg-before-vararg
    **kwargs
):
    """Build a scale-and-shift function using a multi-layer neural network.

    This will be wrapped in a make_template to ensure the variables are only
    created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
    dimensional outputs `loc` ('mu') and `log_scale` ('alpha').

    The default template does not support conditioning and will raise an
    exception if `condition_kwargs` are passed to it. To use conditioning in
    Real NVP bijector, implement a conditioned shift/scale template that
    handles the `condition_kwargs`.

    Args:
      hidden_layers: Python `list`-like of non-negative integer, scalars
        indicating the number of units in each hidden layer. Default: `[512,
          512]`.
      shift_only: Python `bool` indicating if only the `shift` term shall be
        computed (i.e. NICE bijector). Default: `False`.
      activation: Activation function (callable). Explicitly setting to `None`
        implies a linear activation.
      name: A name for ops managed by this function. Default:
        'real_nvp_default_template'.
      *args: `tensorflow.keras.layers.Dense` arguments.
      **kwargs: `tensorflow.keras.layers.Dense` keyword arguments.

    Returns:
      shift: `Float`-like `Tensor` of shift terms ('mu' in
        [Papamakarios et al.  (2016)][1]).
      log_scale: `Float`-like `Tensor` of log(scale) terms ('alpha' in
        [Papamakarios et al. (2016)][1]).

    Raises:
      NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
        graph execution, or if `condition_kwargs` is not empty.

    Notes:
      This version has been adapted to TF2 by MT.

    #### References

    [1]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
         Autoregressive Flow for Density Estimation. In _Neural Information
         Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
    """

    def fn(x, output_units, **condition_kwargs):
        """Fully connected MLP parameterized via `real_nvp_template`."""
        if condition_kwargs:
            raise NotImplementedError(
                "Conditioning not implemented in the default template."
            )

        with tf.name_scope(name or "real_nvp_default_template"):

            if tensorshape_util.rank(x.shape) == 1:
                x = x[tf.newaxis, ...]
                reshape_output = lambda x: x[0]
            else:
                reshape_output = lambda x: x
            for units in hidden_layers:
                x = tf.keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    *args,  # pylint: disable=keyword-arg-before-vararg
                    **kwargs
                )(x)
            x = tf.keras.layers.Dense(
                units=(1 if shift_only else 2) * output_units,
                activation=None,
                kernel_initializer="zeros",
                bias_initializer="zeros",
                *args,  # pylint: disable=keyword-arg-before-vararg
                **kwargs
            )(x)
            if shift_only:
                return reshape_output(x), None
            shift, log_scale = tf.split(x, 2, axis=-1)
            return reshape_output(shift), reshape_output(log_scale)

    return fn
