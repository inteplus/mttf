# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileNet v3 models split into 5 submodels.

The MobileNetV3 model is split into 5 parts:

  - The input parser block that downsamples once (:func:`MobileNetV3Parser`).
  - Block 0 to 3 that downample once for each block (:func:`MobileNetV3LargeBlock`
    or :func:`MobileNetV3SmallBlock`).
  - The mixer block that turns the downsampled grid into a (1,1,feat_dim) batch
    (:func:`MobileNetV3Mixer`).
  - Optionally the output block that may or may not contain the clasification head
    (:func:`MobileNetV3Output`).

Input arguments follow those of MobileNetV3. One can also use :func:`MobileNetV3Split` to create
a model of submodels that is theoretically equivalent to the original MobileNetV3 model. However,
no pre-trained weights exist.
"""


__all__ = [
  'MobileNetV3Input',
  'MobileNetV3Parser',
  'MobileNetV3Block',
  'MobileNetV3Mixer',
  'MobileNetV3Output',
  'MobileNetV3Split',
]


try:
  from tensorflow.keras.applications.mobilenet_v3 import relu, hard_swish, _depth, _inverted_res_block
except ImportError:
  try:
    from keras.applications.mobilenet_v3 import relu, hard_swish, _depth, _inverted_res_block
  except:
    from .mobilenet_v3 import relu, hard_swish, _depth, _inverted_res_block


try:
    from keras import backend
    from keras import models
    from keras.layers import VersionAwareLayers
    from keras.utils import data_utils, layer_utils
except ImportError:
    from tensorflow.python.keras import backend
    from tensorflow.python.keras import models
    from tensorflow.python.keras.layers import VersionAwareLayers
    from tensorflow.python.keras.utils import data_utils, layer_utils
from tensorflow.python.platform import tf_logging as logging


layers = VersionAwareLayers()


def MobileNetV3Input(
    input_shape=None,
):
  '''Prepares a MobileNetV3 input layer.'''

  # If input_shape is None and input_tensor is None using standard shape
  if input_shape is None:
    input_shape = (None, None, 3)

  if backend.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
  else:
    row_axis, col_axis = (1, 2)
  rows = input_shape[row_axis]
  cols = input_shape[col_axis]
  if rows and cols and (rows < 32 or cols < 32):
    raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                     str(input_shape) + '`')

  img_input = layers.Input(shape=input_shape)
  return img_input


def MobileNetV3Parser(
    img_input,
    model_type: str = 'Large', # only 'Small' or 'Large' are accepted
    minimalistic=False,
):
  '''Prepares a MobileNetV3 parser block.'''

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

  if minimalistic:
    activation = relu
  else:
    activation = hard_swish

  x = img_input
  x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)
  x = layers.Conv2D(
      16,
      kernel_size=3,
      strides=(2, 2),
      padding='same',
      use_bias=False,
      name='Conv')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv/BatchNorm')(x)
  x = activation(x)

  # Create model.
  model = models.Model(img_input, x, name='MobileNetV3{}Parser'.format(model_type))

  return model


def MobileNetV3SmallBlock(
    block_id: int, # only 0 to 3 are accepted here
    input_tensor, # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
  '''Prepares a MobileNetV3Small downsampling block.'''

  def depth(d):
    return _depth(d * alpha)

  if minimalistic:
    kernel = 3
    activation = relu
    se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  x = input_tensor
  if block_id == 0:
    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
  elif block_id == 1:
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
  elif block_id == 2:
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
  else:
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)

  # Create model.
  model = models.Model(input_tensor, x, name='MobileNetV3SmallBlock{}'.format(block_id))

  return model


def MobileNetV3LargeBlock(
    block_id: int, # only 0 to 3 are accepted here
    input_tensor, # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
  '''Prepares a MobileNetV3Large downsampling block.'''

  def depth(d):
    return _depth(d * alpha)

  if minimalistic:
    kernel = 3
    activation = relu
    se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  x = input_tensor
  if block_id == 0:
    x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
    x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
  elif block_id == 1:
    x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
  elif block_id == 2:
    x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
    x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
  else:
    x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)

  # Create model.
  model = models.Model(input_tensor, x, name='MobileNetV3LargeBlock{}'.format(block_id))

  return model


def MobileNetV3Mixer(
    input_tensor,
    last_point_ch,
    alpha=1.0,
    model_type: str = 'Large', # only 'Small' or 'Large' are accepted
    minimalistic=False,
):
  '''Prepares a MobileNetV3 mixer block.'''

  x = input_tensor
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

  if minimalistic:
    kernel = 3
    activation = relu
    se_ratio = None
  else:
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

  last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if alpha > 1.0:
    last_point_ch = _depth(last_point_ch * alpha)
  x = layers.Conv2D(
      last_conv_ch,
      kernel_size=1,
      padding='same',
      use_bias=False,
      name='Conv_1')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3,
      momentum=0.999, name='Conv_1/BatchNorm')(x)
  x = activation(x)
  x = layers.GlobalAveragePooling2D()(x)
  if channel_axis == 1:
    x = layers.Reshape((last_conv_ch, 1, 1))(x)
  else:
    x = layers.Reshape((1, 1, last_conv_ch))(x)
  x = layers.Conv2D(
      last_point_ch,
      kernel_size=1,
      padding='same',
      use_bias=True,
      name='Conv_2')(x)
  x = activation(x)

  # Create model.
  model = models.Model(input_tensor, x, name='MobilenetV3{}Mixer'.format(model_type))

  return model


def MobileNetV3Output(
    input_tensor,
    model_type: str = 'Large', # only 'Small' or 'Large' are accepted
    include_top=True,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation='softmax',
):
  '''Prepares a MobileNetV3 output block.'''

  x = input_tensor
  if include_top:
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    x = layers.Activation(activation=classifier_activation,
                          name='Predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
      return None

  # Create model.
  model = models.Model(input_tensor, x, name='MobilenetV3{}Output'.format(model_type))

  return model


def MobileNetV3Split(
    input_shape=None,
    alpha=1.0,
    model_type: str = 'Large', # only 'Small' or 'Large' are accepted
    minimalistic=False,
    include_top=True,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation='softmax',
):
  '''Prepares a model of submodels which is equivalent to a MobileNetV3 model.'''

  input_layer = MobileNetV3Input(input_shape=input_shape)
  input_block = MobileNetV3Parser(
    input_layer,
    model_type=model_type,
    minimalistic=minimalistic,
  )
  x = input_block(input_layer)

  for i in range(4):
    if model_type == 'Large':
      block = MobileNetV3LargeBlock(
        i,
        x,
        alpha=alpha,
        minimalistic=minimalistic,
      )
    else:
      block = MobileNetV3SmallBlock(
        i,
        x,
        alpha=alpha,
        minimalistic=minimalistic,
      )
    x = block(x)

  if model_type == 'Large':
    last_point_ch = 1280
  else:
    last_point_ch = 1024
  mixer_block = MobileNetV3Mixer(
    x,
    last_point_ch,
    alpha=alpha,
    model_type=model_type,
    minimalistic=minimalistic,
  )
  x = mixer_block(x)

  output_block = MobileNetV3Output(
    x,
    model_type=model_type,
    include_top=include_top,
    classes=classes,
    pooling=pooling,
    dropout_rate=dropout_rate,
    classifier_activation=classifier_activation,
  )
  if output_block is not None:
    x = output_block(x)

  # Create model.
  model = models.Model(input_layer, x, name='MobilenetV3{}Split'.format(model_type))

  return model
