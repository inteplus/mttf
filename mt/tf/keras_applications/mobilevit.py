# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileViT model.

Most of the code here has been ripped and updated off from the following
`Keras tutorial <https://keras.io/examples/vision/mobilevit/>`_. Please refer
to the `MobileViT ICLR2022 paper <https://arxiv.org/abs/2110.02178>`_ for more details.

The paper authors' code is `here <https://github.com/apple/ml-cvnets>`_.
"""


from mt import tp, tfc, tf


try:
    from tensorflow.keras.applications.mobilenet_v3 import _inverted_res_block
except ImportError:
    try:
        from keras.applications.mobilenet_v3 import _inverted_res_block
    except:
        from .mobilenet_v3 import _inverted_res_block


try:
    import keras
    from keras import backend
    from keras import models
    from keras.layers import VersionAwareLayers
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import backend
        from tensorflow.keras import models
        from tensorflow.keras.layers import VersionAwareLayers
    except ImportError:
        from tensorflow.python import keras
        from tensorflow.python.keras import backend
        from tensorflow.python.keras import models
        from tensorflow.python.keras.layers import VersionAwareLayers


from .mobilenet_v3_split import MobileNetV3Input


layers = VersionAwareLayers()


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(
    x, expanded_channels, output_channels, strides=1, block_id=0
):
    if block_id == 0:
        raise NotImplementedError(
            "Zero block id for _inverted_res_block() is not implemented in MobileViT."
        )

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    infilters = backend.int_shape(x)[channel_axis]

    m = _inverted_res_block(
        x,
        expanded_channels // infilters,  # expansion
        output_channels,  # filters
        3,  # kernel_size
        strides,  # stride
        0,  # se_ratio
        tf.nn.swish,  # activation
        block_id,
    )

    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    cell_size = 2  # 2x2 for the Transformer block

    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    if x.shape[1] % cell_size != 0:
        raise tfc.ModelSyntaxError(
            "Input tensor must have height divisible by {}. Got {}.".format(
                cell_size, x.shape
            )
        )

    if x.shape[2] % cell_size != 0:
        raise tfc.ModelSyntaxError(
            "Input tensor must have width divisible by {}. Got {}.".format(
                cell_size, x.shape
            )
        )

    # Unfold into patches and then pass through Transformers.
    z = local_features  # (B,H,W,C)
    z = layers.Reshape(
        (
            z.shape[1] // cell_size,
            cell_size,
            z.shape[2] // cell_size,
            cell_size,
            projection_dim,
        )
    )(
        z
    )  # (B,H/P,P,W/P,P,C)
    z = tf.transpose(z, perm=[0, 2, 4, 1, 3, 5])  # (B,P,P,H/P,W/P,C)
    non_overlapping_patches = layers.Reshape(
        (cell_size * cell_size, z.shape[3] * z.shape[4], projection_dim)
    )(
        z
    )  # (B,P*P,H*W/(P*P),C)
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    z = layers.Reshape(
        (
            cell_size,
            cell_size,
            x.shape[1] // cell_size,
            x.shape[2] // cell_size,
            projection_dim,
        )
    )(
        global_features
    )  # (B,P,P,H/P,W/P,C)
    z = tf.transpose(z, perm=[0, 3, 1, 4, 2, 5])  # (B,H/P,P,W/P,P,C)
    folded_feature_map = layers.Reshape((x.shape[1], x.shape[2], projection_dim))(z)

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convolution layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def create_mobilevit(
    input_shape=None,
    model_type: str = "XXS",
    output_all: bool = False,
    name: tp.Optional[str] = None,
):
    """Prepares a model of submodels which is equivalent to a MobileNetV3 model.

    Parameters
    ----------
    input_shape : tuple
        Optional shape tuple, to be specified if you would like to use a model with an input image
        resolution that is not (224, 224, 3). It should have exactly 3 inputs channels
        (224, 224, 3). You can also omit this option if you would like to infer input_shape from an
        input_tensor. If you choose to include both input_tensor and input_shape then input_shape
        will be used if they match, if the shapes do not match then we will throw an error. E.g.
        `(160, 160, 3)` would be one valid value.
    model_type : {'XXS', 'XS', 'S'}
        one of the 3 variants introduced in the paper
    output_all : bool
        If True, the model returns the output tensor of every block before down-sampling, other
        than the input layer.  Otherwise, it returns the output tensor of the last block.
    name : str, optional
        model name, if any. Default to 'MobileViT<model_type>'.

    Returns
    -------
    tensorflow.keras.Model
        the output MobileViT model
    """

    model_type_id = ["XXS", "XS", "S"].index(model_type)

    expansion_factor = 2 if model_type_id == 0 else 4

    inputs = MobileNetV3Input(input_shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x,
        expanded_channels=16 * expansion_factor,
        output_channels=16 if model_type_id == 0 else 32,
        block_id=1,
    )
    outputs = [x]

    # Downsampling with MV2 block.
    output_channels = [24, 48, 64][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=16 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=2,
    )
    x = inverted_residual_block(
        x,
        expanded_channels=24 * expansion_factor,
        output_channels=output_channels,
        block_id=3,
    )
    x = inverted_residual_block(
        x,
        expanded_channels=24 * expansion_factor,
        output_channels=output_channels,
        block_id=4,
    )
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # First MV2 -> MobileViT block.
    output_channels = [48, 64, 96][model_type_id]
    projection_dim = [64, 96, 144][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=48 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=5,
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=projection_dim)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # Second MV2 -> MobileViT block.
    output_channels = [64, 80, 128][model_type_id]
    projection_dim = [80, 120, 192][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=64 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=6,
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=projection_dim)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # Third MV2 -> MobileViT block.
    output_channels = [80, 96, 160][model_type_id]
    projection_dim = [96, 144, 240][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=80 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=7,
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=projection_dim)
    filters = [320, 384, 640][model_type_id]
    x = conv_block(x, filters=filters, kernel_size=1, strides=1)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    if name is None:
        name = "MobileViT{}".format(model_type)
    return keras.Model(inputs, outputs, name=name)
