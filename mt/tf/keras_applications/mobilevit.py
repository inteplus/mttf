# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileViT model.

Most of the code here has been ripped off from the following
`Keras tutorial <https://keras.io/examples/vision/mobilevit/>`_. Please refer
to the MobileViT ICLR2022 paper for more details.
"""


import typing as tp

from mt import tf


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


layers = VersionAwareLayers()


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1, block_id=0):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    infilters = backend.int_shape(x)[channel_axis]

    m = _inverted_res_block(
        x,
        expanded_channels // infilters,  # expansion
        output_channels,  # filters
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
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2,
                 x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int(
        (local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(
        axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def create_mobilevit(num_classes=5):
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16,
        block_id=1,
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2,
        block_id=2,
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24,
        block_id=3,
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24,
        block_id=4,
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2,
        block_id=5,
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2,
        block_id=6,
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2,
        block_id=7,
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)
