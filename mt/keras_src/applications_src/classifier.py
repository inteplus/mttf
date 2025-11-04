"""Standard classifier from a feature vector.
"""

from mt import tp, tfc, logg
from .. import models, layers, regularizers

from ..constraints_src import CenterAround


def create_classifier_block(
    input_dim: int,
    n_classes: int,
    name: str = "dense_classifier",
    params: tfc.ClassifierParams = tfc.ClassifierParams(),
    logger: tp.Optional[logg.IndentedLoggerAdapter] = None,
):
    """Creates a standard classifier block.

    Parameters
    ----------
    input_dim : int
        feature dimensionality of the input tensor
    n_classes : int
        number of output classes
    name : str, optional
        the name of the classifier block
    params : mt.tfc.ClassifierParams
        parameters for creating the classifier block
    logger : mt.logg.IndentedLoggerAdapter, optional
        logger for debugging purposes

    Returns
    -------
    model : tensorflow.keras.models.Model
        an uninitialised model without any compilation details representing the classifier block.
        The model returns `bv_logits` and `bv_probs`.
    """

    msg = f"Creating a classifier block of {n_classes} classes"
    with logg.scoped_info(msg, logger=logger):
        name_scope = tfc.NameScope(name)

        x = bv_feats = layers.Input(shape=(input_dim,), name=name_scope("input"))

        x = layers.LayerNormalization(name=name_scope("prenorm"))(x)

        # dropout, optional
        dropout = getattr(params, "dropout", None)
        if dropout is not None and dropout > 0 and dropout < 1:
            logg.info("Using dropout {dropout}.", logger=logger)
            x = layers.Dropout(dropout, name=name_scope("dropout"))(x)

        # Object classification branch
        # MT-TODO: currently l2_coeff does not take into account batch size. In order to be truly
        # independent of batch size, number of classes and feature dimensionality, the l2 coeff
        # should be l2_coeff / bv_feats.shape[1] / n_classes / batch_size. So we need to pass the
        # batch size to the function as an additional argument.
        l2_coeff = getattr(params, "l2_coeff", None)
        if l2_coeff is not None:
            logg.info(
                "Using param 'l2_coeff' for kernel and bias regularizers.",
                logger=logger,
            )
            logg.info(f"l2_coeff: {l2_coeff}", logger=logger)
            l2 = l2_coeff / bv_feats.shape[1] / n_classes
            logg.info(f"kernel_l2: {l2}", logger=logger)
            kernel_regularizer = regularizers.l2(l2)
            l2 = l2_coeff / n_classes
            logg.info(f"bias_l2: {l2}", logger=logger)
            bias_regularizer = regularizers.l2(l2)
        else:
            kernel_regularizer = None
            bias_regularizer = None

        # zero mean logit biases
        if getattr(params, "zero_mean_logit_biases", False):
            logg.info("Logit biases are constrainted to have zero mean.", logger=logger)
            bias_constraint = CenterAround()
        else:
            bias_constraint = None

        # dense layer
        bv_logits = x = layers.Dense(
            n_classes,
            name=name_scope("logits"),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
        )(x)

        bv_probs = x = layers.Softmax(name=name_scope("probs"))(x)

        # model
        model = models.Model(bv_feats, [bv_logits, bv_probs], name=name)

    return model


def MobileNetV3SmallBlock(
    block_id: int,  # only 0 to 3 are accepted here
    input_tensor,  # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
    """Prepares a MobileNetV3Small downsampling block."""

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
        x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
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
    model = models.Model(input_tensor, x, name=f"MobileNetV3SmallBlock{block_id}")

    return model


def MobileNetV3LargeBlock(
    block_id: int,  # only 0 to 4 are accepted here. 4 is only available as of 2023/05/15
    input_tensor,  # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
    """Prepares a MobileNetV3Large downsampling block."""

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
    elif block_id == 3:
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
    else:
        x = _inverted_res_block(x, 6, depth(320), kernel, 2, se_ratio, activation, 15)
        x = _inverted_res_block(x, 6, depth(320), kernel, 1, se_ratio, activation, 16)
        x = _inverted_res_block(x, 6, depth(320), kernel, 1, se_ratio, activation, 17)

    # Create model.
    model = models.Model(input_tensor, x, name=f"MobileNetV3LargeBlock{block_id}")

    return model


def MobileNetV3Mixer(
    input_tensor,
    params: tfc.MobileNetV3MixerParams,
    last_point_ch,
    alpha=1.0,
    model_type: str = "Large",  # only 'Small' or 'Large' are accepted
    minimalistic=False,
):
    """Prepares a MobileNetV3 mixer block."""

    x = input_tensor
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    if params.variant == "mobilenet":

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
            last_conv_ch, kernel_size=1, padding="same", use_bias=False, name="Conv_1"
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
        )(x)
        x = activation(x)
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_conv_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_conv_ch))(x)
        x = layers.Conv2D(
            last_point_ch, kernel_size=1, padding="same", use_bias=True, name="Conv_2"
        )(x)
        x = activation(x)
    elif params.variant == "maxpool":
        x = layers.GlobalMaxPool2D(x)
    elif params.variant == "mhapool":
        if backend.image_data_format() == "channels_first":
            raise tfc.ModelSyntaxError(
                "Mixer variant 'mhapool' requires channels_last image data format."
            )

        mhapool_params = params.mhapool_cascade_params
        if not isinstance(mhapool_params, tfc.MHAPool2DCascadeParams):
            raise tfc.ModelSyntaxError(
                "Parameter 'params.mhapool_cascade_params' is not of type "
                "mt.tfc.MHAPool2DCascadeParams. Got: {}.".format(type(mhapool_params))
            )

        from ..layers_src import MHAPool2D

        n_heads = mhapool_params.n_heads
        k = 0
        outputs = []
        while True:
            h = x.shape[1]
            w = x.shape[2]

            if h <= 1 and w <= 1:
                break

            c = x.shape[3]
            key_dim = (c + n_heads - 1) // n_heads
            value_dim = int(key_dim * mhapool_params.expansion_factor)
            k += 1
            block_name = f"MHAPool2DCascade_block{k}"
            if k > mhapool_params.max_num_pooling_layers:  # GlobalMaxPool2D
                x = layers.GlobalMaxPooling2D(
                    keepdims=True, name=block_name + "/GlobalMaxPool"
                )(x)
            else:  # MHAPool2D
                x = layers.LayerNormalization()(x)
                if h <= 2 and w <= 2:
                    activation = mhapool_params.final_activation
                else:
                    activation = mhapool_params.activation
                x = MHAPool2D(
                    n_heads,
                    key_dim,
                    value_dim=value_dim,
                    pooling=mhapool_params.pooling,
                    dropout=mhapool_params.dropout,
                    name=block_name + "/MHAPool",
                )(x)

            if mhapool_params.output_all:
                outputs.append(x)
            else:
                outputs = [x]
    else:
        raise tfc.ModelSyntaxError(
            "Unknown mixer variant: '{}'.".format(params.variant)
        )

    # Create model.
    model = models.Model(
        input_tensor, outputs, name="MobileNetV3{}Mixer".format(model_type)
    )

    return model


def MobileNetV3Output(
    input_tensor,
    model_type: str = "Large",  # only 'Small' or 'Large' are accepted
    include_top=True,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
):
    """Prepares a MobileNetV3 output block."""

    x = input_tensor
    if include_top:
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding="same", name="Logits")(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation, name="Predictions")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        else:
            return None

    # Create model.
    model = models.Model(input_tensor, x, name=f"MobileNetV3{model_type}Output")

    return model


def MobileNetV3Split(
    input_shape=None,
    alpha: float = 1.0,
    model_type: str = "Large",
    max_n_blocks: int = 6,
    minimalistic: bool = False,
    mixer_params: tp.Optional[tfc.MobileNetV3MixerParams] = None,
    include_top: bool = True,
    pooling=None,
    classes: int = 1000,
    dropout_rate: float = 0.2,
    classifier_activation="softmax",
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
    alpha : float
        controls the width of the network. This is known as the depth multiplier in the MobileNetV3
        paper, but the name is kept for consistency with MobileNetV1 in Keras.
        - If `alpha` < 1.0, proportionally decreases the number
            of filters in each layer.
        - If `alpha` > 1.0, proportionally increases the number
            of filters in each layer.
        - If `alpha` = 1, default number of filters from the paper
            are used at each layer.
          the mobilenetv3 alpha value
    model_type : {'Small', 'Large'}
        whether it is the small variant or the large variant
    max_n_blocks : int
        the maximum number of blocks in the backbone. It is further constrained by the actual
        maximum number of blocks that the variant can implement.
    minimalistic : bool
        In addition to large and small models this module also contains so-called minimalistic
        models, these models have the same per-layer dimensions characteristic as MobilenetV3
        however, they do not utilize any of the advanced blocks (squeeze-and-excite units,
        hard-swish, and 5x5 convolutions). While these models are less efficient on CPU, they
        are much more performant on GPU/DSP.
    mixer_params : mt.tfc.MobileNetV3MixerParams, optional
        parameters for defining the mixer block
    include_top : bool, default True
        whether to include the fully-connected layer at the top of the network. Only valid if
        `mixer_params` is not null.
    pooling : str, optional
        Optional pooling mode for feature extraction when `include_top` is False and
        `mixer_params` is not null.
        - `None` means that the output of the model will be the 4D tensor output of the last
          convolutional block.
        - `avg` means that global average pooling will be applied to the output of the last
          convolutional block, and thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    classes : int, optional
        Optional number of classes to classify images into, only to be specified if `mixer_params`
        is not null and `include_top` is True.
    dropout_rate : float
        fraction of the input units to drop on the last layer. Only to be specified if
        `mixer_params` is not null and `include_top` is True.
    classifier_activation : object
        A `str` or callable. The activation function to use on the "top" layer. Ignored unless
        `mixer_params` is not null and `include_top` is True. Set `classifier_activation=None` to
        return the logits of the "top" layer. When loading pretrained weights,
        `classifier_activation` can only be `None` or `"softmax"`.
    output_all : bool
        If True, the model returns the output tensor of every submodel other than the input layer.
        Otherwise, it returns the output tensor of the last submodel.
    name : str, optional
        model name, if any. Default to 'MobileNetV3LargeSplit' or 'MobileNetV3SmallSplit'.

    Returns
    -------
    tensorflow.keras.Model
        the output MobileNetV3 model split into 5 submodels
    """

    input_layer = MobileNetV3Input(input_shape=input_shape)
    input_block = MobileNetV3Parser(
        input_layer,
        model_type=model_type,
        minimalistic=minimalistic,
    )
    x = input_block(input_layer)
    outputs = [x]

    num_blocks = 5 if model_type == "Large" else 4
    if num_blocks > max_n_blocks:
        num_blocks = max_n_blocks
    for i in range(num_blocks):
        if model_type == "Large":
            block = MobileNetV3LargeBlock(i, x, alpha=alpha, minimalistic=minimalistic)
        else:
            block = MobileNetV3SmallBlock(i, x, alpha=alpha, minimalistic=minimalistic)
        x = block(x)
        if output_all:
            outputs.append(x)
        else:
            outputs = [x]

    if mixer_params is not None:
        if not isinstance(mixer_params, tfc.MobileNetV3MixerParams):
            raise tfc.ModelSyntaxError(
                "Argument 'mixer_params' is not an instance of "
                "mt.tfc.MobileNetV3MixerParams. Got: {}.".format(type(mixer_params))
            )

        if model_type == "Large":
            last_point_ch = 1280
        else:
            last_point_ch = 1024
        mixer_block = MobileNetV3Mixer(
            x,
            mixer_params,
            last_point_ch,
            alpha=alpha,
            model_type=model_type,
            minimalistic=minimalistic,
        )
        x = mixer_block(x)
        if output_all:
            if isinstance(x, (list, tuple)):
                outputs.extend(x)
            else:
                outputs.append(x)
        else:
            if isinstance(x, (list, tuple)):
                outputs = [x[-1]]
            else:
                outputs = [x]

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
            if output_all:
                outputs.append(x)
            else:
                outputs = [x]

    # Create model.
    if name is None:
        name = f"MobilenetV3{model_type}Split"
    model = models.Model(input_layer, outputs, name=name)

    return model
