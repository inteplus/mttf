"""Standard classifier from a feature vector."""

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
            logg.info(f"Using dropout {dropout}.", logger=logger)
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
