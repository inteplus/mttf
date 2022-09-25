"""Initialises TensorFlow, monkey-patching if necessary."""

from packaging import version

__all__ = ["init"]


def init():
    """Initialises tensorflow, monkey-patching if necessary."""

    import tensorflow
    import sys

    tf_ver = version.parse(tensorflow.__version__)

    if tf_ver < version.parse("2.5"):
        raise ImportError(
            "The minimum TF version that mttf supports is 2.5. Your TF is {}. "
            "Please upgrade.".format(tf_ver)
        )

    # add mobilenet_v3_split module
    from .keras_applications import mobilenet_v3_split, mobilevit

    setattr(tensorflow.keras.applications, "mobilenet_v3_split", mobilenet_v3_split)
    setattr(tensorflow.keras.applications, "mobilevit", mobilevit)
    sys.modules["tensorflow.keras.applications.mobilenet_v3_split"] = mobilenet_v3_split
    sys.modules["tensorflow.keras.applications.mobilevit"] = mobilevit
    setattr(
        tensorflow.keras.applications,
        "MobileNetV3Split",
        mobilenet_v3_split.MobileNetV3Split,
    )

    from .keras_layers import Identical, Upsize2D, Downsize2D

    setattr(tensorflow.keras.layers, "Identical", Identical)
    setattr(tensorflow.keras.layers, "Upsize2D", Upsize2D)
    setattr(tensorflow.keras.layers, "Downsize2D", Downsize2D)

    if tf_ver < version.parse("2.8"):
        # monkey-patch efficientnet_v2
        from .keras_applications import efficientnet_v2

        sys.modules["tensorflow.keras.applications.efficientnet_v2"] = efficientnet_v2
        if tf_ver >= version.parse("2.7"):
            sys.modules["keras.applications.efficientnet_v2"] = efficientnet_v2
            import keras

            setattr(keras.applications, "effiicientnet_v2", efficientnet_v2)
        setattr(tensorflow.keras.applications, "efficientnet_v2", efficientnet_v2)
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2B0",
            efficientnet_v2.EfficientNetV2B0,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2B1",
            efficientnet_v2.EfficientNetV2B1,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2B2",
            efficientnet_v2.EfficientNetV2B2,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2B3",
            efficientnet_v2.EfficientNetV2B3,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2S",
            efficientnet_v2.EfficientNetV2S,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2M",
            efficientnet_v2.EfficientNetV2M,
        )
        setattr(
            tensorflow.keras.applications,
            "EfficientNetV2L",
            efficientnet_v2.EfficientNetV2L,
        )

    return tensorflow


init()
