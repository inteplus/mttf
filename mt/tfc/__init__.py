"""The core part of mttf that can be imported without touching the Tensorflow package."""

import typing as tp
import yaml

__all__ = [
    "ModelSyntaxError",
    "ModelParams",
    "MHAParams",
    "MHAPool2DCascadeParams",
    "MobileNetV3MixerParams",
]


class ModelSyntaxError(SyntaxError):
    pass


class ModelParams(yaml.YAMLObject):
    """Parameters for defining and creating a model.

    This is an abstract class. The user should subclass from this class to define their own class
    which represents the collection of parameters to create models of a given family.

    Parameters
    ----------
    gen : int
        model generation/family number, starting from 1
    """

    yaml_tag = "!ModelParams"

    def __init__(self, gen: int = 1):
        self.gen = gen


class MHAParams(ModelParams):
    """Parameters for creating an MHA layer.

    Parameters
    ----------
    n_heads : int
        number of heads
    key_dim : int, optional
        dimensionality of each (projected) key/query vector. If not provided, it is set as the last
        dim of the query tensor integer-divided by `n_heads`.
    value_dim : int, optional
        dimensionality of each (projected) value vector. If not provided, it is set as `key_dim`.
    output_shape : object
        passed as-is to MultiHeadAttention
    gen : int
        model generation/family number, starting from 1
    """

    yaml_tag = "!MHAParams"

    def __init__(
        self,
        n_heads: int = 4,
        key_dim: tp.Optional[int] = None,
        value_dim: tp.Optional[int] = None,
        output_shape: object = None,
        gen: int = 1,
    ):
        super().__init__(gen=gen)

        self.n_heads = n_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_shape = output_shape

    def to_json(self):
        """Returns an equivalent json object."""
        return {
            "n_heads": self.n_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "output_shape": self.output_shape,
            "gen": self.gen,
        }

    @classmethod
    def from_json(cls, json_obj):
        """Instantiates from a json object."""
        return MHAParams(
            n_heads=json_obj["n_heads"],
            key_dim=json_obj.get("key_dim", None),
            value_dim=json_obj.get("value_dim", None),
            output_shape=json_obj.get("output_shape", None),
            gen=json_obj["gen"],
        )


class MHAPool2DCascadeParams(ModelParams):
    """Parameters for creating a cascade of MHAPool2D layers.

    Parameters
    ----------
    n_heads : int
        number of heads
    expansion_factor : float
        expansion factor at each layer
    pooling : {'avg', 'max'}
        pooling type
    dropout : float
        dropout probability
    gen : int
        model generation/family number, starting from 1
    """

    yaml_tag = "!MHAPool2DCascadeParams"

    def __init__(
        self,
        n_heads: int = 20,
        expansion_factor: float = 1.5,
        pooling: str = "max",
        dropout: float = 0.2,
        gen: int = 1,
    ):
        super().__init__(gen=gen)

        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.pooling = pooling

    def to_json(self):
        """Returns an equivalent json object."""
        return {
            "n_heads": self.n_heads,
            "expansion_factor": self.expansion_factor,
            "pooling": self.pooling,
            "dropout": self.dropout,
            "gen": self.gen,
        }

    @classmethod
    def from_json(cls, json_obj):
        """Instantiates from a json object."""
        return MHAParams(
            n_heads=json_obj["n_heads"],
            expansion_factor=json_obj["expansion_factor"],
            pooling=json_obj["pooling"],
            dropout=json_obj["dropout"],
            gen=json_obj["gen"],
        )


class MobileNetV3MixerParams(ModelParams):
    """Parameters for creating a MobileNetV3Mixer.

    Parameters
    ----------
    variant : {'mobilenet', 'maxpool', 'mha', 'mhapool'}
        Variant of the mixer block. The output tensor has 1x1 spatial resolution. If 'mobilenet' is
        specified, the mixer follows 'mobilenet' style, including mainly 2 Conv layers and one
        GlobalAveragePooling2D layer. If 'maxpool' is specified, grid processing is just a
        GlobalMaxPool2D layer. If 'mha' is specified, a SimpleMHA2D layer is used. If 'mhapool' is
        used, a cascade of MHAPool2D layers is used until the last layer outputs a 1x1 tensor.
    mha_params : mt.tfc.MHAParams, optional
        The parameters defining the MultiHeadAttention layer. Only valid for 'mha' mixer type.
    mhapool_cascade_params : mt.tfc.MHAPool2DCascadeParams, optional
        The parameters defining a cascade of MHAPool2D layers. Only valid for 'mhapool' mixer type.
    gen : int
        model generation/family number, starting from 1
    """

    yaml_tag = "!MobileNetV3MixerParams"

    def __init__(
        self,
        variant: str = "mobilenet",
        mha_params: tp.Optional[MHAParams] = None,
        mhapool_cascade_params: tp.Optional[MHAPool2DCascadeParams] = None,
        gen: int = 1,
    ):
        super().__init__(gen=gen)

        self.variant = variant
        self.mha_params = mha_params
        self.mhapool_cascade_params = mhapool_cascade_params

    def to_json(self):
        """Returns an equivalent json object."""
        mha_params = None if self.mha_params is None else mha_params.to_json()
        if self.mhapool_cascade_params is None:
            mhapool_params = None
        else:
            mhapool_params = self.mhapool_cascade_params.to_json()
        return {
            "variant": self.variant,
            "mha_params": mha_params,
            "mhapool_cascade_params": mhapool_params,
            "gen": self.gen,
        }

    @classmethod
    def from_json(cls, json_obj):
        """Instantiates from a json object."""
        mha_params = json_obj.get("mha_params", None)
        if mha_params is not None:
            mha_params = MHAParams.from_json(mha_params)
        mhapool_params = json_obj.get("mhapool_cascade_params", None)
        if mhapool_params is not None:
            mhapool_params = MHAPool2DCascadeParams.from_json(mhapool_params)
        return MHAParams(
            variant=json_obj["variant"],
            mha_params=mha_params,
            mhapool_cascade_params=mhapool_params,
            gen=json_obj["gen"],
        )
