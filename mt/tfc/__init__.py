"""The core part of mttf that can be imported without touching the Tensorflow package."""

import typing as tp
import yaml

__all__ = ["ModelSyntaxError", "ModelParams", "MHAParams"]


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
            "n_head": self.n_heads,
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
