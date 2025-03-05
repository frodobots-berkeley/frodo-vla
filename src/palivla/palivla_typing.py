import typing

import chex
from flax.typing import Collection, VariableDict

Array = chex.Array
ArrayTree = typing.Union[chex.Array, typing.Mapping[str, "ArrayTree"], typing.Sequence["ArrayTree"]]
Params = Collection
Variables = VariableDict
Updates = ArrayTree
Data = ArrayTree
Info = typing.Dict[str, typing.Any]
