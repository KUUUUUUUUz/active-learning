from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jax import random
MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)

class Decoder(hk.Module):
    """Decoder model."""

    def __init__(
        self,
        hidden_size: int = 512,
        output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = hk.Linear(self._hidden_size)(z)
        z = jax.nn.relu(z)

        logits = hk.Linear(np.prod(self._output_shape))(z)
        logits = jnp.reshape(logits, (-1, *self._output_shape))

        return logits
