from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
class Encoder(hk.Module):
    """Encoder model."""

    def __init__(self, hidden_size: int = 512, latent_size: int = 10):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = hk.Flatten()(x)
        x = hk.Linear(self._hidden_size)(x)
        x = jax.nn.relu(x)

        mean = hk.Linear(self._latent_size)(x)
        log_stddev = hk.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev

