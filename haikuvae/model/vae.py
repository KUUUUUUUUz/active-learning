from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .encoder import Encoder
from .decoder import Decoder

from jax import random
MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)

class VAEOutput(NamedTuple):
    image: jnp.ndarray
    mean: jnp.ndarray
    stddev: jnp.ndarray
    logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 10,
        output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size
        self._output_shape = output_shape

    def __call__(self, x: jnp.ndarray) -> VAEOutput:
        x = x.astype(jnp.float32)
        mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
        logits = Decoder(self._hidden_size, self._output_shape)(z)

        p = jax.nn.sigmoid(logits)
        image = jax.random.bernoulli(hk.next_rng_key(), p)
        #print("recon_is:",logits)
        return VAEOutput(image, mean, stddev, logits)