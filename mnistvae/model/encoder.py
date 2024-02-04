from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx


class ImageEncoder(eqx.Module):
    """A simple image encoder."""

    layers: List[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        # keys = jax.random.split(key, num=6)
        # self.layers = [
        #     eqx.nn.Conv2d(in_channels, 16, kernel_size=3, key=keys[0], padding=1),
        #     jax.nn.relu,
        #     eqx.nn.Conv2d(16, 16, kernel_size=3, key=keys[1], stride=2, padding=1),
        #     jax.nn.relu,
        #     eqx.nn.Conv2d(16, 32, kernel_size=3, key=keys[2], padding=1),
        #     jax.nn.relu,
        #     eqx.nn.Conv2d(32, 32, kernel_size=3, key=keys[2], stride=2, padding=1),
        #     jax.nn.relu,
        #     jnp.ravel,
        #     eqx.nn.Linear(2048, 1025, key=keys[3]),
        #     jax.nn.relu,
        #     eqx.nn.Linear(1025, 512, key=keys[4]),
        #     jax.nn.relu,
        #     eqx.nn.Linear(512, out_channels, key=keys[5]),
        # ]
        keys = jax.random.split(key, num=9) 
        self.layers = [
            eqx.nn.Conv2d(in_channels, 16, kernel_size=3, key=keys[0], padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(16, 32, kernel_size=3, key=keys[1], stride=2, padding=1),
            jax.nn.relu,
            
            eqx.nn.Conv2d(32, 64, kernel_size=3, key=keys[2], stride=2, padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 128, kernel_size=3, key=keys[3], stride=2, padding=1),
            jax.nn.relu,

            
            eqx.nn.Conv2d(128, 128, kernel_size=3, key=keys[4], stride=2, padding=1),
            jax.nn.relu,

            jnp.ravel,
            #  8x8x128
            eqx.nn.Linear(8192, 1024, key=keys[5]), 
            jax.nn.relu,
            eqx.nn.Linear(1024, 512, key=keys[6]),
            jax.nn.relu,
            eqx.nn.Linear(512, out_channels, key=keys[7]),
        ]

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        #x = image[None, ...]
        x = image
        for layer in self.layers:
            #print(x.shape)
            x = layer(x)
        return x