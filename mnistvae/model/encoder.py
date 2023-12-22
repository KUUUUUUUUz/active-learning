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
        keys = jax.random.split(key, num=6)
        self.layers = [
            eqx.nn.Conv2d(in_channels, 16, kernel_size=3, key=keys[0], padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(16, 16, kernel_size=3, key=keys[1], stride=2, padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(16, 32, kernel_size=3, key=keys[2], padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 32, kernel_size=3, key=keys[2], stride=2, padding=1),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(2048, 1025, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Linear(1025, 512, key=keys[4]),
            jax.nn.relu,
            eqx.nn.Linear(512, out_channels, key=keys[5]),
        ]

        # self.layers = [
        #     eqx.nn.Conv2d(in_channels, 16, kernel_size=3, key=keys[0], stride=2, padding=1),
        #     jax.nn.relu,
        #     eqx.nn.Conv2d(16, 32, kernel_size=3, key=keys[1], stride=2, padding=1),
        #     jax.nn.relu,
        #     jnp.ravel,  
        #     eqx.nn.Linear(32 * 8 * 8, out_channels, key=keys[2]) 
        # ]

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        x = image[None, ...]

        for layer in self.layers:
            x = layer(x)
            #print(x.shape)
        return x
