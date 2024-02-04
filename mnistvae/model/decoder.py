from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx


class ImageDecoder(eqx.Module):
    """A simple image decoder."""

    layers: List[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: jax.random.PRNGKey,
    ):
        # keys = jax.random.split(key, num=7)
        # self.layers = [
        #     eqx.nn.Linear(in_channels, 128, key=keys[0]),
        #     jax.nn.relu,
        #     eqx.nn.Linear(128, 256, key=keys[1]),
        #     jax.nn.relu,
        #     eqx.nn.Linear(256, 512, key=keys[2]),
        #     jax.nn.relu,
        #     eqx.nn.Lambda(lambda x: x.reshape((32, 4, 4))),
        #     eqx.nn.ConvTranspose2d(
        #         32, 64, kernel_size=4, key=keys[3], stride=2, padding=1
        #     ),
        #     jax.nn.relu,
        #     eqx.nn.ConvTranspose2d(
        #         64, 64, kernel_size=4, key=keys[4], stride=2, padding=1
        #     ),
        #     jax.nn.relu,
        #     eqx.nn.ConvTranspose2d(
        #         64, 64, kernel_size=4, key=keys[5], stride=2, padding=1
        #     ),
        #     jax.nn.relu,
        #     eqx.nn.Conv2d(
        #         64,
        #         out_channels,
        #         kernel_size=1,
        #         key=keys[6],
        #     ),
        # ]
        keys = jax.random.split(key, num=9)  # Adjust the number of keys if the number of layers changes
        self.layers = [
            eqx.nn.Linear(in_channels, 512, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(512, 1024, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(1024, 8192, key=keys[2]),  # Matches the encoder's flattening point
            jax.nn.relu,
            eqx.nn.Lambda(lambda x: x.reshape((128, 8, 8))),  # Reshape to match the encoder's last conv output
            eqx.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[3]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[5]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, key=keys[6]),
            # No activation after the last layer, assuming the output matches the input image format directly
        ]

    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        x = latents

        for layer in self.layers:
            #print(x.shape)
            x = layer(x)

        return x