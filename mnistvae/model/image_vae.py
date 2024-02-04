from typing import TypedDict
import jax
import jax.numpy as jnp
import equinox as eqx
import distrax as dsx

from .encoder import ImageEncoder
from .decoder import ImageDecoder


class ImageVAEOutput(TypedDict):
    """Output of the image VAE."""

    latent_distribution: dsx.Distribution
    observation_distribution: dsx.Distribution


def parametrize_gaussian(params):
    """Parametrizes a Gaussian distribution from a vector of parameters."""
    loc, scale = jnp.split(params, 2, axis=-1)
    # Ensure positive scale and clip for numerical stability
    scale = jax.nn.softplus(scale)
    scale = jnp.clip(scale, a_min=1e-5, a_max=None)
    return dsx.Normal(loc, scale)


class ImageVAE(eqx.Module):
    """A simple image VAE."""

    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, in_channels: int, n_latents: int = 2, *, key):
        latent_dist_params = 2  # Normal
        #n_obs_dist_params = 1  # Bernoulli
        n_obs_dist_params = 2  # Now we have a normal
        encoder_output_channels = n_latents * latent_dist_params
        self.encoder = ImageEncoder(
            in_channels=in_channels, out_channels=encoder_output_channels, key=key
        )

        decoder_output_channels = in_channels * n_obs_dist_params
        self.decoder = ImageDecoder(
            in_channels=n_latents, out_channels=decoder_output_channels, key=key
        )

    def encode(self, image: jnp.ndarray) -> dsx.Distribution:
        return parametrize_gaussian(self.encoder(image))

    def decode(self, latents: jnp.ndarray) -> dsx.Distribution:
        decoder_output = self.decoder(latents)

        loc, log_scale = jnp.split(decoder_output, 2)
        scale = jax.nn.softplus(log_scale)
        scale = jnp.clip(scale, a_min=1e-5, a_max=None)
        #print(f"loc shape: {loc.shape}, scale shape: {scale.shape}")  
        return dsx.Normal(loc, scale)
    
    def __call__(
        self, image: jnp.ndarray, *, key=jax.random.PRNGKey
    ) -> dsx.Distribution:
        latent_distribution = self.encode(image)
        z = latent_distribution.sample(seed=key)
        observation_distribution = self.decode(z)

        return ImageVAEOutput(
            latent_distribution=latent_distribution,
            observation_distribution=observation_distribution,
            prior=dsx.Normal(
                jnp.zeros_like(latent_distribution.loc),
                jnp.ones_like(latent_distribution.scale),
            ),
        )
