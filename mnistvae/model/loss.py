from typing import TypedDict
import jax
import jax.numpy as jnp
from model.image_vae import ImageVAEOutput


class ELBOLossOutput(TypedDict):
    """Output of the ELBO loss function."""

    loss: jnp.array
    distortion: jnp.array
    rate: jnp.array


def get_evidence_lower_bound(
    target: jnp.array,
    model_output: ImageVAEOutput,
) -> ELBOLossOutput:
    """Computes the evidence lower bound (ELBO) for a given target and model output."""
    distortion = -model_output["observation_distribution"].log_prob(target).sum()
    rate = (
        model_output["latent_distribution"].kl_divergence(model_output["prior"]).sum()
    )
    loss = distortion + rate
    normalization = jnp.prod(target.size)
    loss /= normalization
    rate /= normalization
    distortion /= normalization
    return ELBOLossOutput(loss=loss, distortion=distortion, rate=rate)
