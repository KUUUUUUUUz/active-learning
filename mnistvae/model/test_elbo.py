import jax
import jax.numpy as jnp
import distrax as dsx

from .image_vae import ImageVAEOutput
from .loss import get_evidence_lower_bound, ELBOLossOutput


def test_evidence_lower_bound():
    key = jax.random.PRNGKey(0)

    # Construct a dummy model output
    target_probs = jax.random.uniform(key, (1, 32, 32))
    observation_distribution = dsx.Bernoulli(
        probs=target_probs + 0.1 * jax.random.uniform(key, (1, 32, 32))
    )
    target = dsx.Bernoulli(probs=target_probs).sample(seed=key)
    model_output = ImageVAEOutput(
        latent_distribution=dsx.Normal(jnp.zeros(2), jnp.ones(2)),
        observation_distribution=observation_distribution,
        prior=dsx.Normal(0.0, 1.0),
    )

    # Compute the loss
    loss_output = get_evidence_lower_bound(target, model_output)
    loss_value: jnp.array = loss_output[0]
    aux: ELBOLossOutput = loss_output[1]

    # Check that we get a scalar loss output
    assert loss_value.shape == ()
