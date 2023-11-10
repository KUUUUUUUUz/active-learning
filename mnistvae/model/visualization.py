import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from model.image_vae import ImageVAEOutput


def visualize_reconstructions(
    model: eqx.Module, targets: jnp.array, show_sample: bool = False
):
    """Visualizes the reconstructions of a given model on a given batch of targets
    and shows samples from the prior. Either visualize the distribution as the
    probabilities or as a sample."""
    batch_size = targets.shape[0]
    K = min(4, batch_size)
    targets = targets[:K]
    keys = jax.random.split(jax.random.PRNGKey(0), K)
    outputs: ImageVAEOutput = jax.vmap(model)(targets, key=keys)
    latent_prior_samples = outputs["prior"].sample(seed=jax.random.PRNGKey(0))
    data_prior_samples = jax.vmap(model.decode)(latent_prior_samples)
    if show_sample:
        reconstruction = outputs["observation_distribution"].sample(
            seed=jax.random.PRNGKey(0)
        )
        data_prior_samples = data_prior_samples.sample(seed=jax.random.PRNGKey(0))
    else:
        reconstruction = outputs["observation_distribution"].probs
        data_prior_samples = data_prior_samples.probs

    plt.figure(figsize=(K * 1.8, 7))
    for i in range(K):
        plt.subplot(3, K, i + 1)
        plt.imshow(targets[i], interpolation="none")
        plt.title("Input")
        plt.axis("off")
        plt.subplot(3, K, K + i + 1)
        plt.title("Reconstruction")
        plt.imshow(reconstruction[i, 0], interpolation="none")
        plt.axis("off")

        plt.subplot(3, K, 2 * K + i + 1)
        plt.imshow(data_prior_samples[i, 0], interpolation="none")
        plt.title("Prior sample")
        plt.axis("off")
    plt.show()
