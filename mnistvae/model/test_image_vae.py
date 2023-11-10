import jax

from model.image_vae import ImageVAE, ImageVAEOutput


def test_image_vae():
    key = jax.random.PRNGKey(0)

    model = ImageVAE(in_channels=1, n_latents=2, key=key)
    image_input_example = jax.random.normal(key, (32, 32))
    model_output: ImageVAEOutput = model(image_input_example, key=key)

    latent_sample = model_output["latent_distribution"].sample(seed=key)
    assert latent_sample.shape == (2,)

    observation_sample = model_output["observation_distribution"].sample(seed=key)
    assert observation_sample.shape == (1, 32, 32)
