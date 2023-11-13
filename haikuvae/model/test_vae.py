import jax

from vae import VariationalAutoEncoder, VAEOutput
MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)

def test_image_vae():
    key = jax.random.PRNGKey(0)

    model = VariationalAutoEncoder(512,10,MNIST_IMAGE_SHAPE)
    image_input_example = jax.random.normal(key, (28, 28,1))
    model_output: ImageVAEOutput = model(image_input_example, key=key)

    latent_sample = model_output["latent_distribution"].sample(seed=key)
    assert latent_sample.shape == (2,)

    observation_sample = model_output["observation_distribution"].sample(seed=key)
    assert observation_sample.shape == (28, 28, 1)
