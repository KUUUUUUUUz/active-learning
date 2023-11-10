import jax
from model.encoder import ImageEncoder


def test_image_encoder():
    key = jax.random.PRNGKey(0)
    encoder = ImageEncoder(in_channels=1, out_channels=2, key=key)
    image_input_example = jax.random.normal(key, (32, 32))
    encoder_output = encoder(image_input_example)

    assert encoder_output.shape == (2,)
