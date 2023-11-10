import jax
from model.decoder import ImageDecoder


def test_image_decoder():
    key = jax.random.PRNGKey(0)
    encoder_output_example = jax.random.normal(key, (2,))
    decoder = ImageDecoder(in_channels=2, out_channels=1, key=key)
    decoder_output = decoder(encoder_output_example)

    assert decoder_output.shape == (1, 32, 32)
