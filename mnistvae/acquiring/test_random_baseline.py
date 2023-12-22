import jax
from acquiring.random_baseline import random_get_next

random_get_next_fn = jax.vmap(random_get_next)

key = jax.random.PRNGKey(0)
image_input_example = jax.random.normal(key, (128, 32, 32))
mask_input_example = jax.random.normal(key, (128, 32, 32))
keys_example = jax.random.normal(key, (128, 2))

def random_update(original_random:jnp.array, images: jnp.ndarray, mask_observed: jnp.ndarray, rng):
    images_updated, mask_return = random_get_next_fn(mask_observed, images, original_random, rng)
    return images_updated, mask_return


def test_random():
    img, mask = random_update(image_input_example, image_input_example,mask_input_example, keys_example)
    assert img.shape == (128,32,32)