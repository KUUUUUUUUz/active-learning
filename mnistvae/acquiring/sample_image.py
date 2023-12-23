from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
#from random_baseline import random_get_next

latent_size = 2

def sample_a_image(Nz, Nx,model_output, image_vae, rng) -> jnp.ndarray:
    #model_output: ImageVAEOutput = image_vae(image, key=rng)
    sampled_z_points = model_output["latent_distribution"].sample(seed=rng, sample_shape=(Nz,))

    output_images = []

    for i in range(Nz):
        observation_distribution = image_vae.decode(sampled_z_points[i])
        #print(observation_distribution.shape)
        output = observation_distribution.sample(seed=rng, sample_shape=(Nx,))
        #print(output.shape)
        output_images.append(output)
    #print output_images.shapex

    output_images = jnp.concatenate(output_images, axis=0)

    return jnp.var(output_images, axis =0) 

"""ORIGINAL = reconsturced picture """

def get_next(R, mask_observed, images, original)-> jnp.ndarray:
    mask_missing = ~mask_observed
    R_masked = R * mask_missing.astype(R.dtype)
    i = jnp.argmax(R_masked)

    i_2d = jnp.unravel_index(i, (32, 32))
    i = (i_2d[0], i_2d[1])

    mask_observed = mask_observed.at[i].set(True)

    new_pixel_value = original[i]
    images_updated= images.at[i].set(new_pixel_value)

    return images_updated, mask_observed


def update_images(images: jnp.ndarray, mask_observed: jnp.ndarray) -> jnp.ndarray:

    mask_observed = mask_observed.astype(bool)

    # Element-wise multiplication to zero out unobserved pixels
    new_images = images * mask_observed
    return new_images


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """Calculate binary (logistic) cross-entropy from distribution logits.

    Args:
        x: input variable tensor, must be of same shape as logits
        logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))

    Returns:
        A scalar representing binary CE for the given Bernoulli distribution.
    """
   # print("x shape is:", x[0])
    if x.shape != logits.shape:
        raise ValueError("inputs x and logits must be of the same shape")
    x = jnp.reshape(x, (x.shape[0], -1))
    #print("x reshape ? is:", x[0])
    logits = jnp.reshape(logits, (logits.shape[0], -1))
   # print("logits is:", logits)
    return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)
    #return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits), axis=(1,2,3))



