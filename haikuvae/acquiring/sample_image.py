from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

def sample_a_image(Nz, Nx, mean_i, log_var_i, decoder, new_decoder_params, rng) -> jnp.ndarray:
    
    random = jax.random.normal(rng, (Nz, 10))
    #Latent_space size = 10
    sampled_z_points = mean_i + log_var_i * random
    output_x = decoder.apply(new_decoder_params, rng, sampled_z_points)

    p = jax.nn.sigmoid(output_x)  # Shape: (Nz, 28, 28, 1)
    images = []
    for i in range(Nz):
        img = jax.random.bernoulli(rng, p[i], shape=(Nx, *p[i].shape))
        images.append(img.reshape(-1, *p[i].shape))
    images = jnp.concatenate(images, axis=0)
    #return R = variance of pixel. Shape: (32, 28, 28, 1)
    return jnp.var(images, axis=0)


def get_next(R, mask_observed, images, original)-> jnp.ndarray:
    mask_missing = ~mask_observed
    R_masked = R * mask_missing.astype(R.dtype)
    i = jnp.argmax(R_masked)
    i_2d = jnp.unravel_index(i, (28, 28))
    i = (i_2d[0], i_2d[1])
    mask_observed = mask_observed.at[i].set(True)
    #print(np.sum(mask_observed))
    new_pixel_value = original[i]
    images_updated= images.at[i].set(new_pixel_value)
    #return new images that reveal our new pixels
    return images_updated, mask_observed 


def update_images(images: jnp.ndarray, mask_observed: jnp.ndarray) -> jnp.ndarray:
    # Ensure that mask_observed is a boolean array
    mask_observed = mask_observed.astype(bool)

    # Element-wise multiplication to zero out unobserved pixels
    new_images = images * mask_observed

    return new_images


def random_get_next(mask_observed, images, original, rng_key):
    # Find indices of unobserved pixels
    unobserved_indices = jnp.where(~mask_observed)

    # Check if there are any unobserved indices
    if unobserved_indices[0].size == 0:
        return images, mask_observed  # Return the inputs unchanged if all pixels are observed

    # Randomly choose one index from unobserved indices
    random_idx = random.choice(rng_key, unobserved_indices[0].shape[0], shape=())
    i = (unobserved_indices[0][random_idx], unobserved_indices[1][random_idx])

    # Update the mask and images with the original pixel value
    mask_observed = mask_observed.at[i].set(True)
    new_pixel_value = original[i]
    images_updated = images.at[i].set(new_pixel_value)

    return images_updated, mask_observed








