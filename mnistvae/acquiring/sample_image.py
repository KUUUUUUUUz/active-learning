from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

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



def get_next(R, mask_observed, images, original)-> jnp.ndarray:
    mask_missing = ~mask_observed
    R_masked = R * mask_missing.astype(R.dtype)
    i = jnp.argmax(R_masked)
    i_2d = jnp.unravel_index(i, (32, 32))
    i = (i_2d[0], i_2d[1])
    
    mask_observed = mask_observed.at[i].set(True)

    new_pixel_value = original[i]
    images_updated= images.at[i].set(new_pixel_value)
    is_true = images_updated[i]
    #print(f"The value at index {i} is: {is_true}")
    #print(np.sum(mask_observed))
    #return new images that reveal our new pixels
    return images_updated, mask_observed 


def update_images(images: jnp.ndarray, mask_observed: jnp.ndarray) -> jnp.ndarray:
    # Ensure that mask_observed is a boolean array
    mask_observed = mask_observed.astype(bool)

    # Element-wise multiplication to zero out unobserved pixels
    new_images = images * mask_observed
   
    return new_images


# def update_pixel(image, mask, i, j):
#     # Function to be executed if the condition is True
#     def true_fun(_):
#         return image.at[i, j].set(0)

#     # Function to be executed if the condition is False
#     def false_fun(_):
#         return image

#     # Conditional update using jax.lax.cond
#     return jax.lax.cond(mask[i, j], true_fun, false_fun, None)

# def update_images(images: jnp.ndarray, mask_observed: jnp.ndarray) -> jnp.ndarray:
#     for i in range(images.shape[0]):
#         for j in range(images.shape[1]):
#             # Use jax.lax.cond for conditional update
#             images = update_pixel(images, mask_observed, i, j)
#     return images


#def remove_half()

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



