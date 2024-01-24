from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from functools import partial
from jax import jit

@partial(jit, static_argnums=1)
def random_get_next(mask_observed, images, original, rng_key):
    max_unobserved =32

    unobserved_indices = jnp.where(~mask_observed, size=max_unobserved)
    #print(f"[0]:{unobserved_indices[0].shape}, [1]:{unobserved_indices[1].shape}")
    # Check if there are any unobserved indices
    if unobserved_indices[0].size == 0 and unobserved_indices[1].size == 0:
        return images, mask_observed
    num_unobserved = unobserved_indices[0].size
    random_idx = random.randint(rng_key, shape=(), minval=0, maxval=num_unobserved)
    i = (unobserved_indices[0][random_idx], unobserved_indices[1][random_idx])


   # random_idx = random.choice(rng_key, unobserved_indices[0].shape[0], shape=())

    #i = (unobserved_indices[0][random_idx], unobserved_indices[1][random_idx])

    # Update the mask and images with the original pixel value
    mask_observed = mask_observed.at[i].set(True)
    new_pixel_value = original[i]
    images_updated = images.at[i].set(new_pixel_value)
    return images_updated, mask_observed
    #return images_updated, mask_observed, jnp.array(i)