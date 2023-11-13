from typing import Iterator, Mapping, NamedTuple, Sequence, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

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


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

    Args:
        mean: mean vector of the first distribution
        var: diagonal vector of covariance matrix of the first distribution

    Returns:
        A scalar representing KL divergence of the two Gaussian distributions.
    """
    #return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)
    return -0.5 * jnp.sum(1. + jnp.log(var) - mean**2. - var, axis=1)

@jax.jit
def loss_fn(params: hk.Params, rng_key,batch) -> jnp.ndarray:
  """ELBO: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs: VAEOutput = model.apply(params, rng_key, batch["image"])
  print("batch imgL",batch["image"])
  log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  elbo = log_likelihood - kl

  return -jnp.mean(elbo)

def loss_fn2(params: hk.Params, rng_key,batch) -> jnp.ndarray:
  outputs: VAEOutput = model.apply(params, rng_key, batch["image"])
  log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  elbo = log_likelihood - kl

  return jnp.mean(log_likelihood), jnp.mean(kl)
