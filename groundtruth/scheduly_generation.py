import copy
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update('jax_platform_name', "cpu")
from jax import numpy as jnp

def generate_time_pattern(key, num_states):
    """
    Generate a normalized time pattern across a given number of states.
    The times are random but sum to 1.
    """
    key, rng = jax.random.split(key)
    weights = jax.random.uniform(rng, minval=0, maxval=1, shape=(num_states,))
    weights = weights / jnp.sum(weights)
    return weights


def full_pattern(key,
    scales,
    locations,
    min_time_buckets=2,
    max_time_buckets=4,
    seed=None,
    is_gaussian=False,
):
    """
    Recursively generate a hierarchical time pattern.

    Args:
        scales (list[str]): List of scales, from largest to smallest.
            Example: ["year", "month", "day"]
        locations (list[str]): Physical locations (only at the leaf scale).
        min_time_buckets (int): Minimum number of splits for higher scales.
        max_time_buckets (int): Maximum number of splits for higher scales.
        seed (int, optional): Random seed for reproducibility.
        is_gaussian (bool): Is the distribution gaussian or deterministic?.

    Returns:
        dict: Nested dictionary representing the full pattern.
    """
    current_scale = scales[0]

    # Leaf scale: assign location distribution
    if len(scales) == 1:

        TO_DELETE = []
        for i, _ in enumerate(locations):
            if len(locations) - len(TO_DELETE) == 1:
                break
            key, rng = jax.random.split(key)
            if jax.random.uniform(rng, minval=0,maxval=1, shape=(1,)).item() > 0.75:
                TO_DELETE.append(i)

        key, rng = jax.random.split(key)
        distribution = generate_time_pattern(rng, len(locations))
        for delete in TO_DELETE:
            distribution[delete] = 0
        return {
            current_scale: {
                "distribution": distribution,
                "locations": locations
            }
        }

    # Higher scales: split into random buckets
    key, rng = jax.random.split(key)
    num_buckets = jax.random.randint(rng, minval=min_time_buckets, maxval=max_time_buckets, shape=(1,)).item() # random.randint(min_time_buckets, max_time_buckets)
    key, rng = jax.random.split(key)
    distribution = generate_time_pattern(rng, num_buckets)

    subpatterns = [
        full_pattern(
            key,
            scales[1:], locations,
            min_time_buckets, max_time_buckets,
            seed, is_gaussian
        )
        for _ in range(num_buckets)
    ]

    MISSING_BUCKETS = max_time_buckets - num_buckets
    if MISSING_BUCKETS > 0:
        distribution = jnp.concatenate([distribution, jnp.zeros((MISSING_BUCKETS,))]) # {k: jnp.concatenate([v, jnp.zeros((MISSING_BUCKETS,))]) for k,v in distribution.items()}
        for _ in range(MISSING_BUCKETS):
            subpatterns.append(copy.deepcopy(subpatterns[-1]))

    return {
        current_scale: {
            "distribution": distribution,
            "subpatterns": subpatterns
        }
    }


# Example usage
scales = ["year", "month", "day"]
locations = ["home", "work", "key rack"]

pattern = full_pattern(
    key=jax.random.PRNGKey(0),
    scales=scales, locations=locations,
    min_time_buckets=2,
    max_time_buckets=5,
    seed=42,
    is_gaussian=True
)

print(pattern)
