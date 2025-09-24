import functools
import os


os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update('jax_platform_name', "cpu")

import jax
import jax.numpy as jnp


def generate_time_pattern(key, num_states):
    """
    Generate a normalized time pattern across a given number of states.
    Uses jax.random for reproducibility.
    """
    weights = jax.random.uniform(key, (num_states,))
    return weights / jnp.sum(weights)

@functools.partial(jax.jit, static_argnums=(1,3,4))
def full_pattern(
    key,
    scales,
    locations,
    min_time_buckets=2,
    max_time_buckets=4,
):
    """
    Recursively generate a hierarchical time pattern using JAX.
    Returns nested dicts with distributions.
    """

    current_scale = scales[0]

    # Leaf scale: assign location distribution
    if len(scales) == 1:
        key, subkey = jax.random.split(key)
        distribution = generate_time_pattern(subkey, len(locations))
        return {
            current_scale: {
                "distribution": distribution,
                "locations": jnp.array(locations)  # keep as array for jit-compatibility
            }
        }

    # Higher scales: split into random buckets
    key, subkey1, subkey2 = jax.random.split(key, 3)
    num_buckets = jax.random.randint(subkey1, (), min_time_buckets, max_time_buckets + 1)
    distribution = generate_time_pattern(subkey2, num_buckets)

    # Generate subpatterns for each bucket
    subkeys = jax.random.split(key, num_buckets + 1)
    subpatterns = [
        full_pattern(
            subkeys[i],
            scales[1:],
            locations,
            min_time_buckets,
            max_time_buckets,
        )
        for i in range(num_buckets)
    ]

    return {
        current_scale: {
            "distribution": distribution,
            "subpatterns": subpatterns
        }
    }


# Example usage
scales = tuple(["year", "month", "day"])
locations = jnp.array([0, 1, 2])

key = jax.random.PRNGKey(42)
pattern = full_pattern(
    key,
    scales,
    locations,
    min_time_buckets=2,
    max_time_buckets=5,
)

print(pattern)
