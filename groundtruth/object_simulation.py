import functools
from typing import List
from typing_extensions import Self

import jax
from flax import struct
import jax.numpy as jnp

from groundtruth.floor import split_key
from groundtruth.objects import ObjectCollection
from groundtruth.scheduly_generation2 import full_pattern
from groundtruth.spoof_vmap import spoof_vmap

TIME_SCALES_MAPPING = {
    "year": 365,
    "month": 30,
    "week": 7,
    "day": 1
}

@struct.dataclass
class ScheduleDistribution:
    means: jnp.array
    stds: jnp.array

    @jax.jit
    def sample_duration(self, key, i: int):
        key, rngs = split_key(key, len(self))

        def sample(k, m, s):
            return jax.random.normal(key=k) * s + m # fixme lognormal

        samples = jax.vmap(sample)(rngs, self.means, self.stds)
        return jnp.take(samples, jnp.astype(i, int)) #samples[i]

        means, stds = self.means[i], self.stds[i]
        return jax.random.normal(key, means, stds)

    def __len__(self):
        return self.stds.shape[0]


@struct.dataclass
class Schedule:
    time_scale: float
    distribution: ScheduleDistribution
    subpatterns: Self
    is_leaf: bool = struct.field(pytree_node=False)
    current_pattern: int
    locations: jnp.array
    time_left_for_mode: float
    dt: float

    def __len__(self):
        return len(self.distribution)

    @classmethod
    def create(cls, dico, locations, dt=1/24): # default dt is one hour
        #if "distribution" not in dico:
        time_scale_name = list(dico.keys())[0]
        time_scale = TIME_SCALES_MAPPING[time_scale_name]
        description = dico[time_scale_name]

        distribution = ScheduleDistribution(description["distribution"]["means"], description["distribution"]["stds"])
        if "locations" in description:
            subpatterns = []
        else:
            subpatterns = [Schedule.create(s, locations) for s in description["subpatterns"]]
            if len(subpatterns) > 0:
                subpatterns = spoof_vmap(subpatterns)

        return cls(time_scale, distribution, subpatterns, "locations" in description, -1., jnp.array(locations), -1., dt)

    @jax.jit
    def tick(self, key):
        new_time_left = self.time_left_for_mode - self.dt

        time_over = jnp.astype(new_time_left < 0, int)

        new_current_mode = (self.current_pattern + time_over) % len(self.distribution)

        new_time_left = new_time_left * (1-time_over) + time_over * self.distribution.sample_duration(key, new_current_mode) * self.time_scale
        self = self.replace(current_pattern=new_current_mode,time_left_for_mode=new_time_left)

        if self.is_leaf:
            return self
        else:
            key, rngs = split_key(key, len(self.subpatterns))
            new_subpatterns = jax.vmap(lambda key, x: x.tick(key))(rngs, self.subpatterns)
            return self.replace(subpatterns=new_subpatterns)

    @jax.jit
    def get_current_mode(self):
        if self.is_leaf:
            return jnp.take(self.locations, jnp.astype(self.current_pattern, int))

        ret = jax.vmap(lambda x: x.get_current_mode())(self.subpatterns)

        subpattern = jnp.atleast_1d(jnp.take(ret, jnp.astype(self.current_pattern, int), axis=0).squeeze())
        return jnp.concatenate([jnp.atleast_1d(self.current_pattern), subpattern]) #ret[self.current_pattern]

    @jax.jit
    def step(self, key):
        self = self.tick(key)
        mode = self.get_current_mode()
        return self, mode

if __name__ == "__main__":


    scales = ["year", "month", "day"]
    locations = [0,1,2]

    pattern = full_pattern(
        key=jax.random.PRNGKey(0),
        scales=scales, locations=locations,
        min_time_buckets=2,
        max_time_buckets=5,
        seed=42,
        is_gaussian=True
    )
    key = jax.random.PRNGKey(0)
    schedule = Schedule.create(pattern, locations)
    schedule.tick(key).get_current_mode()


    schedule, results = jax.lax.scan(f=Schedule.step, init=schedule, xs=split_key(key, 100)[-1])
    print(results)

