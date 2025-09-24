import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update('jax_platform_name', "cpu")

import time

from PIL import Image
from ai2thor.controller import Controller
import prior

from semistaticsim.ai2thor_hippo_controller import get_sim

dataset = prior.load_dataset("procthor-10k")
dataset

house = dataset["train"][0]
type(house), house.keys(), house
Controller(scene=house)
tmp = get_sim(house)


time.sleep(100)