import json
import os
from typing import List
from tqdm import tqdm

from flax import struct
import jax.numpy as jnp
import shapely
from functools import singledispatchmethod

import matplotlib.pyplot as plt
from groundtruth.floor import FloorPolygon
from semistaticsim.utils.dict2xyztup import dict2xyztuple

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update('jax_platform_name', "cpu")
import prior

from ai2thor.controller import Controller

AFFORDANCE_NAMES = [
    "receptacle",
    "toggleable",
    "breakable",
    "canFillWithLiquid",
    "dirtyable",
    "canBeUsedUp",
    "cookable",
    "sliceable",
    "openable",
    "pickupable",
    "moveable"
]

def str2int(str):
    # Convert string to bytes
    b = str.encode("utf-8")  # or "ascii" if you know it's ASCII

    # Convert bytes to integer
    i = int.from_bytes(b, byteorder="big")  # use "little" for little-endian
    return i

def int2str(i):
    b_back = i.to_bytes((i.bit_length() + 7) // 8, byteorder="big")
    s_back = b_back.decode("utf-8")
    return s_back


@struct.dataclass
class ProcThorObject:
    id: str = struct.field(pytree_node=False)  # str bytes
    affordances: jnp.ndarray
    position: jnp.ndarray
    rotation: jnp.ndarray
    aabb: dict
    oobb: dict
    spawn_coordinates_above_receptacle: jnp.ndarray

    @property
    def is_pickupable(self):
        return self.affordances[AFFORDANCE_NAMES.index("pickupable")]

    @property
    def is_receptacle(self):
        return self.affordances[AFFORDANCE_NAMES.index("receptacle")]

    @singledispatchmethod
    @classmethod
    def create(cls, arg):
        raise NotImplementedError()

    @create.register(dict)
    @classmethod
    def _(cls, dict):
        id = dict["id"]
        position = dict["position"]
        rotation = dict["rotation"]
        aabb = dict["aabb"]
        oobb = dict["oobb"]

        if oobb is None:
            oobb = aabb["cornerPoints"]

        affordances = []
        for name in AFFORDANCE_NAMES:
            val = dict["affordances"][name]
            affordances.append(bool(val))
        affordances = jnp.array(affordances)

        position = jnp.array(dict2xyztuple(position))
        rotation = jnp.array(dict2xyztuple(rotation))

        spawn_coordinates_above_receptacle = jnp.array(dict["spawn_coordinates_above_receptacle"])

        return cls(id, affordances, position, rotation, aabb, oobb, spawn_coordinates_above_receptacle)

    @create.register(list)
    @classmethod
    def _(cls, list):
        objects = [ProcThorObject.create(dico) for dico in list]

        leaves_proto, tree_def = jax.tree.flatten(objects[0])
        leaves_acc = []
        for leaf in leaves_proto:
            if isinstance(leaf, jnp.ndarray):
                pass
            else:
                leaf = jnp.array([leaf])
            leaves_acc.append(leaf[None])

        for i, o in enumerate(objects[1:]):
            leaves, tree_def = jax.tree.flatten(o)
            if leaves_acc is None:
                leaves_acc = leaves
            else:
                tmp = []
                for leaf_acc, leaf in zip(leaves_acc,leaves):
                    if isinstance(leaf, jnp.ndarray):
                        pass
                    else:
                        leaf = jnp.array([leaf])
                    tmp.append(jnp.concatenate([leaf_acc, leaf[None]]))
                leaves_acc = tmp
        ret = jax.tree.unflatten(tree_def, leaves_acc)
        return ret

    @jax.jit
    def sample_from_surface(self, key, num_samples=1):
        return jax.random.choice(key, self.spawn_coordinates_above_receptacle, shape=(num_samples,), replace=False)

@struct.dataclass
class ObjectCollection:
    surface_receptacles: List[ProcThorObject]
    openable_receptacles: List[ProcThorObject]
    pickupables: List[ProcThorObject]
    #moveables: List[ProcThorObject]
    #rest_of_objects: List[ProcThorObject]

    @classmethod
    def create(cls, path, do_surface_receptacles_only=True):
        if not do_surface_receptacles_only:
            raise NotImplementedError()

        with open(f"{path}/receptacles.json", "r") as f:
            receptacles = json.load(f)
        surface_receptacles = [r for r in receptacles if not r["affordances"]["openable"]]
        openable_receptacles = [r for r in receptacles if r["affordances"]["openable"]]

        with open(f"{path}/all_objects.json", "r") as f:
            all_objects = json.load(f)
        #del receptacles

        with open(f"{path}/pickupables.json", "r") as f:
            pickupables = json.load(f)

        with open(f"{path}/movables.json", "r") as f:
            moveables = json.load(f)

        if do_surface_receptacles_only:
            for p in pickupables:
                for i, r in enumerate(p["valid_receptacles"]):
                    if receptacles[r]["affordances"]["openable"]:
                        p["valid_receptacles"][i] = -1

        surface_instances = ProcThorObject.create(surface_receptacles)
        openable_instances = ProcThorObject.create(openable_receptacles)
        pickupable_instances = ProcThorObject.create(pickupables)
        # moveable_instances = ProcThorObject.create(moveables) fixme

        return cls(surface_instances, openable_instances, pickupable_instances)#, moveable_instances)



def write_object_affordances_for_scene(path, scene, receptacle_mappings, pickupable_mappings):
    c = Controller(scene=scene)

    objects = c.last_event.metadata["objects"]
    receptacles = []
    pickupables = []
    movables = []
    uninterestings = []
    for i, object in enumerate(objects):

        BREAK = False
        for exclude in ["room", "door", "wall", "floor"]:
            if exclude in object["name"]:
                BREAK = True
                break
        if BREAK:
            break


        id = object["objectId"]

        affordances = {affordance: object[affordance] for affordance in AFFORDANCE_NAMES}
        position = object["position"]
        rotation = object["rotation"]
        aabb = object["axisAlignedBoundingBox"]
        oobb = object["objectOrientedBoundingBox"]

        if affordances["receptacle"]:
            spawnCoordinatesAboveReceptacle = c.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=id, anywhere=True).metadata["actionReturn"]
            spawnCoordinatesAboveReceptacle = [dict2xyztuple(p) for p in spawnCoordinatesAboveReceptacle]
        else:
            spawnCoordinatesAboveReceptacle = []

        ret = {
            "id": id,
            "position": position,
            "rotation": rotation,
            "aabb": aabb,
            "affordances": affordances,
            "oobb": oobb,
            "spawn_coordinates_above_receptacle": spawnCoordinatesAboveReceptacle
        }

        if id.split("|")[0] in receptacle_mappings:
            receptacles.append(ret)
        elif id.split("|")[0] in pickupable_mappings and not ret["affordances"]["receptacle"]:
            pickupables.append(ret)
        elif affordances["moveable"]:
            movables.append(ret)
        else:
            uninterestings.append(ret)
    c.stop()

    def find_receptacles_for_pickupable(clean_p_id, receptacle_list, mapping):
        ret = []
        for receptacle_index, receptacle in enumerate(receptacle_list):
            clean_receptacle_id = receptacle["id"].split("|")[0]
            if clean_receptacle_id in mapping[clean_p_id]:
                ret.append(receptacle_index)
        MISSING = len(receptacles) - len(ret)
        ret += [-1] * MISSING
        return ret

    for p in pickupables:
        p_id = p["id"].split("|")[0]
        p["valid_receptacles"] = find_receptacles_for_pickupable(p_id, receptacles, pickupable_mappings)

    for r in receptacles:
        r_id = r["id"].split("|")[0]
        p["valid_pickupables"] = find_receptacles_for_pickupable(r_id, pickupables, receptacle_mappings)

    def set_empty_receptacles(o):
        if isinstance(o, list):
            return [set_empty_receptacles(o) for o in o]
        o["valid_receptacles"] = []

    def set_empty_pickupables(o):
        if isinstance(o, list):
            return [set_empty_pickupables(o) for o in o]
        o["valid_pickupables"] = []

    set_empty_receptacles(receptacles)
    set_empty_pickupables(pickupables)
    set_empty_receptacles(uninterestings)
    set_empty_pickupables(uninterestings)
    set_empty_pickupables(movables)
    set_empty_receptacles(movables)

    os.makedirs(path, exist_ok=True)
    def save_objects(list,name):
        with open(path + f"/{name}.json", "w") as f:
            json.dump(list, f)

    save_objects(pickupables, "pickupables")
    save_objects(receptacles, "receptacles")
    save_objects(uninterestings, "uninterestings")
    save_objects(movables, "movables")
    save_objects(pickupables + receptacles + uninterestings + movables, "all_objects")

def write_object_affordances_for_procthor(path):
    dataset = prior.load_dataset("procthor-10k")
    dataset = {
        "test": dataset.test,
        "train": dataset.train,
        "val": dataset.val,
    }

    with open("/".join(__file__.split("/")[:-1]) + "/receptacles.json", "r") as f:
        receptacle_mappings = json.load(f)
    pickupable_mappings = {}
    for r, ps in receptacle_mappings.items():
        for p, desc in ps.items():
            if p not in pickupable_mappings:
                pickupable_mappings[p] = {}
            pickupable_mappings[p][r] = desc

    for split, data in dataset.items():
        for i, scene in enumerate(tqdm(data)):
            write_object_affordances_for_scene(path + f"/object_data/{split}/{i}", scene, receptacle_mappings, pickupable_mappings)
            return

if __name__ == "__main__":
    curdir = "/".join(__file__.split("/")[:-1])
    #write_object_affordances_for_procthor(curdir)

    ObjectCollection.create(curdir + "/object_data/test/0")
    exit()

    with open(curdir + "/object_data/test/0/receptacles.json", "r") as f:
        objects = json.load(f)
    tmp = ProcThorObject.create(objects)
    o = ProcThorObject.create(objects[2])
    print(o.sample_from_surface(jax.random.PRNGKey(3)))