import copy
import functools
from dataclasses import field, dataclass
from typing import List, Dict, Any

import jax
import numpy as np
from typing_extensions import Self

import jax.numpy as jnp


from hippo.simulation.spatialutils.proximity_spatial_funcs import isOnTop, isInside, isBeside, distance
from hippo.reconstruction.scenedata import dict2xyztuple, HippoObject, _Hippo, xyztuple_precision
from hippo.utils.git_diff import git_diff


#from hippo.hippocontainers.skills import *

@dataclass
class RuntimeObject(_Hippo):
    #object_name: str
    #object_description: str
    id: str

    position: jnp.ndarray
    rotation: jnp.ndarray
    size: jnp.ndarray

    skill_portfolio: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    heldBy: str = None


    def change_posrotsize(self, position, rotation, size) -> Self:
        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)

        if isinstance(size, dict):
            size = dict2xyztuple(size)

        position = xyztuple_precision(position)
        rotation = xyztuple_precision(rotation)
        size = xyztuple_precision(size)

        position = jnp.array(position)
        rotation = jnp.array(rotation)
        size = jnp.array(size)
        return self.replace(position=position, rotation=rotation, size=size)

    def cast_precision(self):
        return self.replace(
            position=jnp.round(self.position, 3),
            rotation=jnp.round(self.rotation, 3),
            size=jnp.round(self.size, 3)
        )
    
    def as_llmjson(self):
        dico = self.asdict()

        import numpy as np
        def arr2tup(key):
            arr = dico[key]
            arr = np.array(arr)
            arr = arr.tolist()
            dico[key] = (float(arr[0]), float(arr[1]), float(arr[2]))
        arr2tup("position"); arr2tup("rotation"); arr2tup("size")

        del dico["skill_portfolio"]
        dico.update(self.skill_portfolio.output_dict())

        dico.update(dico["metadata"])
        del dico["metadata"]
        del dico["size"] # todo do we need to keep this ?
        del dico["rotation"] # todo do we need to keep this ?
        return dico

    @classmethod
    def from_hippoobject(cls, hippoobject):
        dico = hippoobject.as_holodeckdict()

        #skill_metadata = dico["_skill_metadata"]
        #for k in ["breakable", "toggleable", "sliceable"]:
        #    assert k in skill_metadata

        #for k in ["isBroken", "isToggled", "isSliced"]:
        #    if k not in skill_metadata:
        #        skill_metadata[k] = False

        #dico.update(skill_metadata)
        return cls.fromdict(dico)

    @classmethod
    def fromdict(cls, dico):
        #object_name = dico["object_name"]
        #object_description = dico["object_description"]
        id = dico["id"]
        position = dico["position"]
        rotation = dico["rotation"]
        size = dico["size"]

        if isinstance(position, dict):
            position = dict2xyztuple(position)

        if isinstance(rotation, dict):
            rotation = dict2xyztuple(rotation)

        if isinstance(size, dict):
            size = dict2xyztuple(size)

        position = jnp.array(position)
        rotation = jnp.array(rotation)
        size = jnp.array(size)

        if "_skill_metadata" not in dico:
            raise NotImplementedError()

        from hippo.simulation.skillsandconditions.skills_abstract import SkillPortfolio
        skill_portfolio = SkillPortfolio.create(dico["_skill_metadata"])

        self = cls(
            id=id,
            position=position,
            rotation=rotation,
            size=size,
            heldBy=dico["heldBy"] if "heldBy" in dico else None,
            skill_portfolio=skill_portfolio
        )

        return self.cast_precision()


@dataclass
class RuntimeObjectContainer(_Hippo):
    object_names: List[str]
    objects_map: Dict[str, RuntimeObject]

    obj_isOnTopOf: jnp.ndarray

    @property
    def obj_hasOnTopOf(self):
        return self.obj_isOnTopOf.T

    obj_isInsideOf: jnp.ndarray

    @property
    def obj_hasInsideOf(self):
        return self.obj_isInsideOf.T

    obj_isBesideOf: jnp.ndarray # symmetric

    @property
    def obj_hasBesideOf(self):
        return self.obj_isBesideOf  # symmetric

    obj_distances: jnp.ndarray

    @classmethod
    def coerce_ai2thor_metadata_objects(cls, metadata_objects):
        ret = []
        for o in metadata_objects:
            # removes
            if o["assetId"] == "":
                continue


            # cleans
            if "objectId" in o and o["objectId"]:
                o["id"] = o["objectId"]
            else:
                o["id"] = o["name"]

            o["size"] = o["axisAlignedBoundingBox"]["size"]

            ret.append(o)

        return ret


    @classmethod
    def create(cls, objects: List[RuntimeObject] | List[Dict] | List[HippoObject], is_ai2thor_metadata=False):
        if is_ai2thor_metadata:
            objects = cls.coerce_ai2thor_metadata_objects(objects)

        object_names = []
        objects_map = {}
        for o in objects:

             if isinstance(o, dict):
                 o = RuntimeObject.fromdict(o)
             if isinstance(o, HippoObject):
                 o = RuntimeObject.from_hippoobject(o)

             objects_map[o.id] = o
             object_names.append(o.id)


        proto = jnp.zeros((len(objects), len(objects)), dtype=bool)

        self = cls(
            object_names,
            objects_map,
            obj_isOnTopOf=proto,
            obj_isInsideOf=proto,
            obj_isBesideOf=proto,
            obj_distances=proto.astype(jnp.float32),
        )

        return self.resolve_spatial_attributes()

    def resolve_spatial_attributes(self) -> Self:
        positions = [self.objects_map[o].position for o in self.object_names]
        positions = jnp.stack(positions)

        sizes = [self.objects_map[o].size for o in self.object_names]
        sizes = jnp.stack(sizes)

        kwargs = resolve_spatial_attributes(positions, sizes)

        return self.replace(**kwargs)


    def bool_spatial_attribute_to_list(self, j, attrvec):
        attrvec = np.array(attrvec).squeeze()
        attrvec[j] = False
        valid_indices = np.arange(len(attrvec))[attrvec]

        ret = []
        for valid_index in valid_indices:
            ret.append(self.object_names[valid_index])
        return ret

    def float_spatial_attribute_to_dict(self, j, attrvec):
        attrvec = np.array(attrvec).squeeze()
        attrvec[j] = False
        valid_indices = np.arange(len(attrvec))[attrvec > 0]

        ret = {}
        for valid_index in valid_indices:
            ret[self.object_names[valid_index]] = float(attrvec[valid_index])
        return ret

    def update_from_ai2thor(self, objects: List[Dict]):
        objects = RuntimeObjectContainer.coerce_ai2thor_metadata_objects(objects)

        NUM_UPDATED_OBJECTS = 0

        dico = {}
        for object in objects:
            id = object["id"]

            if id not in self.objects_map:
                continue

            existing_object = self.objects_map[id]
            existing_object = existing_object.change_posrotsize(object["position"], object["rotation"], object["size"])

            heldBy = object.get("heldBy", None)
            if object["isPickedUp"]:
                assert heldBy is not None, f"object {id} is picked up but has no heldBy"

            existing_object = existing_object.replace(heldBy=heldBy)

            dico[id] = existing_object

            NUM_UPDATED_OBJECTS += 1
        assert NUM_UPDATED_OBJECTS == len(self.objects_map)
        return self.replace(objects_map=dico).resolve_spatial_attributes()

    def diff(self, new_runtimecontainer: Self):
        selfdict = self.as_llmjson()
        newdict = new_runtimecontainer.as_llmjson()

        return git_diff(selfdict, newdict)

    def get_object_by_id(self, object_id):
        return self.objects_map[object_id]

    def update_object(self, object) -> Self:
        dico = copy.deepcopy(self.objects_map)
        dico[object.id] = object
        return self.replace(objects_map=dico)
    
    def as_llmjson(self):

        obj_isOnTopOf = np.array(self.obj_isOnTopOf)
        obj_isInsideOf = np.array(self.obj_isInsideOf)
        obj_isBesideOf = np.array(self.obj_isBesideOf)
        obj_distances = np.array(self.obj_distances)

        final_dict = {}
        for i, name in enumerate(self.object_names):
            objdict = self.objects_map[name].as_llmjson()

            objdict["isOnTopOf"] = self.bool_spatial_attribute_to_list(i, obj_isOnTopOf[i])
            objdict["isInsideOf"] = self.bool_spatial_attribute_to_list(i, obj_isInsideOf[i])
            objdict["isBesideOf"] = self.bool_spatial_attribute_to_list(i, obj_isBesideOf[i])
            objdict["object_distances"] = self.float_spatial_attribute_to_dict(i, obj_distances[i])

            final_dict[name] = objdict

        return final_dict


@jax.jit
def swap_yz(mat):
    Z = mat[:,-1]
    return mat.at[:,-1].set(mat[:,-2]).at[:,-2].set(Z)

@jax.jit
def resolve_spatial_attributes(positions, sizes):

    def for_each_object(positions, sizes, p1,s1):
        def each_other_object(p1, s1, p2,s2):
            return isOnTop(p1,s1,p2,s2), isInside(p1,s1,p2,s2), isBeside(p1,s1,p2,s2), distance(p1,s1,p2,s2)
        return jax.vmap(functools.partial(each_other_object, p1, s1))(positions, sizes)

    positions = swap_yz(positions)
    sizes = swap_yz(sizes)

    obj_isOnTopOf, obj_isInsideOf, obj_isBesideOf, obj_distances = jax.vmap(functools.partial(for_each_object, positions, sizes))(positions, sizes)

    obj_distances = jax.vmap(lambda x: jnp.round(x, decimals=2))(obj_distances.squeeze())

    return {
        "obj_isOnTopOf": obj_isOnTopOf.squeeze(),
        "obj_isInsideOf": obj_isInsideOf.squeeze(),
        "obj_isBesideOf": obj_isBesideOf.squeeze(),
        "obj_distances": obj_distances.squeeze(),
    }



