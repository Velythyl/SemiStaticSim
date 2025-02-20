import itertools
import uuid

from hippo.conceptgraph_intake import load_conceptgraph
from hippo.scenedata import HippoObjectPlan, HippoRoomPlan
from hippo.spatial_utils import get_size, disambiguate, get_bounding_box
from hippo.string_utils import get_uuid

import numpy as np


def get_hippos(path, pad=1):
    cg = load_conceptgraph(path)
    cg_objects = cg["segGroups"]

    hippo_objects = []

    pcds = []
    for obj in cg_objects:
        object_name = obj["label"].lower().replace(":","").replace(",", " ").strip()

        pcd = np.asarray(obj["pcd"].points)[:,[0,2,1]]

        object_description = obj["caption"]
        clip_features = obj["clip_features"]

        pcds.append(pcd)

        hippo_objects.append(
            HippoObjectPlan(
                object_name=object_name,
                object_description=object_description,
                roomId=None, position=None, _clip_features=clip_features,
                _desired_size=None)
        )

    allpoints = np.concatenate(pcds)
    _minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)

    pad = np.ones(3) * pad
    pad[1] = 0

    def shift_above_0(item):
        return item - _minbound + pad

    pcds = [shift_above_0(pcd) for pcd in pcds]

    allpoints = np.concatenate(pcds)
    minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)
    minbound, maxbound = minbound-pad, maxbound+pad
    assert (minbound == 0).all()

    id2objs = {}
    name2objs = {}
    for hippo_object, pcd in (zip(hippo_objects,pcds)):
        position = np.median(pcd, axis=0)
        position[1] = np.min(pcd[:,1])
        print(position)
        assert (position >= (minbound+pad)).all()
        assert (position <= (maxbound-pad)).all()
        size = get_size(pcd, as_dict=False)
        hippo_object = hippo_object.replace(position=position, _desired_size=size)

        if hippo_object.object_name not in name2objs:
            name2objs[hippo_object.object_name] = []
        name2objs[hippo_object.object_name].append((hippo_object, get_bounding_box(pcd), len(pcd)))
        id2objs[hippo_object.id] = hippo_object

    def keep_delete(keep, id):
        if not keep:
            if id in id2objs:
                del id2objs[id]

    # disambiguation step
    for name, hippo_objects in name2objs.items():
        if len(hippo_objects) == 1:
            continue

        for (obj1, bbox1, num_pcd1), (obj2, bbox2, num_pcd2) in itertools.combinations(hippo_objects, 2):
            keep1, keep2 = disambiguate(bbox1, bbox2)

            if keep1 == keep2:  # either true, true or false, false
                keep_delete(keep1, obj1.id)
                keep_delete(keep2, obj2.id)

            if keep1 != keep2:
                if num_pcd2 > num_pcd1:
                    keep_delete(False, obj1.id)
                else:
                    keep_delete(False, obj2.id)


    hippo_objects = list(id2objs.values())

    minbound = (minbound[0] ,minbound[2])
    maxbound = (maxbound[0],maxbound[2])
    roomId = cg["sceneId"] + f"-{get_uuid()}"
    coords = ((minbound[0], minbound[1]), (minbound[0], maxbound[1]), (maxbound[0], maxbound[1]), (maxbound[0], minbound[1]))
    roomplan = HippoRoomPlan(id=roomId, coords=coords)

    return roomplan, hippo_objects


def _get_hippos(path, pad=1):
    cg = load_conceptgraph(path)
    cg_objects = cg["segGroups"]

    hippo_objects = []

    pcds = []

    TEMP_BBOX_KEY = "TEMP_BBOX_KEY"

    name2obj = {}
    done_objects = set()
    for obj in cg_objects:
        object_name = obj["label"].lower().replace(":","").replace(",", " ").strip()

        pcd = np.asarray(obj["pcd"].points)[:,[0,2,1]]

        bbox = get_size(pcd)

        #if object_name not in name2obj:
        #    name2obj[object_name] = []
        #else:
        #    for obj in name2obj[object_name]:
        #        keep_self, keep_other = disambiguate(bbox, obj[TEMP_BBOX_KEY])

        #if object_name in done_objects:
        #    disambiguate(bbox,)
        #    continue
        done_objects.add(object_name)

        object_description = obj["caption"]
        #positions.append(obj["centroid"])
        #x,y,z = obj["centroid"]
        #position = [x+4, z, y+4]
        clip_features = obj["clip_features"]

        pcds.append(pcd)
        #bbox = get_bounding_box(obj["pcd"], as_dict=False)

        #print(np.median(np.asarray(obj["pcd"].points),axis=0) * 100)
        #print(np.array(obj["centroid"]) * 100)

        hippo_objects.append(
            HippoObjectPlan(
                object_name=object_name,
                object_description=object_description,
                roomId=None, position=None, _clip_features=clip_features,
                _desired_size=None)
        )

    allpoints = np.concatenate(pcds)
    _minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)

    pad = np.ones(3) * pad
    pad[1] = 0

    def shift_above_0(item):
        return item - _minbound + pad

    pcds = [shift_above_0(pcd) for pcd in pcds]

    allpoints = np.concatenate(pcds)
    minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)
    minbound, maxbound = minbound-pad, maxbound+pad
    assert (minbound == 0).all()

    for i, pcd in enumerate(pcds):
        position = np.median(pcd, axis=0)
        position[1] = np.min(pcd[:,1])
        print(position)
        assert (position >= (minbound+pad)).all()
        assert (position <= (maxbound-pad)).all()
        size = get_size(pcd, as_dict=False)
        hippo_objects[i] = hippo_objects[i].replace(position=position, _desired_size=size)

    minbound = (minbound[0] ,minbound[2])
    maxbound = (maxbound[0],maxbound[2])
    roomId = cg["sceneId"] + f"-{get_uuid()}"
    coords = ((minbound[0], minbound[1]), (minbound[0], maxbound[1]), (maxbound[0], maxbound[1]), (maxbound[0], minbound[1]))
    roomplan = HippoRoomPlan(id=roomId, coords=coords)

    return roomplan, hippo_objects

if __name__ == "__main__":
    get_hippoplans("./rgbd_interactions_2_l14")
