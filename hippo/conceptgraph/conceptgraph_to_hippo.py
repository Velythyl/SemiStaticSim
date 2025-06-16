import itertools
from typing import Callable

from hippo.conceptgraph.conceptgraph_intake import load_conceptgraph
from hippo.reconstruction.scenedata import HippoObject, HippoRoomPlan
from hippo.utils.spatial_utils import get_size, disambiguate, get_bounding_box, filter_points_by_y_quartile
from hippo.utils.string_utils import get_uuid

import numpy as np


def get_hippos(path, pad=lambda bounddists: bounddists * 0.25):
    cg = load_conceptgraph(path)
    cg_objects = cg["segGroups"]

    hippo_objects = []

    pcds = []
    pcd_colors = []
    for i, obj in enumerate(cg_objects):
        #if i in [3, 13, 14, 25, 26]:
        #    continue

        object_name = obj["label"].lower().replace(":","").replace(",", " ").strip()

        pcd = np.asarray(obj["pcd"].points)[:,[0,2,1]]
        pcd_color = np.asarray(obj["pcd"].colors)

        object_description = obj["caption"]
        clip_features = obj["clip_features"]

        pcds.append(pcd)
        pcd_colors.append(pcd_color)

        hippo_objects.append(
            HippoObject(
                object_name=object_name,
                object_description=object_description,
                roomId=None, _position=None, _clip_features=clip_features,
                _desired_size=None, _cg_paths=obj["paths"])
        )

    allpoints = np.concatenate(pcds)
    _minbound, maxbound = allpoints.min(axis=0), allpoints.max(axis=0)

    bounddists = maxbound - _minbound

    if isinstance(pad, int) or isinstance(pad, float):
        pad = np.ones(3) * pad
    elif isinstance(pad, np.ndarray) or isinstance(pad, tuple) or isinstance(pad, list):
        pad = np.array(pad)
        pad[1] = 0
    elif isinstance(pad, Callable):
        pad = pad(bounddists)
        pad = np.array(pad)
        pad[1] = 0

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
    for hippo_object, pcd, pcd_color in (zip(hippo_objects,pcds, pcd_colors)):
        original_pcd = pcd

        pcd, pcd_color = filter_points_by_y_quartile(pcd, 1, 99, points_colors=pcd_color)

        position = np.median(pcd, axis=0)
        position[1] = np.min(pcd[:,1])
        #if position[1] < 0.1:
        #    position[1] = 0.0

        print(position)
        assert (position >= (minbound+pad)).all()
#        assert (position <= (maxbound-pad)).all()
        size = get_size(pcd, as_dict=False)
        hippo_object = hippo_object.replace(_position=position, _desired_size=size)
        hippo_object = hippo_object.set_pcd_(pcd, pcd_color)

        if hippo_object.object_name not in name2objs:
            name2objs[hippo_object.object_name] = []
        name2objs[hippo_object.object_name].append((hippo_object, get_bounding_box(original_pcd), len(original_pcd)))
        id2objs[hippo_object.object_name_id] = hippo_object

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
                keep_delete(keep1, obj1.object_name_id)
                keep_delete(keep2, obj2.object_name_id)
            elif keep1 != keep2:    # same thing as else lol
                if num_pcd2 > num_pcd1:
                    keep_delete(False, obj1.object_name_id)
                else:
                    keep_delete(False, obj2.object_name_id)


    hippo_objects = list(id2objs.values())

    minbound = (minbound[0] ,minbound[2])
    maxbound = (maxbound[0],maxbound[2])
    roomId = cg["sceneId"] + f"-{get_uuid()}"
    coords = ((minbound[0], minbound[1]), (minbound[0], maxbound[1]), (maxbound[0], maxbound[1]), (maxbound[0], minbound[1]))
    roomplan = HippoRoomPlan(id=roomId, coords=coords)

    hippo_objects = [h.replace(roomId=roomId) for h in hippo_objects]

    return roomplan, hippo_objects


if __name__ == "__main__":
    _, hippos = get_hippos("../sacha_kitchen", pad=2)
    hippos = [hippo.to_runtimeobject() for hippo in hippos]