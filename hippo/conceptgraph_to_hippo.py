import uuid

from hippo.conceptgraph_intake import load_conceptgraph
from hippo.scenedata import HippoObjectPlan, HippoRoomPlan
from hippo.spatial_utils import get_bounding_box
from hippo.string_utils import get_uuid

import numpy as np

def get_hippos(path, pad=1):
    cg = load_conceptgraph(path)
    cg_objects = cg["segGroups"]

    hippo_objects = []

    pcds = []

    done_objects = set()
    for obj in cg_objects:
        object_name = obj["label"].lower().replace(":","").replace(",", " ").strip()

        if object_name in done_objects:
            continue
        done_objects.add(object_name)

        object_description = obj["caption"]
        #positions.append(obj["centroid"])
        #x,y,z = obj["centroid"]
        #position = [x+4, z, y+4]
        clip_features = obj["clip_features"]

        pcds.append(np.asarray(obj["pcd"].points)[:,[0,2,1]])
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
        size = get_bounding_box(pcd, as_dict=False)
        hippo_objects[i] = hippo_objects[i].replace(position=position, _desired_size=size)

    minbound = (minbound[0] ,minbound[2])
    maxbound = (maxbound[0],maxbound[2])
    roomId = cg["sceneId"] + f"-{get_uuid()}"
    coords = ((minbound[0], minbound[1]), (minbound[0], maxbound[1]), (maxbound[0], maxbound[1]), (maxbound[0], minbound[1]))
    roomplan = HippoRoomPlan(id=roomId, coords=coords)


    return roomplan, hippo_objects

if __name__ == "__main__":
    get_hippoplans("./rgbd_interactions_2_l14")
