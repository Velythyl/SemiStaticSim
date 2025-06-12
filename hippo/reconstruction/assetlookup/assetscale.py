import numpy as np
def dict2xyztuple(dic):
    return (dic['x'], dic['y'], dic['z'])
from hippo.utils.spatial_utils import pcd_bbox_size, transform_ai2thor_object
from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

def scale_object(cfg, ai2thor_obj, target_pcd):
    if cfg.assetfitting.scaling == "axis":
        return axis_scale(ai2thor_obj, target_pcd)

    elif cfg.assetfitting.scaling.startswith("aspect"):
        type = cfg.assetfitting.scaling.replace("aspect", "").strip()

        assert type in ["fit", "fill"]

        return aspect_scale(ai2thor_obj, target_pcd, type)

    raise NotImplemented(f"Scaling not implemented for <{cfg.assetfitting.scaling}>")


def axis_scale(ai2thor_obj, target_pcd):


    FIXED_TARGET_SIZE = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))

    obj_pcd_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    scaling = FIXED_TARGET_SIZE / obj_pcd_size
    print(np.linalg.norm(scaling))
    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, add_scaling_to_transmat(np.eye(4), scaling))
    return ai2thor_obj


def aspect_scale(ai2thor_obj, target_pcd, mode='fit'):
    """
    Scale object while maintaining aspect ratio.

    Args:
        ai2thor_obj: The object to scale
        target_pcd: The target point cloud to match
        mode: 'fit' (scale to fit inside target, default) or
              'fill' (scale to fill target, may exceed in some dimensions)
    """
    from hippo.utils.spatial_utils import pcd_bbox_size
    from hippo.reconstruction.assetlookup.assetalign import pcd_or_mesh_to_np, add_scaling_to_transmat

    target_size = np.array(dict2xyztuple(pcd_bbox_size(target_pcd)))
    obj_size = np.array(dict2xyztuple(pcd_bbox_size(pcd_or_mesh_to_np(ai2thor_obj))))

    scale_factors = target_size / obj_size

    if mode == 'fit':
        uniform_scale = np.min(scale_factors)  # Ensures object fits within target
    elif mode == 'fill':
        uniform_scale = np.max(scale_factors)  # Ensures object fills target
    else:
        raise ValueError("mode must be either 'fit' or 'fill'")

    scaling = np.array([uniform_scale, uniform_scale, uniform_scale])
    ai2thor_obj = transform_ai2thor_object(ai2thor_obj, add_scaling_to_transmat(np.eye(4), scaling))

    return ai2thor_obj