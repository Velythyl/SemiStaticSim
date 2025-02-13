import numpy as np

from hippo.spatial_utils import get_bounding_box


def real_object_to_sim_object(real_object):
    """

    :param real_object: a JSON containing the object name, a short description (?), and a point cloud
    :return: a JSON compatible with Holodeck
    """

    pcd = real_object["point_cloud"]

    sim_object = {

        "assetMetadata": get_bounding_box(pcd, as_dict=True, array_backend=np)
    }

    bject_type = target_object_information["object_name"]
    object_description = target_object_information["description"]
    object_size = target_object_information["size"]