import open3d
import open3d as o3d
import numpy as np

def get_bounding_box(points, as_dict=True, array_backend=np):
    if isinstance(points, o3d.geometry.MeshBase):
        bbox = points.get_axis_aligned_bounding_box()

        max_bounds = bbox.get_max_bound()
        min_bounds = bbox.get_min_bound()
    else:
        if isinstance(points, o3d.geometry.PointCloud):
            points = array_backend.array(np.asarray(points.points))

        min_bounds, max_bounds = points.min(axis=0), points.max(axis=0)

    dico = {k: max_bounds[i] - min_bounds[i] for i,k in enumerate(["x", "y", "z"])}

    if as_dict:
        return dico

    return dico["x"], dico["y"], dico["z"]
