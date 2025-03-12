import copy

import open3d
import open3d as o3d
import numpy as np
from ai2thor.util.runtime_assets import load_existing_thor_asset_file, save_thor_asset_file

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR


def get_bounding_box(points):

    if isinstance(points, o3d.geometry.MeshBase):
        bbox = points.get_axis_aligned_bounding_box()

        max_bounds = bbox.get_max_bound()
        min_bounds = bbox.get_min_bound()
    else:
        if isinstance(points, o3d.geometry.PointCloud):
            points = np.asarray(points.points)

        min_bounds, max_bounds = points.min(axis=0), points.max(axis=0)

    return *min_bounds, *max_bounds


def get_size(points, as_dict=True):
    bbox = get_bounding_box(points)
    min_bounds, max_bounds = bbox[:3], bbox[3:]

    dico = {k: max_bounds[i] - min_bounds[i] for i,k in enumerate(["x", "y", "z"])}

    if as_dict:
        return dico

    return dico["x"], dico["y"], dico["z"]


def calculate_volume(bbox):
    """Calculate the volume of an axis-aligned bounding box."""
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)


def calculate_intersection_volume(bbox1, bbox2):
    """Calculate the volume of the intersection between two bounding boxes."""
    x_min1, y_min1, z_min1, x_max1, y_max1, z_max1 = bbox1
    x_min2, y_min2, z_min2, x_max2, y_max2, z_max2 = bbox2

    x_min_int = max(x_min1, x_min2)
    y_min_int = max(y_min1, y_min2)
    z_min_int = max(z_min1, z_min2)
    x_max_int = min(x_max1, x_max2)
    y_max_int = min(y_max1, y_max2)
    z_max_int = min(z_max1, z_max2)

    intersection_bbox = (x_min_int, y_min_int, z_min_int, x_max_int, y_max_int, z_max_int)
    return calculate_volume(intersection_bbox)


def is_fully_contained(bbox1, bbox2):
    """Check if bbox2 is fully contained in bbox1, or vice versa."""

    def contained(bbox_inner, bbox_outer):
        return (bbox_outer[0] <= bbox_inner[0] <= bbox_inner[3] <= bbox_outer[3] and
                bbox_outer[1] <= bbox_inner[1] <= bbox_inner[4] <= bbox_outer[4] and
                bbox_outer[2] <= bbox_inner[2] <= bbox_inner[5] <= bbox_outer[5])

    return contained(bbox2, bbox1) or contained(bbox1, bbox2)


def disambiguate(bbox1, bbox2, tresh_overlap=0.8, tresh_noice=0.5):
    # Step 1: Check for zero volumes
    volume1 = calculate_volume(bbox1)
    volume2 = calculate_volume(bbox2)

    if volume1 == 0 and volume2 == 0:
        return (False, False)  # Both are noise
    elif volume1 == 0:
        return (False, True)  # Only obj2 is real
    elif volume2 == 0:
        return (True, False)  # Only obj1 is real

    # Step 2: Check for full containment
    if is_fully_contained(bbox1, bbox2):
        return (True, False)  # obj2 is noise, fully contained in obj1
    if is_fully_contained(bbox2, bbox1):
        return (False, True)  # obj1 is noise, fully contained in obj2

    # Step 3: Calculate intersection volume
    intersection_volume = calculate_intersection_volume(bbox1, bbox2)

    # Step 4: Calculate overlap ratios
    smaller_volume = min(volume1, volume2)
    larger_volume = max(volume1, volume2)
    overlap_ratio = intersection_volume / smaller_volume
    noise_ratio = intersection_volume / larger_volume

    # fixme requires tuning
    # Step 5: Decision logic for partial overlaps
    if overlap_ratio > 0.1:
        return (True, False) if volume1 > volume2 else (False, True)
    elif noise_ratio > 0.1:
        return (True, False) if volume1 > volume2 else (False, True)

    return True, True

def rescale_asset(asset_dir, asset_id, scaling):
    import os
    asset_path = os.path.join(asset_dir, asset_id)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(asset_path)

    o3d.io.read_triangle_mesh(asset_path)


def read_list_dictpoints(obj, key):
    return np.array([[p["x"], p["y"], p["z"]] for p in obj[key]])

def write_list_dictpoints(obj, key, value):
    #obj = copy.deepcopy(obj)
    obj[key] = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in value]
    return obj

def scale_ai2thor_object(obj, scale_factors):
    #obj = copy.deepcopy(obj)

    sx, sy, sz = scale_factors

    vertices = read_list_dictpoints(obj, "vertices")
    vertices *= np.array([sx, sy, sz])
    obj = write_list_dictpoints(obj, "vertices", vertices)

    # Scale visibility points if they exist
    if 'visibilityPoints' in obj:
        vis_points = read_list_dictpoints(obj, "visibilityPoints")
        vis_points *= np.array([sx, sy, sz])
        obj = write_list_dictpoints(obj, "visibilityPoints", vis_points)

    # Adjust normals for non-uniform scaling
    if sx != sy or sy != sz or sx != sz:
        normals = read_list_dictpoints(obj, "normals")
        scale_matrix = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1 / sz]])
        normals = normals @ scale_matrix.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize
        obj = write_list_dictpoints(obj, "normals", normals)

    new_colliders = []
    for collider in obj["colliders"]:
        vertices = read_list_dictpoints(collider, "vertices")
        vertices *= np.array([sx, sy, sz])
        collider = write_list_dictpoints(collider, "vertices", vertices)
        new_colliders.append(collider)

    obj["colliders"] = new_colliders
    return obj

def vis_ai2thor_object(obj):
    combined_mesh = o3d.geometry.TriangleMesh()

    for collider in obj["colliders"]:
        # Load vertices and reshape
        vertices = read_list_dictpoints(collider, "vertices") # np.array(collider["vertices"], dtype=np.float32).reshape(-1, 3)

        # Load triangle indices and reshape
        triangles = np.array(collider["triangles"], dtype=np.int32).reshape(-1, 3)

        # Create a mesh for this collider
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # Merge with combined mesh
        combined_mesh += mesh

    o3d.visualization.draw_geometries([mesh])
    return
    # Create Open3D triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(read_list_dictpoints(obj, "vertices"))
    mesh.triangles = o3d.utility.Vector3iVector(obj["triangles"])

    # Optional: Compute vertex normals for better visualization
    mesh.compute_vertex_normals()

    # Visualize
    o3d.visualization.draw_geometries([mesh])


def filter_points_by_y_quartile(points, lower_percentile=5, upper_percentile=95):
    """
    Filters points based on their y-axis values, removing points outside the specified percentile range.

    Parameters:
    - points (numpy.ndarray): (N, 3) array of 3D points.
    - lower_percentile (float): Lower bound percentile (e.g., 5 for the 5th percentile).
    - upper_percentile (float): Upper bound percentile (e.g., 95 for the 95th percentile).

    Returns:
    - filtered_points (numpy.ndarray): The points within the y-axis bounds.
    - bbox_min (numpy.ndarray): Min corner of the bounding box (x_min, y_min, z_min).
    - bbox_max (numpy.ndarray): Max corner of the bounding box (x_max, y_max, z_max).
    """
    # Extract y-values from the points
    y_values = points[:, 1]

    # Calculate percentiles for the y-axis
    y_lower = np.percentile(y_values, lower_percentile)
    y_upper = np.percentile(y_values, upper_percentile)

    # Filter out points based on y-axis bounds
    filtered_points = points[(y_values >= y_lower) & (y_values <= y_upper)]

    # Compute bounding box
    #bbox_min = np.min(filtered_points, axis=0)
    #bbox_max = np.max(filtered_points, axis=0)

    return filtered_points #, bbox_min, bbox_max



if __name__ == '__main__':
    x = load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, "546bb213065c45b4aa3267e85d03e725/546bb213065c45b4aa3267e85d03e725")

    save_thor_asset_file(x, "../temp.pkl.gz")

    vis_ai2thor_object(x)
    exit()
    scale_ai2thor_object(x, (1, 0.8, 2))


    print(x.keys())

    rescale_asset(OBJATHOR_ASSETS_DIR, "9c219d7d26ac47beaa46539a9a4fa8da/9c219d7d26ac47beaa46539a9a4fa8da.pkl.gz", None)



# Example usage
#pos1 = (1, 1, 1)
#bbox1 = (0, 0, 0, 2, 2, 2)
#pos2 = (1.5, 1.5, 1.5)
#bbox2 = (1, 1, 1, 2.5, 2.5, 2.5)

#keep = disambiguate(pos1, bbox1, pos2, bbox2)
#print(keep)  # Should print a tuple like (True, False) or (True, True)

