import open3d
import open3d as o3d
import numpy as np

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

# Example usage
#pos1 = (1, 1, 1)
#bbox1 = (0, 0, 0, 2, 2, 2)
#pos2 = (1.5, 1.5, 1.5)
#bbox2 = (1, 1, 1, 2.5, 2.5, 2.5)

#keep = disambiguate(pos1, bbox1, pos2, bbox2)
#print(keep)  # Should print a tuple like (True, False) or (True, True)

