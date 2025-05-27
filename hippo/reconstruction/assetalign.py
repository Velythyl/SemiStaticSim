import functools
from typing import Dict

import jax.numpy as jnp
import numpy as np
import open3d as o3d
import jax


def rotate_point_cloud_y_axis_OLD(points, theta):
    """
    Rotate a point cloud around the y-axis by a given angle in degrees.

    Args:
        points: (N, 3) numpy array or jax array representing the point cloud.
        degrees: Rotation angle in degrees (positive for counter-clockwise rotation).

    Returns:
        Rotated point cloud as a jax array of shape (N, 3).
    """

    # Compute rotation matrix for y-axis
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    rotation_matrix = jnp.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # Apply rotation
    rotated_points = jnp.dot(points, rotation_matrix.T)  # Transpose for correct multiplication

    rotated_points = center_point_cloud(rotated_points)

    return rotated_points

def rotate_point_cloud_z_axis(points, theta):
    """
    Rotate a point cloud around the z-axis by a given angle in degrees.

    Args:
        points: (N, 3) numpy array or jax array representing the point cloud.
        degrees: Rotation angle in degrees (positive for counter-clockwise rotation).

    Returns:
        Rotated point cloud as a jax array of shape (N, 3).
    """

    # Compute rotation matrix for z-axis
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    rotation_matrix = jnp.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    # Apply rotation
    rotated_points = jnp.dot(points, rotation_matrix.T)  # Transpose for correct multiplication

    rotated_points = center_point_cloud(rotated_points)

    return rotated_points


def pairwise_distances(points):
    """
    Compute pairwise Euclidean distances between 3D points.

    Args:
        points: Array of shape (n, 3) containing n 3D points

    Returns:
        Distance matrix of shape (n, n) where D[i,j] is the distance between points i and j
    """
    # Compute squared differences between all pairs of points
    diff = points[:, None, :] - points[None, :, :]  # Shape: (n, n, 3)
    squared_diffs = diff ** 2

    # Sum along the last dimension (x,y,z coordinates) and take square root
    distances = jnp.sqrt(jnp.sum(squared_diffs, axis=-1))

    return distances

def point2point_dist(p1, p2):
    return jnp.sqrt(jnp.sum((p2-p1)**2))

def get_dists_for_p1(p1, target_pcd):
    return jax.vmap(functools.partial(point2point_dist, p1))(target_pcd)

def get_mindist_for_p1(target_pcd, p1):
    return get_dists_for_p1(p1, target_pcd).min()

def get_score_OLD(pcd_to_rotate, pcd_to_match, rad):
    pcd_to_rotate = rotate_point_cloud_z_axis(pcd_to_rotate, rad)
    mindists = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_match))(pcd_to_rotate)

    return mindists.mean()


def get_score(pcd_to_rotate, pcd_to_match, rad):
    pcd_to_rotate = rotate_point_cloud_z_axis(pcd_to_rotate, rad)

    # Distance from rotated to target
    mindists_rot_to_target = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_match))(pcd_to_rotate)

    # Distance from target to rotated (important for symmetry)
    mindists_target_to_rot = jax.vmap(functools.partial(get_mindist_for_p1, pcd_to_rotate))(pcd_to_match)

    return (mindists_rot_to_target.mean() + mindists_target_to_rot.mean()) / 2
from tqdm import tqdm


def center_point_cloud(points):
    """
    Center a point cloud so its centroid is at (0, 0, 0).

    Parameters:
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D points

    Returns:
    numpy.ndarray
        Centered point cloud
    """
    # Calculate the centroid (mean along each axis)
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    return centered_points

def pcd_or_mesh_to_np(pcd_or_mesh, NUM_POINTS_TO_KEEP=2500):
    if isinstance(pcd_or_mesh, jnp.ndarray):
        return pcd_or_mesh_to_np(np.array(pcd_or_mesh))

    if isinstance(pcd_or_mesh, np.ndarray):
        if pcd_or_mesh.shape[0] > NUM_POINTS_TO_KEEP:
            pcd_or_mesh = pcd_or_mesh[np.random.choice(np.arange(pcd_or_mesh.shape[0]), size=NUM_POINTS_TO_KEEP, replace=False)]
        return center_point_cloud(pcd_or_mesh.copy())

    if isinstance(pcd_or_mesh, o3d.geometry.PointCloud):
        pcd = pcd_or_mesh.voxel_down_sample(voxel_size=0.05)
        pcd = np.asarray(pcd.points)
        return pcd_or_mesh_to_np(pcd)

    if isinstance(pcd_or_mesh, Dict):
        newpcs = []
        for p in pcd_or_mesh["vertices"]:
            newpcs.append(np.array([p['x'], p['y'], p['z']]))
        pcd = np.vstack(newpcs)
        return pcd_or_mesh_to_np(pcd)

    if isinstance(pcd_or_mesh, o3d.geometry.TriangleMesh):
        return pcd_or_mesh_to_np(pcd_or_mesh.sample_points_uniformly(number_of_points=NUM_POINTS_TO_KEEP))

def global_align(pcd_to_rotate: np.array, pcd_to_match: np.array):
    trials = jnp.linspace(0,  2*np.pi, 200)

    BATCH_SIZE = 5

    scores = []
    for b_trials in tqdm(trials.reshape(-1 ,BATCH_SIZE)):
        scores.append(jax.vmap(functools.partial(get_score, pcd_to_rotate, pcd_to_match))(b_trials))
        #scores.append(get_score(pcd_to_rotate, pcd_to_match, trial))
    scores = jnp.concatenate(scores)
    #scores = jax.vmap(functools.partial(get_score, pcd_to_rotate, pcd_to_match))(trials)
    best_rot = scores.argmin()
    found_rot = trials[best_rot]
    return found_rot

def swap_yz(pcd: np.array):
    return pcd[:,[0,2,1]]

def vis(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


def make_z_trans_init_matrix(theta):
    """
    Create a 4x4 homogeneous transformation matrix for z-axis rotation.

    Args:
        theta: Rotation angle in radians

    Returns:
        4x4 numpy array representing the transformation
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def fine_tune(pcd_to_align, target_pcd, initial_zrot):
    #initted_pcd = rotate_point_cloud_z_axis(pcd_to_align, initial_zrot)
    trans_init = make_z_trans_init_matrix(initial_zrot)

    pcd_to_align = transform_point_cloud(pcd_to_align, trans_init)
    _pcd_to_align = o3d.geometry.PointCloud()
    _pcd_to_align.points = o3d.utility.Vector3dVector(pcd_to_align)
    pcd_to_align = _pcd_to_align

    _target_pcd = o3d.geometry.PointCloud()
    _target_pcd.points = o3d.utility.Vector3dVector(target_pcd)
    target_pcd = _target_pcd

    fine_tuned = o3d.pipelines.registration.registration_icp(
        pcd_to_align, target_pcd, 0.02, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return fine_tuned.transformation


def transform_point_cloud(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to a 3D point cloud.

    Args:
        points: Array of shape (N, 3) containing N 3D points
        transformation_matrix: 4x4 affine transformation matrix

    Returns:
        Transformed points of shape (N, 3)
    """
    # Convert points to homogeneous coordinates (N, 4)
    homogeneous_points = jnp.concatenate([
        points,
        jnp.ones((points.shape[0], 1))  # Add homogeneous coordinate of 1
    ], axis=1)

    # Apply transformation (N, 4) = (N, 4) @ (4, 4)
    transformed_points = homogeneous_points @ transformation_matrix.T

    # Convert back to Cartesian coordinates by dividing by w (N, 3)
    return transformed_points[:, :3] / transformed_points[:, 3:4]


def align(pcd_to_align, spoof_rad=None, target_pcd=None):
    if target_pcd is None:
        target_pcd = pcd_to_align
        assert spoof_rad is not None
    if spoof_rad is not None:
        pcd_to_align = rotate_point_cloud_z_axis(pcd_or_mesh_to_np(pcd), spoof_rad)

    pcd_to_align = pcd_or_mesh_to_np(pcd_to_align)
    target_pcd = pcd_or_mesh_to_np(target_pcd)
    vis(target_pcd)
    vis(pcd_to_align)
    found_rot = global_align(pcd_to_align, target_pcd)
    vis(rotate_point_cloud_z_axis(pcd_to_align, found_rot))

    print(make_z_trans_init_matrix(found_rot))

    tuned_transform = fine_tune(pcd_to_align, target_pcd, found_rot)

    print(tuned_transform)

    vis(transform_point_cloud(pcd_to_align, tuned_transform))



if __name__ == "__main__":
    from hippo.conceptgraph.conceptgraph_intake import load_conceptgraph

    # 21
    #cg = load_conceptgraph("/home/charlie/Desktop/Holodeck/hippo/datasets/replica_room0_cg-detector_2025-04-04-18-03-58")
    #pcd = cg["segGroups"][21]['pcd']
    #pcd = pcd_or_mesh_to_np(pcd)
    #align(pcd, rad=np.pi/2)

    # 24
    mesh = "/home/charlie/TRELLIS/datasets/replica_room0_cg-detector_2025-04-04-18-03-58/segments/24/masked/convert/c036e0c8"
    from ai2thor.util.runtime_assets import load_existing_thor_asset_file
    obj = load_existing_thor_asset_file("/home/charlie/TRELLIS/datasets/replica_room0_cg-detector_2025-04-04-18-03-58/segments/24/masked/convert",
                                        f"c036e0c8/c036e0c8")  # load_existing_thor_asset_file(OBJATHOR_ASSETS_DIR, f"{assetId}/{assetId}")
    cg = load_conceptgraph(
        "/home/charlie/Desktop/Holodeck/hippo/datasets/replica_room0_cg-detector_2025-04-04-18-03-58")
    pcd = cg["segGroups"][24]['pcd']
    obj = swap_yz(pcd_or_mesh_to_np(obj))
    align(obj, spoof_rad=None, target_pcd=pcd)

