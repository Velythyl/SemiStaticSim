import jax
import jax.numpy as jnp


def is_single_plane(points, noise_threshold=0.015, min_points=10, min_extent_ratio=10.0):
    """
    Check if a point cloud is approximately a single plane with noise tolerance.

    Args:
        points: (N, 3) array of 3D points
        noise_threshold: Max ratio of smallest eigenvalue to sum of eigenvalues to accept as planar
        min_points: Minimum number of points required for plane detection
        min_extent_ratio: Minimum ratio of extent along largest eigenvalue to smallest extent direction

    Returns:
        bool: True if the points form a single plane within noise and extent criteria
        plane_params: (4,) array containing plane equation coefficients (a,b,c,d) for ax+by+cz+d=0
    """
    if len(points) < min_points:
        return False, jnp.array([0., 0., 0., 0.])

    centroid = jnp.mean(points, axis=0)
    centered = points - centroid

    cov = jnp.cov(centered, rowvar=False)

    # SVD is equivalent to eigen decomposition of covariance
    with jax.default_device(jax.devices('cpu')[0]):
        _, s, vh = jnp.linalg.svd(cov)
    eigenvalues = s
    eigenvectors = vh.T

    # Check flatness: smallest eigenvalue vs sum
    ratio = eigenvalues[2] / (eigenvalues.sum() + 1e-10)
    is_flat_enough = ratio < noise_threshold

    # Project points onto eigenbasis to compute extents
    projected = centered @ eigenvectors  # (N,3) in principal axes

    extents = jnp.max(projected, axis=0) - jnp.min(projected, axis=0)  # length along each axis

    # Require that the two largest extents are much bigger than the smallest
    extent_ratio = (extents[0] + 1e-6) / (extents[2] + 1e-6)  # largest to smallest extent
    is_large_enough = extent_ratio > min_extent_ratio

    is_plane = is_flat_enough and is_large_enough

    normal = eigenvectors[:, 2]
    d = -jnp.dot(normal, centroid)
    plane_params = jnp.concatenate([normal, jnp.array([d])])
    print(f"This object is a plane? {is_plane}")
    return is_plane, plane_params



# JIT compile for better performance
#is_single_plane_jitted = jit(is_single_plane)

# Example usage:
if __name__ == "__main__":
    # Generate test data - a plane with some noise
    key = jax.random.PRNGKey(0)
    num_points = 1000

    # Create a plane in xy direction with some noise
    plane_points = jax.random.uniform(key, (num_points, 2), minval=-1, maxval=1)
    z_noise = jax.random.normal(key, (num_points,)) * 0.02  # Small noise
    points = jnp.column_stack([plane_points, z_noise])

    # Check if it's a plane
    is_plane, plane_eq = is_single_plane(points)
    print(f"Is plane: {is_plane}")
    print(f"Plane equation: {plane_eq}")

    # Create non-planar data (a cube)
    cube_points = jax.random.uniform(key, (num_points, 3), minval=-1, maxval=1)
    is_plane_cube, _ = is_single_plane(cube_points)
    print(f"Cube is plane: {is_plane_cube}")

    # Create non-planar data (a sphere)
    # Generate random points on a sphere surface
    key, subkey = jax.random.split(key)
    sphere_points = jax.random.normal(subkey, (num_points, 3))
    sphere_points = sphere_points / jnp.linalg.norm(sphere_points, axis=1, keepdims=True)
    # Add some noise to make it more realistic
    key, subkey = jax.random.split(key)
    sphere_points += jax.random.normal(subkey, (num_points, 3)) * 0.01
    is_plane_sphere, _ = is_single_plane(sphere_points)
    print(f"Sphere is plane: {is_plane_sphere}")