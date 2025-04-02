import jax.numpy as jnp
from jax import jit
import jax

def _bool_ifelse_elementwise(cond, iftrue, iffalse):
    return iftrue * cond + iffalse * (1-cond)

@jax.jit
def bool_ifelse(cond, iftrue, iffalse):
    cond = jnp.atleast_1d(cond)
    iftrue = jnp.atleast_1d(iftrue)
    iffalse = jnp.atleast_1d(iffalse)

    MAIN_SHAPE = cond.shape[0]

    if len(iffalse.shape) == 0:
        iffalse = jnp.ones_like(cond) * iffalse

    if len(iftrue.shape) == 0:
        iftrue = jnp.ones_like(cond) * iftrue

    if iffalse.shape[0] != MAIN_SHAPE:
        iffalse = jnp.repeat(iffalse[None], MAIN_SHAPE, axis=0)
    if iftrue.shape[0] != MAIN_SHAPE:
        iftrue = jnp.repeat(iftrue[None], MAIN_SHAPE, axis=0)

    cond = cond.astype(int)

    return jax.vmap(_bool_ifelse_elementwise)(cond, iftrue, iffalse)

def isOnTop(pos1, size1, pos2, size2, tol_overlap=0.5, tol_dist=0.1):
    """Returns True if obj1 is on top of obj2 within given tolerances."""
    # Compute XY overlap as the intersection over the area of obj1
    overlap_x = jnp.maximum(0, jnp.minimum(pos1[0] + size1[0] / 2, pos2[0] + size2[0] / 2) -
                jnp.maximum(pos1[0] - size1[0] / 2, pos2[0] - size2[0] / 2))
    overlap_y = jnp.maximum(0, jnp.minimum(pos1[1] + size1[1] / 2, pos2[1] + size2[1] / 2) -
                jnp.maximum(pos1[1] - size1[1] / 2, pos2[1] - size2[1] / 2))
    overlap_area = overlap_x * overlap_y
    obj1_area = size1[0] * size1[1]

    xy_overlap_ratio = bool_ifelse(obj1_area>0, overlap_area / obj1_area, 0)

    # Check if obj1's bottom is close to obj2's top
    top_z_obj2 = pos2[2] + size2[2] / 2
    bottom_z_obj1 = pos1[2] - size1[2] / 2
    z_close = jnp.abs(bottom_z_obj1 - top_z_obj2) <= tol_dist

    return (xy_overlap_ratio >= tol_overlap) & z_close

def isInside(pos1, size1, pos2, size2, tol_overlap=0.99,  tol_dist=None):
    """Returns True if obj1 is inside obj2 within the overlap tolerance."""
    # Compute XYZ overlap volume
    overlap_x = jnp.maximum(0, jnp.minimum(pos1[0] + size1[0] / 2, pos2[0] + size2[0] / 2) -
                jnp.maximum(pos1[0] - size1[0] / 2, pos2[0] - size2[0] / 2))
    overlap_y = jnp.maximum(0, jnp.minimum(pos1[1] + size1[1] / 2, pos2[1] + size2[1] / 2) -
                jnp.maximum(pos1[1] - size1[1] / 2, pos2[1] - size2[1] / 2))
    overlap_z = jnp.maximum(0, jnp.minimum(pos1[2] + size1[2] / 2, pos2[2] + size2[2] / 2) -
                jnp.maximum(pos1[2] - size1[2] / 2, pos2[2] - size2[2] / 2))

    overlap_volume = overlap_x * overlap_y * overlap_z
    obj1_volume = size1[0] * size1[1] * size1[2]

    overlap_ratio = bool_ifelse(obj1_volume > 0, overlap_volume / obj1_volume, 0)

    return overlap_ratio >= tol_overlap


def isBeside_OLD(pos1, size1, pos2, size2, tol_dist=0.1):
    """Returns True if obj1 is beside obj2 within the distance tolerance."""
    # Check if the bases of both objects are at the same height (Z level)
    base_z_obj1 = pos1[2] - size1[2] / 2  # Bottom of obj1
    base_z_obj2 = pos2[2] - size2[2] / 2  # Bottom of obj2
    same_surface = jnp.abs(base_z_obj1 - base_z_obj2) <= (tol_dist / 4) # less tolerant for vertical alignment

    # Check if they are adjacent in the XY plane
    adjacent_x = jnp.abs(pos1[0] - pos2[0]) <= (size1[0] + size2[0]) / 2 + tol_dist
    adjacent_y = jnp.abs(pos1[1] - pos2[1]) <= (size1[1] + size2[1]) / 2 + tol_dist

    return same_surface & (adjacent_x | adjacent_y)


def isBeside_EDGE(pos1, size1, pos2, size2, tol_dist=0.5):
    """Check if edge-to-edge distance is within tolerance."""
    same_surface = jnp.abs((pos1[2] - size1[2] / 2) - (pos2[2] - size2[2] / 2)) <= (tol_dist / 4)

    # Calculate min edge-to-edge distance in XY
    dx = jnp.abs(pos1[0] - pos2[0]) - (size1[0] + size2[0]) / 2
    dy = jnp.abs(pos1[1] - pos2[1]) - (size1[1] + size2[1]) / 2
    edge_dist = jnp.maximum(dx, dy)  # Or use jnp.sqrt(dx**2 + dy**2) for diagonal
    return same_surface & (edge_dist <= tol_dist)


def isBeside(pos1, size1, pos2, size2, tol_dist=0.5):
    """Check if objects overlap in one axis and are close in the other."""
    same_surface = jnp.abs((pos1[2] - size1[2] / 2) - (pos2[2] - size2[2] / 2)) <= (tol_dist / 4)

    # Overlap in X and proximity in Y (or vice versa)
    overlap_x = (jnp.abs(pos1[0] - pos2[0]) <= (size1[0] + size2[0]) / 2)
    proximity_y = (jnp.abs(pos1[1] - pos2[1]) <= (size1[1] + size2[1]) / 2 + tol_dist)

    # Overlap in Y and proximity in X (or vice versa)
    overlap_y = (jnp.abs(pos1[1] - pos2[1]) <= (size1[1] + size2[1]) / 2)
    proximity_x = (jnp.abs(pos1[0] - pos2[0]) <= (size1[0] + size2[0]) / 2 + tol_dist)

    return same_surface & ((overlap_x & proximity_y) | (overlap_y & proximity_x))

def isNear(pos1, size1, pos2, size2,tol_overlap=None, tol_dist=0.5):
    """Returns True if obj1 is near obj2 within the distance tolerance."""
    distance = jnp.linalg.norm(pos1 - pos2)
    return distance <= tol_dist

def distance(pos1, size1, pos2, size2, tol_overlap=None, tol_dist=None):
    """Returns the Euclidean distance between obj1 and obj2."""
    return jnp.linalg.norm(pos1 - pos2)
