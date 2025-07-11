import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import numpy as np


def sample_prism_surface(center, dims, N):
    """
    Sample N points approximately uniformly on the surface of a rectangular prism.

    Parameters
    ----------
    center : array-like of shape (3,)
        (cx, cy, cz) — center of the prism.
    dims : array-like of shape (3,)
        (dx, dy, dz) — width (x), height (y), depth (z).
    N : int
        Number of points to sample.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Sampled points.
    """
    cx, cy, cz = center
    dx, dy, dz = dims

    # Areas of each pair of opposite faces
    A_xy = dx * dy
    A_yz = dy * dz
    A_xz = dx * dz
    total_area = 2 * (A_xy + A_yz + A_xz)

    # Probabilities proportional to face areas
    probs = np.array([
        A_xy, A_xy,  # z = top/bottom
        A_yz, A_yz,  # x = left/right
        A_xz, A_xz  # y = front/back
    ]) / total_area

    # Sample which face each point goes to
    faces = np.random.choice(6, size=N, p=probs)

    points = np.zeros((N, 3))

    # Half-dimensions
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    # For each face, sample points
    for i in range(6):
        mask = (faces == i)
        n_i = np.sum(mask)
        if n_i == 0:
            continue

        u = np.random.uniform(-1, 1, size=(n_i,))
        v = np.random.uniform(-1, 1, size=(n_i,))

        if i == 0:  # z = +hz
            points[mask] = np.column_stack((
                cx + u * hx,
                cy + v * hy,
                np.full(n_i, cz + hz)
            ))
        elif i == 1:  # z = -hz
            points[mask] = np.column_stack((
                cx + u * hx,
                cy + v * hy,
                np.full(n_i, cz - hz)
            ))
        elif i == 2:  # x = +hx
            points[mask] = np.column_stack((
                np.full(n_i, cx + hx),
                cy + u * hy,
                cz + v * hz
            ))
        elif i == 3:  # x = -hx
            points[mask] = np.column_stack((
                np.full(n_i, cx - hx),
                cy + u * hy,
                cz + v * hz
            ))
        elif i == 4:  # y = +hy
            points[mask] = np.column_stack((
                cx + u * hx,
                np.full(n_i, cy + hy),
                cz + v * hz
            ))
        elif i == 5:  # y = -hy
            points[mask] = np.column_stack((
                cx + u * hx,
                np.full(n_i, cy - hy),
                cz + v * hz
            ))

    return points


def pcd_visualize(pcd):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    #view_control = visualizer.get_view_control()
    #view_control.set_front([1, 0, 0])
    #view_control.set_up([0, 0, 1])
    #view_control.set_lookat([0, 0, 0])
    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass
    visualizer.close()

def load_point_cloud(path):
    path = Path(path)
    assert path.exists(), path
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))



    if False:
        # Get points as numpy array
        points = np.asarray(pcd.points)
        # Mirror along X axis
        points[:, 0] *= -1  # Assign back
        pcd.points = o3d.utility.Vector3dVector(points)

        points = np.asarray(pcd.points)
        prism = sample_prism_surface([0,0,0], [1,1,1], 1000)
        points = np.concatenate([points, prism])
        pcd.points = o3d.utility.Vector3dVector(points)

    segments_anno = load_segments_anno(path)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)


    return pcd_o3d

def vis_cg(cg_pcds):
    for objpcd in cg_pcds:
        np.random.seed(int(np.sum(np.asarray(objpcd.points))) % (2**32 - 1))
        color_for_obj = np.random.choice(range(256), size=3)
        color_for_obj = np.repeat(color_for_obj[None], len(objpcd.points), axis=0)
        objpcd.colors = o3d.utility.Vector3dVector(color_for_obj / 255)

    combined = cg_pcds[0]
    for other in cg_pcds[1:]:
        combined = combined + other

    o3d.visualization.draw_geometries([combined])

def load_clip_features(path):
    path = Path(path)
    with open(path / "clip_features.npy", "rb") as f:
        return np.load(f)

def load_segments_anno(path):
    path = Path(path)
    with open(path / "segments_anno.json", "r") as f:
        return json.load(f)

def load_conceptgraph(path):
    pcd_dict = load_point_cloud(path)
    clip_features = load_clip_features(path)
    segments_anno = load_segments_anno(path)

    def setclip(grp, clip, pcd):
        grp["clip_features"] = clip
        grp["pcd"] = pcd
        return grp
    segments_anno["segGroups"] = [setclip(grp,clip,pcd) for grp, clip, pcd in zip(segments_anno["segGroups"], clip_features, pcd_dict)]

    from pathlib import Path
    for grp in segments_anno["segGroups"]:
        grp["paths"] = {
            "mask": str(Path(f"{path}/segments/{grp['id']}/mask").resolve()),
            "rgb": str(Path(f"{path}/segments/{grp['id']}/rgb").resolve())
        }

    for i, grp in enumerate(copy.deepcopy(segments_anno["segGroups"])):
        for to_remove in ["ceiling", "window blinds", "blinds", "wall", "window blinds", "blinds"]: #, "door"]:
            if to_remove in grp["label"]:
                segments_anno["segGroups"][i] = None

        """
        FOUND_KEEP = False
        for only_keep in ["wooden cabinet", "grid window", "armchair", "boat photograph"]:
            if only_keep in grp["label"]:
                FOUND_KEEP = True
        if not FOUND_KEEP:
            segments_anno["segGroups"][i] = None
        """

    segments_anno["segGroups"] = list(filter(lambda x: x is not None, segments_anno["segGroups"]))

    def vis_id2obj():
        pcds = []
        for v in segments_anno["segGroups"]:
            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(np.array(v._cg_pcd_points))
            #pcd.colors = o3d.utility.Vector3dVector(np.array(v._cg_pcd_colours))
            pcds.append(v["pcd"])
        return vis_cg(pcds)
    #vis_id2obj()

    return segments_anno



if __name__ == "__main__":
    PATH = "../sacha_kitchen"
    #load_clip_features(PATH)

    x = load_conceptgraph(PATH)

    x = load_point_cloud(PATH)

    for _x in x:
        pcd_visualize(_x)