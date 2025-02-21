import json
from pathlib import Path

import numpy as np
import open3d as o3d

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
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))

    segments_anno = load_segments_anno(path)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d

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

    return segments_anno



if __name__ == "__main__":
    PATH = "../rgbd_interactions_2_l14"
    #load_clip_features(PATH)

    x = load_conceptgraph(PATH)

    x = load_point_cloud(PATH)

    for _x in x:
        pcd_visualize(_x)