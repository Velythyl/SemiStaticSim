import copy
import dataclasses
import functools
import json
import random
from functools import partial, singledispatch

from flax import struct
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon
from shapely.ops import triangulate
from functools import singledispatchmethod
import random
from shapely.geometry import Polygon
from shapely.ops import triangulate
import jax.numpy as jnp
import jax
import numpy as np
import networkx as nx

from semistaticsim.simulation.spatialutils.filter_positions import build_grid_graph


@partial(jax.jit, static_argnums=(1,))
def split_key(key, num_keys):
    key, *rng = jax.random.split(key, num_keys + 1)
    rng = jnp.reshape(jnp.stack(rng), (num_keys, 2))
    return key, rng

def point_to_shapely(point):
    if len(point) == 3:
        def cast(p):
            return shapely.Point(p[0], p[2])
    else:
        def cast(p):
            return shapely.Point(p[0], p[1])
    return cast(point)



@struct.dataclass
class JaxPolygon:
    coords_polygon: list[tuple[float, float]]
    triangles: jnp.ndarray
    triangle_areas: jnp.ndarray

    @property
    def shapely_polygon(self):
        return shapely.Polygon(self.coords_polygon)

    @classmethod
    def create(cls, coords):
        if isinstance(coords[0], dict):
            coords = list(map(lambda x: (x["x"], x["z"]), coords))
        self = JaxPolygon(coords, None, None)

        polygon = self.shapely_polygon
        def polygon_to_triangles(polygon):
            """
            Triangulate a shapely polygon and return list of triangles.
            """
            tris = shapely.ops.triangulate(polygon)

            ret = []
            for tri in tris:
                centroid = tri.centroid
                if not polygon.contains(centroid):
                    continue
                ret.append(list(tri.exterior.coords)[:-1])

            return ret
        tris = jnp.array(polygon_to_triangles(polygon))

        def triangle_area(tri):
            x1, y1 = tri[0]
            x2, y2 = tri[1]
            x3, y3 = tri[2]
            return 0.5 * jnp.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        triangle_areas = jax.vmap(triangle_area)(tris)

        return self.replace(triangles=tris, triangle_areas=triangle_areas)

    @property
    def num_triangles(self):
        return self.triangles.shape[0]

    @property
    def bbox(self):
        coords = self.coords_polygon

        min_x, min_y = jnp.inf, jnp.inf
        max_x, max_y = -jnp.inf, -jnp.inf

        for coord in coords:
            min_x = min(min_x, coord[0])
            min_y = min(min_y, coord[1])
            max_x = max(max_x, coord[0])
            max_y = max(max_y, coord[1])

        return min_x, min_y, max_x, max_y

    def containment_mask(self, points):
        return [self.shapely_floor_polygon.contains(point_to_shapely(point)) for point in points]

    def points_distance_to_polygon_edge(self, points):
        polygon = self.shapely_floor_polygon
        return [point_to_shapely(point).distance(polygon.boundary) for point in points]

    def points_distance_to_polygon_edge_mask(self, points, mindist):
        dists = self.points_distance_to_polygon_edge(points)
        return [d > mindist for d in dists]

    @property
    def floor_polygon_coords(self):
        return list(map(lambda x: (x["x"], x["z"]), self.coords_polygon))

    @property
    def shapely_floor_polygon(self):
        return Polygon(self.floor_polygon_coords)

    @property
    def shapely_triangles(self):
        ret = []
        for tri in self.triangles:
            ret.append(Polygon(tri))
        return ret

    def plot_self(self, samples=None, plot_triangles=False, reachability_graph=None, show=False):
        plt.plot(*self.shapely_floor_polygon.exterior.xy)
        if plot_triangles:
            for tri in self.shapely_triangles:
                x, y = tri.exterior.xy
                plt.fill(x, y, alpha=0.2)

        if reachability_graph is not None:

            def draw_grid_graph_2d(G, path=None, node_size=100, node_color='lightblue', edge_color='gray',
                                   show_labels=False):
                pos = {node: (node[2], node[0]) for node in G.nodes}  # node positions are their coordinates

                nx.draw(G, pos,
                        node_size=node_size,
                        node_color=node_color,
                        edge_color=edge_color,
                        with_labels=show_labels,
                        font_size=8)

                # If a path is provided, draw it in red
                if path:
                    # Draw path edges first (so nodes will be on top)
                    path_edges = list(zip(path[:-1], path[1:]))
                    nx.draw_networkx_edges(G, pos,
                                           edgelist=path_edges,
                                           edge_color='red',
                                           width=2)
                    # Draw path nodes
                    nx.draw_networkx_nodes(G, pos,
                                           nodelist=path,
                                           node_size=node_size,
                                           node_color='red')

            draw_grid_graph_2d(reachability_graph)

        if samples is not None:
            plt.scatter(samples[:,0], samples[:,1], c="orange", zorder=10)

        if show:
            plt.show()

    @functools.partial(jax.jit, static_argnums=2)
    def sample_from_floor(self, key, num_samples=1):
        if num_samples > 1:
            _, keys = split_key(key, num_samples)
            return jax.vmap(self.sample_from_floor)(keys).squeeze()

        probs = self.triangle_areas / self.triangle_areas.sum()

        key_tri, key_uv = jax.random.split(key)
        tri_indices = jax.random.choice(key_tri, self.triangles.shape[0], (1,), p=probs)
        chosen_tris = self.triangles[tri_indices]

        uvs = jax.random.uniform(key_uv, (1, 2))
        u = jnp.sqrt(uvs[:, 0])
        v = uvs[:, 1]
        w0 = 1 - u
        w1 = u * (1 - v)
        w2 = u * v
        weights = jnp.stack([w0, w1, w2], axis=-1)
        weights_expanded = weights[:, :, jnp.newaxis]

        sampled_points = jnp.sum(chosen_tris * weights_expanded, axis=1)
        return sampled_points  # Returns [1, 2] array

if __name__ == "__main__":
    curdir = "/".join(__file__.split("/")[:-1])
    # write_object_affordances_for_procthor(curdir)

    with open(curdir + "/object_data/test/0/objects.json", "r") as f:
        objects = json.load(f)
    x = objects[0]["oobb"]

    def get_surface_coords(object):
        oobb = objects[0]["oobb"]

        coords =


    JaxPolygon.create(objects[0])