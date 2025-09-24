import copy
import dataclasses
import functools
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
class FloorPolygon:
    scene_room_id: int
    scene_room_floorPolygon: list[dict]
    triangles: jnp.ndarray
    triangle_areas: jnp.ndarray

    @singledispatchmethod
    @classmethod
    def create(cls, arg):
        raise NotImplementedError()

    @create.register(int)
    @classmethod
    def _(cls, id, room):
        self = FloorPolygon(id, room, None, None)

        polygon = Polygon(self.floor_polygon_coords)
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

    @create.register(dict)
    @classmethod
    def _(cls, dict):
        rooms = dict["rooms"]

        MAX_NUM_TRIANGLES = -jnp.inf
        selves = []
        for room in rooms:
            self = FloorPolygon.create(room)
            MAX_NUM_TRIANGLES = max(MAX_NUM_TRIANGLES, self.numtriangles)
            selves.append(self)
        if len(selves) != 1:
            raise NotImplementedError() # todo pad triangles with zero-surface triangles to make vmap play nice
        return selves[0]

    @property
    def num_triangles(self):
        return self.triangles.shape[0]

    @property
    def floor_bbox(self):
        coords = self.floor_polygon_coords

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

    #@property
    def networkx(self, grid_size=0.05):
        min_x, min_y, max_x, max_y = self.floor_bbox

        x = np.arange(min_x + grid_size * 2, max_x - grid_size, grid_size)
        y = np.arange(min_y + grid_size * 2, max_y - grid_size, grid_size)

        DEFAULT_ROBOT_HEIGHT = 0.95

        # Create 2D grid
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1, dtype=np.float32)
        grid_points = [(x, DEFAULT_ROBOT_HEIGHT, z) for x, z in grid_points]

        mask = self.containment_mask(grid_points)
        grid_points = [p for i, p in enumerate(grid_points) if mask[i]]
        mask = self.points_distance_to_polygon_edge_mask(grid_points, 0.3)
        grid_points = [p for i, p in enumerate(grid_points) if mask[i]]
        grid_points = [(z,y,x) for (x,y,z) in grid_points]

        G = nx.Graph()
        points_set = set(grid_points)

        # Add all points as nodes
        for p in points_set:
            G.add_node(p)

        # Define neighbor directions (axis-aligned)
        DIM = 3
        directions = []
        for d in range(DIM):
            if d == 1:
                continue
            for offset in [-grid_size, grid_size]:
                dir_vector = [0] * DIM
                dir_vector[d] = offset
                directions.append(tuple(dir_vector))

        # Add edges between neighbors
        for p in points_set:
            for d in directions:
                neighbor = tuple(np.add(p, d))
                if neighbor in points_set:
                    G.add_edge(p, neighbor)

        return G

    @property
    def floor_polygon_coords(self):
        return list(map(lambda x: (x["x"], x["z"]), self.scene_room_floorPolygon))

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

@struct.dataclass
class ReachabilityGraph:
    full_graph: nx.Graph

    def filter_(self, runtime_container):
        def is_inside_aabb(obj_pos, obj_size, agent_pos):
            """
            Check if an agent at `agent_pos` is inside the AABB defined by `obj_pos` and `obj_size`.
            Returns True if inside, False if outside.
            """
            # Only check x (0) and z (2) dimensions, skip y (1)
            x_inside = jnp.logical_and(
                obj_pos[0] - obj_size[0] <= agent_pos[0],
                agent_pos[0] <= obj_pos[0] + obj_size[0]
            )
            z_inside = jnp.logical_and(
                obj_pos[2] - obj_size[2] <= agent_pos[2],
                agent_pos[2] <= obj_pos[2] + obj_size[2]
            )
            return jnp.logical_and(x_inside, z_inside)

        @jax.jit
        def _filter_agent_positions(agent_positions, object_positions, object_sizes, margin=0.0):
            """
            Remove agent positions that are inside any object's AABB, considering a margin.
            """
            inflated_sizes = object_sizes + margin

            def is_inside_any_object(single_agent_pos, all_object_poses, all_object_sizes):
                return jax.vmap(functools.partial(is_inside_aabb, agent_pos=single_agent_pos))(all_object_poses,
                                                                                               all_object_sizes).any()

            mask = jax.vmap(functools.partial(is_inside_any_object, all_object_poses=object_positions,
                                              all_object_sizes=inflated_sizes))(agent_positions)
            return jnp.logical_not(mask)

        reachable_positions = copy.deepcopy(self.full_graph)
        todo_positions = reachable_positions.nodes

        centers = [obj.position for obj in runtime_container.objects_map.values()]
        sizes = [obj.size for obj in runtime_container.objects_map.values()]

        todo_positions = jnp.array(todo_positions)
        centers = jnp.array(centers)
        sizes = jnp.array(sizes)

        mask = _filter_agent_positions(todo_positions.at[:, 1].set(0.95), centers.at[:, 1].set(0.95), sizes,
                                       margin=0)

        print("Num positions to remove", jnp.sum(jnp.logical_not(mask)))

        mask = np.array(mask)
        todo_positions = np.array(todo_positions)


        for position in todo_positions[jnp.logical_not(mask)]:
            for node in nx.nodes(reachable_positions):
                if np.all(np.allclose(node, position)):
                    reachable_positions.remove_node(node)
                    break

        # Step 2: Keep only the largest connected component
        if reachable_positions.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(reachable_positions), key=len)
            reachable_positions = reachable_positions.subgraph(largest_cc_nodes).copy()

        return ReachabilityGraph(reachable_positions)
