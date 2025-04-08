import copy
import dataclasses
import functools

import networkx as nx
import numpy as np
import jax.numpy as jnp
import jax

from hippo.simulation.spatialutils.filter_positions import filter_reachable_positions


def astar(start_node, end_node, full_reachability_graph, runtime_container):
    filtered = filter_reachable_positions(full_reachability_graph, runtime_container)

    def dist(p1, p2):
        return jnp.linalg.norm(p1 - p2)

    all_nodes = jnp.array(filtered.nodes)
    def shunt(pos):
        pos = jnp.array(pos)

        argret = jax.vmap(functools.partial(dist, p2=pos))(all_nodes).argmin()
        return tuple(np.array(all_nodes[argret]).tolist())

    start_node = shunt(start_node)
    end_node = shunt(end_node)


    path = nx.astar_path(filtered, start_node, end_node)
    for node in path:
        yield node

class AStar:
    def __init__(self, end_node, full_reachability_graph, runtime_container):
        self.end_node = end_node
        self.full_reachability_graph = copy.deepcopy(full_reachability_graph)
        self.runtime_container = runtime_container

        self.generated_nodes = []

    def generate_path(self, start_node):
        full_reachability_graph = copy.deepcopy(self.full_reachability_graph)
        full_reachability_graph.remove_nodes_from(self.generated_nodes)

        filtered = filter_reachable_positions(self.full_reachability_graph, self.runtime_container)

        def dist(p1, p2):
            return jnp.linalg.norm(p1 - p2)

        all_nodes = jnp.array(filtered.nodes)

        def shunt(pos):
            pos = jnp.array(pos)

            argret = jax.vmap(functools.partial(dist, p2=pos))(all_nodes).argmin()
            return tuple(np.array(all_nodes[argret]).tolist())

        start_node = shunt(start_node)
        end_node = shunt(self.end_node)

        path = nx.astar_path(filtered, start_node, end_node)
        for node in path:
            self.generated_nodes.append(node)
            yield node

