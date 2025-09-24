import os

import shapely

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update('jax_platform_name', "cpu")
import prior

from groundtruth.floor import FloorPolygon

dataset = prior.load_dataset("procthor-10k")
dataset

house = dataset["train"][0]
type(house), house.keys(), house

from shapely import Polygon
import matplotlib.pyplot as plt

floor = house["rooms"][0]["floorPolygon"]
coords = list(map(lambda x: (x["x"], x["z"]), floor))

floor = FloorPolygon.create(0, house["rooms"][0]["floorPolygon"])#.sample_from_floor()
#floor.plot_self()
samples = floor.sample_from_floor(jax.random.PRNGKey(0), num_samples=100)

networkx = floor.networkx()

floor.plot_self(samples, reachability_graph=networkx, show=True)
exit()


polygon = Polygon(coords)

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
        ret.append(tri)

    return ret
plt.plot(*polygon.exterior.xy)
plt.show()

tris = polygon_to_triangles(polygon)

for tri in tris:
    x, y = tri.exterior.xy
    plt.fill(x, y, alpha=0.4)  # alpha controls transparency

plt.show()



polygon.tri

plt.show()
exit()
polygon.area

from shapely.ops import triangulate

floor = FloorPolygon(0, house["rooms"][0]["floorPolygon"])#.sample_from_floor()
floor.plot_self()
samples = floor.sample_from_floor()
floor.plot_self(samples)
#plot_floor_polygons(house)

#samples = sample_from_floor(house, n_samples=20, seed=123)

#plot_floor_samples(house, samples)
#plt.show()
