from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, Point
from itertools import combinations
import networkx as nx
import geopandas as gpd
from artifacts.progress import GenerationProgress


class VGraph:
    
    """
    "An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles", Thomas Lozano-Perez (1983)
    
    https://lis.csail.mit.edu/pubs/tlp/collision-free-planning-cacm.pdf
    """
    
    def __init__(self, width, height, free_space, obstacle_space, occupancy, agv_radius,
                 access_points, io_points, buffer_segments, recorder=None):
        self.width = width
        self.height = height
        self.free_space = free_space
        self.obstacle_space = obstacle_space
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.buffer_segments = buffer_segments
        self.access_points = access_points
        self.io_points = io_points
        self.recorder = recorder

        self.G = nx.Graph()
        
        self.number_of_nodes = self.G.number_of_nodes
        self.number_of_edges = self.G.number_of_edges


    def _get_convex_vertices(self, polygon):
        convex_vertices = []
        coords = list(polygon.exterior.coords)[:-1]  # remove duplicate closing point
        for i in range(len(coords)):
            prev_vertex = coords[i - 1]
            curr_vertex = coords[i]
            next_vertex = coords[(i + 1) % len(coords)]
            vector_a = (curr_vertex[0] - prev_vertex[0], curr_vertex[1] - prev_vertex[1])
            vector_b = (next_vertex[0] - curr_vertex[0], next_vertex[1] - curr_vertex[1])
            cross_product = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
            if cross_product < 0:  # clockwise turn indicates convex vertex
                convex_vertices.append(curr_vertex)
        return convex_vertices

    def create_visibility_graph(self):

        vertex_progress = GenerationProgress("Visibility graph", "Collecting vertices")
        vertex_progress.start()

        # check if obstacle_space is a MultiPolygon, else no .geoms attribute
        if self.obstacle_space.is_empty:
            obstacle_polygons = []
        elif self.obstacle_space.geom_type == 'MultiPolygon':
            obstacle_polygons = list(self.obstacle_space.geoms)
        else:
            obstacle_polygons = [self.obstacle_space]
            
        # collect all boundary vertices
        vertices = []
        for poly in obstacle_polygons:
            # exclude concave vertices
            convex_vertices = self._get_convex_vertices(poly)
            vertices.extend(convex_vertices)
        
        # add obstacle vertices as nodes
        for i, v in enumerate(vertices):
            self.G.add_node(i, pos=v, type="obstacle")
        
        # add IO points as nodes
        for i, (x, y) in enumerate(self.io_points, start=len(vertices)):
            self.G.add_node(i, pos=(x, y), type="io")
    
        # add access points as nodes
        for i, (x, y) in enumerate(self.access_points, start=len(vertices) + len(self.io_points)):
            self.G.add_node(i, pos=(x, y), type="access")
            # print(f"Added access point node {i} at position {(x, y)}")
    
        num_nodes_total = len(vertices) + len(self.io_points) + len(self.access_points)
        vertex_progress.finish(
            True,
            f"collected {len(vertices)} obstacle vertices | total nodes: {num_nodes_total}",
        )
        
        total_pairs = num_nodes_total * (num_nodes_total - 1) // 2
        visibility_progress = GenerationProgress("Visibility graph", "Checking visibility")
        visibility_progress.start(f"target: {total_pairs} node pairs")
        # check visibility
        for pair_number, (i, j) in enumerate(combinations(range(num_nodes_total), 2), start=1):
            if pair_number % 100 == 0 or pair_number == total_pairs:
                visibility_progress.update(pair_number, total_pairs, f"checking nodes {i} and {j}")
            
            a = Point(self.G.nodes[i]["pos"])
            b = Point(self.G.nodes[j]["pos"])
            line = LineString([a, b])
    
            # if line inside free space, add edge with weight equal length to graph
            if self.free_space.covers(line):
                self.G.add_edge(i, j, weight=line.length)

        visibility_progress.finish(
            True,
            f"generated {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges",
        )

    def record_graph(self):
        """Record the visibility graph using via recorder if available."""
        if self.recorder is None:
            return None

        return self.recorder.save(
            data_type="vgraph",
            data={
                "occupancy": self.occupancy,
                "agv_radius": self.agv_radius,
                "buffer_segments": self.buffer_segments,
                "access_points": self.access_points,
                "io_points": self.io_points,
                "bounds": (0, 0, self.width, self.height),
                "nodes": {
                    node: {"pos": tuple(attributes["pos"]), "type": attributes["type"]}
                    for node, attributes in self.G.nodes(data=True)
                },
                "edges": sorted(tuple(sorted(edge)) for edge in self.G.edges),
            },
            plot=True,
        )
                
# def main():
    
#     from model.store import load_scenario_as_shapely
#     scenario = load_scenario_as_shapely()
    
#     access_points = [(point[1], point[0]) for point in scenario["access_points"]] # direction not required, swap ccords for shapely
#     io_points = [(point[1], point[0]) for point in scenario["io_points"]]
    
#     vgraph = VGraph(
#         width=scenario["width"],
#         height=scenario["height"],
#         free_space=scenario["free_space"],
#         obstacle_space=scenario["obstacle_space"],
#         occupancy=scenario["occupancy"],
#         agv_radius=scenario["agv_radius"],
#         access_points=access_points,
#         io_points=io_points
#     )
#     vgraph.create_visibility_graph()
    
#     print("Nodes:", vgraph.number_of_nodes())
#     print("Edges:", vgraph.number_of_edges())
    
#     vgraph.visualize()


# if __name__ == "__main__":
#     main()
