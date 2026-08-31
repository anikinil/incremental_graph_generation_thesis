from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, Point, box
from itertools import combinations
import networkx as nx
import geopandas as gpd
from artifacts.progress import GenerationProgress
from scenario.geometry import navigation_space


class VGraph:
    
    """
    "An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles", Thomas Lozano-Perez (1983)
    
    https://lis.csail.mit.edu/pubs/tlp/collision-free-planning-cacm.pdf
    """
    
    def __init__(self, width, height, occupancy, agv_radius,
                 access_points, io_points, num_buffer_segments, recorder=None):
        self.width = width
        self.height = height
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.num_buffer_segments = num_buffer_segments
        self.access_points = access_points
        self.io_points = io_points
        self.recorder = recorder
        self.iteration = 0

        self.G = nx.Graph()

        self.number_of_vertices = self.G.number_of_nodes
        self.number_of_edges = self.G.number_of_edges


    def _get_convex_vertices(self, polygon):
        """Get all convex vertices of a polygon."""
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

        vertex_progress = GenerationProgress("Visibility graph", "Collecting vertices", self.iteration)
        vertex_progress.start()

        free_space, obstacle_space = navigation_space(self.occupancy, self.agv_radius, self.num_buffer_segments)

        # check if obstacle_space is a MultiPolygon, else no .geoms attribute
        if obstacle_space.is_empty:
            obstacle_polygons = []
        elif obstacle_space.geom_type == 'MultiPolygon':
            obstacle_polygons = list(obstacle_space.geoms)
        else:
            obstacle_polygons = [obstacle_space]
            
        # collect all convex boundary vertices of the obstacle polygons
        vertices = []
        for poly in obstacle_polygons:
            # exclude concave vertices
            convex_vertices = self._get_convex_vertices(poly)
            vertices.extend(convex_vertices)
        
        # add obstacle vertices as vertices
        for i, v in enumerate(vertices):
            self.G.add_node(i, pos=v, type="obstacle")
        
        # add IO points as vertices
        for i, (x, y) in enumerate(self.io_points, start=len(vertices)):
            self.G.add_node(i, pos=(x, y), type="io")
    
        # add access points as vertices
        for i, (x, y) in enumerate(self.access_points, start=len(vertices) + len(self.io_points)):
            self.G.add_node(i, pos=(x, y), type="access")
            # print(f"Added access point node {i} at position {(x, y)}")
    
        num_vertices_total = len(vertices) + len(self.io_points) + len(self.access_points)
        vertex_progress.finish(True,
            f"collected {len(vertices)} obstacle vertices | total vertices: {num_vertices_total}")
        
        total_pairs = num_vertices_total * (num_vertices_total - 1) // 2
        visibility_progress = GenerationProgress("Visibility graph", "Checking visibility", self.iteration)
        visibility_progress.start(f"target: {total_pairs} node pairs")
        # check visibility
        for pair_number, (i, j) in enumerate(combinations(range(num_vertices_total), 2), start=1):
            if pair_number % 100 == 0 or pair_number == total_pairs:
                visibility_progress.update(pair_number, total_pairs, f"checking vertices {i} and {j}")
            
            a = Point(self.G.nodes[i]["pos"])
            b = Point(self.G.nodes[j]["pos"])
            line = LineString([a, b])
    
            # if line inside free space, add edge with weight equal length to graph
            if free_space.covers(line):
                self.G.add_edge(i, j)

        visibility_progress.finish(True, f"generated {self.G.number_of_nodes()} vertices and {self.G.number_of_edges()} edges")

    def record_graph(self):
        """Record the visibility graph using via recorder if available."""
        if self.recorder is None:
            return None

        save_progress = GenerationProgress("Visibility graph", "Saving graph to disk", self.iteration)
        save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="vgraph",
                data={
                    "height": self.height,
                    "width": self.width,
                    "occupancy": self.occupancy,
                    "agv_radius": self.agv_radius,
                    "num_buffer_segments": self.num_buffer_segments,
                    "access_points": self.access_points,
                    "io_points": self.io_points,
                    "bounds": (0, 0, self.width, self.height),
                    "vertices": {
                        vertex: {"pos": tuple(attributes["pos"]), "type": attributes["type"]}
                        for vertex, attributes in self.G.nodes(data=True)
                    },
                    "edges": sorted(tuple(sorted(edge)) for edge in self.G.edges),
                },
                plot=True,
            )
        except Exception as error:
            save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts
