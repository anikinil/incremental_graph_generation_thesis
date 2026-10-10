from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from itertools import combinations
import networkx as nx
import geopandas as gpd
from artifacts.progress import ProgressReportingAlgorithm
from algorithms.geometry import navigation_space
from shapely.geometry import LineString

class VGraph(ProgressReportingAlgorithm):
    """
    "An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles", Thomas Lozano-Perez (1983)
    
    https://lis.csail.mit.edu/pubs/tlp/collision-free-planning-cacm.pdf
    """
    
    # TODO use the same corner translation method as in the paper
    
    progress_name = "Visibility graph"

    def __init__(self, width, height, occupancy, robot_radius,
                 access_points, interaction_points, num_buffer_segments, recorder=None,
                 print_progress=True):
        self.width = width
        self.height = height
        self.occupancy = occupancy
        self.robot_radius = robot_radius
        self.num_buffer_segments = num_buffer_segments
        self.access_points = access_points
        self.interaction_points = interaction_points
        self.recorder = recorder
        self.iteration = 0
        self.print_progress = bool(print_progress)

        self.G = nx.Graph()

        self.number_of_nodes = self.G.number_of_nodes
        self.number_of_edges = self.G.number_of_edges

    def _get_convex_vertices(self, polygon):
        """Get all convex vertices of a polygon."""
        
        convex_vertices = []
        coords = list(polygon.exterior.coords)[:-1]  # remove duplicate closing point
        # direction of the polygon is counter-clockwise, so a clockwise turn indicates a convex vertex
        for i in range(len(coords)):
            prev_vertex = coords[i - 1]
            curr_vertex = coords[i]
            
            # check if curr_vertex touches the wall
            if (curr_vertex[0] == 0 or curr_vertex[0] == self.width or
                curr_vertex[1] == 0 or curr_vertex[1] == self.height):
                continue  # skip vertices that touch the wall
            
            next_vertex = coords[(i + 1) % len(coords)]
            vector_a = (curr_vertex[0] - prev_vertex[0], curr_vertex[1] - prev_vertex[1])
            vector_b = (next_vertex[0] - curr_vertex[0], next_vertex[1] - curr_vertex[1])
            cross_product = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
            if cross_product < 0:  # clockwise turn indicates convex vertex
                convex_vertices.append(curr_vertex)
        return convex_vertices


    def create_visibility_graph(self):

        node_progress = self._create_progress("Collecting nodes")
        if node_progress is not None:
            node_progress.start()
            
        start_time = self._start_generation_time()

        free_space, obstacle_space = navigation_space(self.occupancy, self.robot_radius, self.num_buffer_segments)

        # check if obstacle_space is a MultiPolygon, else no .geoms attribute
        if obstacle_space.is_empty:
            obstacle_polygons = []
        elif obstacle_space.geom_type == 'MultiPolygon':
            obstacle_polygons = list(obstacle_space.geoms)
        else:
            obstacle_polygons = [obstacle_space]
            
        # collect all convex boundary vertices of the obstacle polygons
        nodes = []
        for poly in obstacle_polygons:
            # exclude concave vertices
            convex_vertices = self._get_convex_vertices(poly)
            nodes.extend(convex_vertices)
        
        # add obstacle nodes as nodes
        for i, v in enumerate(nodes):
            self.G.add_node(i, pos=v, type="obstacle")
        
        # add interaction points as nodes
        for i, (x, y) in enumerate(self.interaction_points, start=len(nodes)):
            self.G.add_node(i, pos=(x, y), type="interaction")
    
        # add access points as nodes
        for i, (x, y) in enumerate(self.access_points, start=len(nodes) + len(self.interaction_points)):
            self.G.add_node(i, pos=(x, y), type="access")
            # print(f"Added access point node {i} at position {(x, y)}")
    
        num_nodes_total = len(nodes) + len(self.interaction_points) + len(self.access_points)
        if node_progress is not None:
            node_progress.finish(True,
                f"collected {len(nodes)} obstacle nodes | total nodes: {num_nodes_total}")
        
        visibility_progress = self._create_progress("Checking visibility")
        if visibility_progress is not None:
            total_pairs = num_nodes_total * (num_nodes_total - 1) // 2
            visibility_progress.start(f"target: {total_pairs} node pairs")
        # Keep the uninstrumented path free of progress counters and checks.
        if visibility_progress is not None:
            for pair_number, (i, j) in enumerate(combinations(range(num_nodes_total), 2), start=1):
                if pair_number % 100 == 0 or pair_number == total_pairs:
                    visibility_progress.update(pair_number, total_pairs, f"checking nodes {i} and {j}")

                line = LineString([self.G.nodes[i]['pos'], self.G.nodes[j]['pos']])
                if free_space.covers(line):
                    self.G.add_edge(i, j)
        else:
            for i, j in combinations(range(num_nodes_total), 2):
                line = LineString([self.G.nodes[i]['pos'], self.G.nodes[j]['pos']])
                if free_space.covers(line):
                    self.G.add_edge(i, j)

        if visibility_progress is not None:
            visibility_progress.finish(True, f"generated {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges", completed=total_pairs, total=total_pairs)

        self._record_generation_time(self._generation_seconds(start_time), "generation")

    def record_graph(self):
        """Record the visibility graph using the recorder if available."""
        if self.recorder is None:
            return None

        save_progress = self._create_progress("Saving graph to disk")
        if save_progress is not None:
            save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="vgraph",
                data={
                    "height": self.height,
                    "width": self.width,
                    "occupancy": self.occupancy,
                    "robot_radius": self.robot_radius,
                    "num_buffer_segments": self.num_buffer_segments,
                    "access_points": self.access_points,
                    "interaction_points": self.interaction_points,
                    "bounds": (0, 0, self.width, self.height),
                    "nodes": {
                        node: {"pos": tuple(attributes["pos"]), "type": attributes["type"]}
                        for node, attributes in self.G.nodes(data=True)
                    },
                    "edges": sorted(tuple(sorted(edge)) for edge in self.G.edges),
                },
                plot=True,
            )
        except Exception as error:
            if save_progress is not None:
                save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            if save_progress is not None:
                save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        if save_progress is not None:
            save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts
