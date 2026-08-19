from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, Point
from itertools import combinations
import networkx as nx
import geopandas as gpd


class VGraph:
    
    """
    "An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles", Thomas Lozano-Perez (1983)
    
    https://lis.csail.mit.edu/pubs/tlp/collision-free-planning-cacm.pdf
    """
    
    def __init__(self, width, height, free_space, obstacle_space, occupancy, agv_radius, access_points, io_points):
        self.width = width
        self.height = height
        self.free_space = free_space
        self.obstacle_space = obstacle_space
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.access_points = access_points
        self.io_points = io_points

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

        # check if obstacle_space is a MultiPolygon, else no .geoms attribute
        if self.obstacle_space.geom_type == 'MultiPolygon':
            obstacle_polygons = list(self.obstacle_space.geoms)
        else:
            obstacle_polygons = [self.obstacle_space]
            
        print(f"Collecting vertices...")
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
        
        print(f"Checking visibility..")
        # check visibility
        for i, j in combinations(range(num_nodes_total), 2):
            print(f"\r{' ' * 100}\r{i*100/num_nodes_total:.2f}% - Checking visibility between {i} and {j}", end="", flush=True)
            a = Point(self.G.nodes[i]["pos"])
            b = Point(self.G.nodes[j]["pos"])
            line = LineString([a, b])
    
            # if line inside free space, add edge with weight equal length to graph
            if self.free_space.covers(line):
                self.G.add_edge(i, j, weight=line.length)
                
def main():
    
    from model.store import load_scenario_as_shapely
    scenario = load_scenario_as_shapely()
    
    access_points = [(point[1], point[0]) for point in scenario["access_points"]] # direction not required, swap ccords for shapely
    io_points = [(point[1], point[0]) for point in scenario["io_points"]]
    
    vgraph = VGraph(
        width=scenario["width"],
        height=scenario["height"],
        free_space=scenario["free_space"],
        obstacle_space=scenario["obstacle_space"],
        occupancy=scenario["occupancy"],
        agv_radius=scenario["agv_radius"],
        access_points=access_points,
        io_points=io_points
    )
    vgraph.create_visibility_graph()
    
    print("Nodes:", vgraph.number_of_nodes())
    print("Edges:", vgraph.number_of_edges())
    
    vgraph.visualize()


if __name__ == "__main__":
    main()
