from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, Point
from itertools import combinations
import networkx as nx
import geopandas as gpd


class VGraph:
    def __init__(self, width, height, free_space, obstacle_space, occupancy, agv_radius, io_points, access_points):
        self.width = width
        self.height = height
        self.free_space = free_space
        self.obstacle_space = obstacle_space
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.io_points = io_points
        self.access_points = access_points

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
            print(f"{i*100/num_nodes_total:.2f}% - Checking visibility between {i} and {j}", end="\r")
            a = Point(self.G.nodes[i]["pos"])
            b = Point(self.G.nodes[j]["pos"])
            line = LineString([a, b])
    
            # if line inside free space, add edge with weight equal length to graph
            if self.free_space.covers(line):
                self.G.add_edge(i, j, weight=line.length)
                
    def visualize(self, title = None):
        _, ax = plt.subplots(figsize=(25, 20))

        LIGHT_GREY = (0.8, 0.8, 0.8)

        # draw obstacle/free space first (light grey)
        gpd.GeoSeries(self.obstacle_space).plot(ax=ax, color=LIGHT_GREY, zorder=0)
        gpd.GeoSeries(self.free_space).plot(ax=ax, color="white", zorder=0)
        
        # draw wall buffer
        ax.add_patch(plt.Rectangle((0, 0), self.width, self.agv_radius, color=LIGHT_GREY, zorder=0))
        ax.add_patch(plt.Rectangle((0, self.height - self.agv_radius), self.width, self.agv_radius, color=LIGHT_GREY, zorder=0))
        ax.add_patch(plt.Rectangle((0, 0), self.agv_radius, self.height, color=LIGHT_GREY, zorder=0))
        ax.add_patch(plt.Rectangle((self.width - self.agv_radius, 0), self.agv_radius, self.height, color=LIGHT_GREY, zorder=0))

        # draw occupancy grid on top
        # [free space (transparent), simple obstacle (dark grey), unit load (grey)]
        cmap = ListedColormap([(0, 0, 0, 0), (0.3, 0.3, 0.3, 1), (0.5, 0.5, 0.5, 1)])
        ax.pcolormesh(self.occupancy, cmap=cmap, edgecolors="black", linewidth=0.5, shading="flat", zorder=1)
        
        # draw visibility graph edges
        for u, v in self.G.edges:
            x1, y1 = self.G.nodes[u]["pos"]
            x2, y2 = self.G.nodes[v]["pos"]
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=1, zorder=2)


        # access points
        ax.scatter([p[0] for p in self.access_points], [p[1] for p in self.access_points], s=30, color="red", zorder=3)

        # IO points
        ax.scatter([p[0] for p in self.io_points], [p[1] for p in self.io_points], s=50, color="green", zorder=3)

        # grid boundaries
        for x in range(self.width + 1):
            ax.axvline(x, color="black", linewidth=1, zorder=4)
        for y in range(self.height + 1):
            ax.axhline(y, color="black", linewidth=1, zorder=4)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.margins(0)
        ax.invert_yaxis()

        plt.title(title if title else "Visibility Graph")
        plt.show()


def main():
    
    from load_scenario import load_scenario_as_shapely
    # free_space, obstacle_space, occupancy, occupancy_rate, agv_radius, io_points, access_points, width, height = load_scenario_as_shapely()
    scenario = load_scenario_as_shapely()
    
    vgraph = VGraph(
        width=scenario["width"],
        height=scenario["height"],
        free_space=scenario["free_space"],
        obstacle_space=scenario["obstacle_space"],
        occupancy=scenario["occupancy"],
        agv_radius=scenario["agv_radius"],
        io_points=scenario["io_points"],
        access_points=scenario["access_points"]
    )
    vgraph.create_visibility_graph()
    
    print("Nodes:", vgraph.number_of_nodes())
    print("Edges:", vgraph.number_of_edges())
    
    vgraph.visualize()


if __name__ == "__main__":
    main()
