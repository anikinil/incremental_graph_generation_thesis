import numpy as np
from math import ceil

from scenario.geometry import navigation_space


class DynamicVGraph:
    def __init__(self, height, width, occupancy, agv_radius, buffer_segments, access_points, io_points, recorder=None):
        
        self.height = height
        self.width = width
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.buffer_segments = buffer_segments
        self.access_points = access_points
        self.io_points = io_points
        self.recorder = recorder
        
        self.vertices = []
        self.edges = []
        
        # mappings between grid cells and graph edges
        self.cell_to_edge = {}
        self.edge_to_cells = {}
        
        self.edge_active_status = {}  # dictionary to track the active status of edges
        
    
    def _extract_convex_vertices(self, obstacle_space):
        """Get all convex vertices of the obstacle space."""
        
        def _extract_convex_vertices_from_polygon(polygon):
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
            convex_vertices = _extract_convex_vertices_from_polygon(poly)
            vertices.extend(convex_vertices)
    
        return vertices
    
    def _determine_edge_cells(self, vertex1, vertex2):
        """Returns cells crossed by a 2D Amanatides-Woo traversal.
        
        "A Fast Voxel Traversal Algorithm for Ray Tracing", John Amanatides and Andrew Woo (1987)
        
        https://m4xc.dev/articles/amanatides-and-woo/
        https://gitlab.com/athilenius/fast-voxel-traversal-rs/-/blob/main/src/raycast_2d.rs?ref_type=heads
        """

        x0, y0 = vertex1 # starting point of the ray
        x1, y1 = vertex2 # ending point of the ray
        dx = x1 - x0 # total change in x
        dy = y1 - y0 # total change in y

        current_x = int(x0) # current cell x-coordinate
        current_y = int(y0) # current cell y-coordinate
        end_x = int(x1) # end cell x-coordinate
        end_y = int(y1) # end cell y-coordinate

        # set x-varibales
        if dx > 0:
            step_x = 1 # step of one cell in the positive x direction
            t_delta_x = 1 / dx # how far along the ray we must move for the x-coordinate to change by 1
            t_max_x = (current_x + 1 - x0) / dx # how far along the ray we must move for the first x-boundary to be crossed
        elif dx < 0:
            step_x = -1
            t_delta_x = 1 / -dx
            t_max_x = (current_x - x0) / dx
        else: # dx == 0: the ray is vertical, x-boundaries will never be crossed
            step_x = 0
            t_delta_x = float("inf")
            t_max_x = float("inf")

        # set y-variables
        if dy > 0:
            step_y = 1
            t_delta_y = 1 / dy
            t_max_y = (current_y + 1 - y0) / dy
        elif dy < 0:
            step_y = -1
            t_delta_y = 1 / -dy
            t_max_y = (current_y - y0) / dy
        else: # dy == 0: the ray is horizontal, y-boundaries will never be crossed
            step_y = 0
            t_delta_y = float("inf")
            t_max_y = float("inf")

        # traverse the grid until the end cell is reached starting from the first cell
        cells = [(current_y, current_x)] # (y, x) for row, column indexing
        while current_x != end_x or current_y != end_y:
            if t_max_x < t_max_y: # if the next x-boundary is closer than the next y-boundary, step in x direction
                current_x += step_x # step to the next cell in the x direction
                t_max_x += t_delta_x # move next x-boundary along the ray
            else: # if the next y-boundary is closer than the next x-boundary, step in y direction
                current_y += step_y
                t_max_y += t_delta_y
            cells.append((current_y, current_x)) # add traversed cell ((y, x) for row, column indexing)

        return cells


    def _deactivate_edges_in_obstacle_cells(self):
        """Deactivate edges that cross obstacle_cells."""
        
        for row in range(self.height):
            for col in range(self.width):
                if self.occupancy[row, col] != 0: # obstacle cell
                    cell = (row, col)
                    if cell in self.cell_to_edge:
                        for edge in  self.cell_to_edge[cell]:
                            self.edge_active_status[edge] = False # deactivate edge
    

    def preprocessing(self):

        # get all convex vertices of the obstacle polygons
        free_space, obstacle_space = navigation_space(self.occupancy, self.agv_radius, self.buffer_segments)
        vertices = self._extract_convex_vertices(obstacle_space) + self.io_points + self.access_points
        
        # add access points and io points to the vertex list
        vertices = vertices + self.io_points + self.access_points
                
        # for each vertex pair determine which cells the edge between them crosses
        # store mappings between cells and edges in dictionaries for fast lookup
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)): # order vertices by index to avoid duplicate edges
                edge = (vertices[i], vertices[j])
                cells = self._determine_edge_cells(edge[0], edge[1])
                self.edge_to_cells[edge] = cells
                for cell in cells:
                    if cell not in self.cell_to_edge:
                        self.cell_to_edge[cell] = []
                    self.cell_to_edge[cell].append(edge)
                    self.edge_active_status[edge] = True # initialize edge as active

        # mark edges that cross obstacle cells as inactive in edge_active_status
        self._deactivate_edges_in_obstacle_cells()

        # construct the graph from the vertices and active edges
        for vertex in vertices:
            self.vertices.append(vertex)
        for edge in self.edge_to_cells.keys():
            if self.edge_active_status[edge]:
                self.edges.append(edge)

    
    
    def record_graph(self, phase="manual"):
        """Save the roadmap for the plotter."""
        
        pass

        
