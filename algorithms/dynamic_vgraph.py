import numpy as np
from math import ceil

from artifacts.progress import GenerationProgress
from scenario.geometry import navigation_space


class DynamicVGraph:
    def __init__(self, height, width, occupancy, agv_radius, access_points, io_points, recorder=None):
        
        self.height = height
        self.width = width
        self.occupancy = occupancy
        self.agv_radius = agv_radius
        self.num_buffer_segments = 0 # always 0, since it's the only way to avoid collisions (temporary)
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
        """Returns grid cells crossed by the buffered edge between two vertices."""
        
        # offset the vertices in the positive and negativ directions of the edge normal by the AGV radius to get the buffered edge
        x1, y1 = vertex1
        x2, y2 = vertex2
        edge_vector = np.array([x2 - x1, y2 - y1])
        edge_normal = np.array([-edge_vector[1], edge_vector[0]]) / np.linalg.norm(edge_vector)  # normalized edge normal
        offset_vector = edge_normal * self.agv_radius
        start1 = (x1 + offset_vector[0], y1 + offset_vector[1])
        start2 = (x1 - offset_vector[0], y1 - offset_vector[1])
        end1 = (x2 + offset_vector[0], y2 + offset_vector[1])
        end2 = (x2 - offset_vector[0], y2 - offset_vector[1])
        
        def _amanatides_woo(start, end):
            """Returns cells crossed by a 2D Amanatides-Woo traversal.
            
            "A Fast Voxel Traversal Algorithm for Ray Tracing", John Amanatides and Andrew Woo (1987)
            
            https://m4xc.dev/articles/amanatides-and-woo/
            https://gitlab.com/athilenius/fast-voxel-traversal-rs/-/blob/main/src/raycast_2d.rs?ref_type=heads
            """

            x0, y0 = start # starting point of the ray
            x1, y1 = end # ending point of the ray
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

            cells = [(current_y, current_x)] # (y, x) for row, column indexing
            remaining_steps = abs(end_x - current_x) + abs(end_y - current_y)
            for _ in range(remaining_steps):
                # if the next x-boundary is closer than the next y-boundary or the current cell is in the same column as the end cell
                if t_max_y < t_max_x or current_x == end_x:
                    current_y += step_y # step in the y direction
                    t_max_y += t_delta_y # update the next y-boundary crossing distance
                # if the next y-boundary is closer than the next x-boundary or the current cell is in the same row as the end cell
                elif t_max_x < t_max_y or current_y == end_y:
                    current_x += step_x # step in the x direction
                    t_max_x += t_delta_x # update the next x-boundary crossing distance
                cells.append((current_y, current_x)) # add traversed cell ((y, x) for row, column indexing)

            return cells
        
        # get the grid cells crossed by the two offset edges using Amanatides-Woo traversal
        cells1 = _amanatides_woo(start1, end1)
        cells2 = _amanatides_woo(start2, end2)
        
        return list(set(cells1 + cells2))  # return unique cells crossed by the buffered edge


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

        vertex_progress = GenerationProgress("Dynamic visibility graph", "Collecting vertices")
        vertex_progress.start()

        # get all convex vertices of the obstacle polygons
        free_space, obstacle_space = navigation_space(self.occupancy, self.agv_radius, num_buffer_segments=self.num_buffer_segments)
        vertices = list(dict.fromkeys(self._extract_convex_vertices(obstacle_space) + list(self.io_points) + list(self.access_points)))

        vertex_progress.finish(True, f"collected {len(vertices)} visibility vertices")

        total_pairs = len(vertices) * (len(vertices) - 1) // 2
        edge_progress = GenerationProgress("Dynamic visibility graph", "Mapping candidate edges")
        edge_progress.start(f"target: {total_pairs} vertex pairs")
                
        # for each vertex pair determine which cells the edge between them crosses
        # store mappings between cells and edges in dictionaries for fast lookup
        pair_number = 0
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

                # update progress
                pair_number += 1
                if pair_number % 100 == 0:
                    edge_progress.update(pair_number, total_pairs)

        # mark edges that cross obstacle cells as inactive in edge_active_status
        self._deactivate_edges_in_obstacle_cells()

        # construct the graph from the vertices and active edges
        for vertex in vertices:
            self.vertices.append(vertex)
        for edge in self.edge_to_cells.keys():
            if self.edge_active_status[edge]:
                self.edges.append(edge)

        edge_progress.finish(True, f"mapped {len(self.edge_to_cells)} candidate edges | active: {len(self.edges)}")

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_io_points):
        pass
    
    def record_graph(self, phase="manual"):
        """Record all visibility candidates and their current active status."""
        if self.recorder is None:
            return None

        edges = sorted(self.edge_to_cells)
        return self.recorder.save(
            data_type="dynamic_vgraph",
            data={
                "occupancy": self.occupancy.copy(),
                "agv_radius": self.agv_radius,
                "num_buffer_segments": self.num_buffer_segments,
                "access_points": sorted(self.access_points),
                "io_points": sorted(self.io_points),
                "height": self.height,
                "width": self.width,
                "vertices": sorted({tuple(vertex) for vertex in self.vertices}),
                "edges": edges,
                "edge_active": {
                    edge: bool(self.edge_active_status[edge])
                    for edge in edges
                },
                "mapped_arc_cells": sum(len(edge_set) for edge_set in self.cell_to_edge.values()),
                "phase": phase,
            },
            plot=True,
        )

        
