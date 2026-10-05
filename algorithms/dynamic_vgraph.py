import numpy as np
from math import ceil

from artifacts.progress import GenerationProgress
from algorithms.geometry import buffered_edge_cells, navigation_space


class DynamicVGraph:
    """
    Dynamic Visibility Graph, with the method of activating/deactivating edges affected by dynamic obstacles, partially based on:
    
    "FAR Planner: Fast, Attemptable Route Planner using Dynamic Visibility Update"
    Yang, Cao, Zhu, Oh, and Zhang (2022)
    """
    
    
    def __init__(self, height, width, occupancy, agv_radius, access_points, io_points, recorder=None):
        
        self.height = height
        self.width = width
        self.occupancy = occupancy.copy()
        self.agv_radius = agv_radius
        self.num_buffer_segments = 0 # always 0, since it's the only way to avoid collisions (temporary)
        self.access_points = access_points
        self.io_points = io_points
        self.recorder = recorder
        
        self.vertices = []
        self.edges = []
        
        # mappings between grid cells, graph edges, and vertices
        self.cell_to_edge = {}
        self.edge_to_cells = {}
        self.vertex_to_edges = {}
        
        self.edge_active_status = {}  # dictionary to track the active status of edges

        self.iteration = 0
        
    
    def _extract_convex_vertices(self, obstacle_space):
        """Get all convex vertices of the obstacle space."""
        
        def _extract_convex_vertices_from_polygon(polygon):
            """Get all convex vertices of a polygon."""
            convex_vertices = []
            coords = list(polygon.exterior.coords)[:-1]  # remove duplicate closing point
            for i in range(len(coords)):
                prev_vertex = coords[i - 1]
                curr_vertex = coords[i]
                
                # check if curr_vertex touches the wall
                if (curr_vertex[0] == 0 or curr_vertex[0] == self.height or
                    curr_vertex[1] == 0 or curr_vertex[1] == self.width):
                    continue  # skip vertices that touch the wall
                
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

        return buffered_edge_cells(vertex1, vertex2, self.agv_radius, self.height, self.width)


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
        _, obstacle_space = navigation_space(self.occupancy, self.agv_radius, num_buffer_segments=self.num_buffer_segments)
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
                for vertex in edge:
                    if vertex not in self.vertex_to_edges:
                        self.vertex_to_edges[vertex] = set()
                    self.vertex_to_edges[vertex].add(edge)
                for cell in cells:
                    if cell not in self.cell_to_edge:
                        self.cell_to_edge[cell] = set()
                    self.cell_to_edge[cell].add(edge)
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
        """Apply layout updates and update the visibility graph."""

        # update the occupancy grid and collect added and removed obstacle cells
        added_cells = set()
        removed_cells = set()
        for update in obstacle_updates:
            action = update["action"]
            if action == "add":
                cell, value = update["cell"], update["obstacle_type"]
                self.occupancy[cell] = value
                added_cells.add(cell)
            elif action == "remove":
                cell = update["cell"]
                self.occupancy[cell] = 0
                removed_cells.add(cell)
            elif action == "move":
                cell_from, cell_to = update["from_cell"], update["to_cell"]
                value = self.occupancy[cell_from]
                self.occupancy[cell_from] = 0
                self.occupancy[cell_to] = value
                removed_cells.add(cell_from)
                added_cells.add(cell_to)
                
        print("APPLIED LAYOUT UPDATES")

        # activate edges affected by removed obstacle cells, if no remaining obstacle blocks them
        for cell in removed_cells:
            if cell in self.cell_to_edge:
                for edge in self.cell_to_edge[cell]:
                    if all(self.occupancy[edge_cell] == 0 for edge_cell in self.edge_to_cells[edge]):
                        self.edge_active_status[edge] = True

        self.access_points = new_access_points
        self.io_points = new_io_points

        print("UPDATED SPECIAL POINTS")

        # TODO make incremental, only look at relebvant parts instead of recomputing whole obstacle space
        # get all vertices of the updated obstacle space and special points
        _, obstacle_space = navigation_space(self.occupancy, self.agv_radius, num_buffer_segments=self.num_buffer_segments)
        new_vertices = list(dict.fromkeys(self._extract_convex_vertices(obstacle_space) + list(self.io_points) + list(self.access_points)))

        # remove vertices that are no longer part of the obstacle space or special points
        new_vertex_set = set(new_vertices)
        removed_vertices = set(self.vertices) - new_vertex_set
        print(f"REMOVING {len(removed_vertices)} VERTICES")
        removed_edges = set()
        for vertex in removed_vertices:
            removed_edges.update(self.vertex_to_edges.get(vertex, set()))

        for edge in removed_edges:
            cells = self.edge_to_cells.pop(edge)
            self.edge_active_status.pop(edge)
            for edge_vertex in edge:
                self.vertex_to_edges[edge_vertex].remove(edge)
                if not self.vertex_to_edges[edge_vertex]:
                    del self.vertex_to_edges[edge_vertex]
            for cell in cells:
                self.cell_to_edge[cell].remove(edge)
                if not self.cell_to_edge[cell]:
                    del self.cell_to_edge[cell]

        self.vertices = [vertex for vertex in self.vertices if vertex not in removed_vertices]

        print("REMOVED VERTICES")

        # add new obstacle and special point vertices with edges to all existing vertices
        current_vertex_set = set(self.vertices)
        added_vertices = [vertex for vertex in new_vertices if vertex not in current_vertex_set]
        added_edges = set()
        for vertex in added_vertices:
            for other_vertex in self.vertices:
                edge = (other_vertex, vertex)
                cells = self._determine_edge_cells(edge[0], edge[1])
                self.edge_to_cells[edge] = cells
                self.edge_active_status[edge] = True
                added_edges.add(edge)
                for edge_vertex in edge:
                    if edge_vertex not in self.vertex_to_edges:
                        self.vertex_to_edges[edge_vertex] = set()
                    self.vertex_to_edges[edge_vertex].add(edge)
                for cell in cells:
                    if cell not in self.cell_to_edge:
                        self.cell_to_edge[cell] = set()
                    self.cell_to_edge[cell].add(edge)
            self.vertices.append(vertex)
            
        print("ADDED VERTICES AND EDGES")

        # deactivate new edges that cross an existing obstacle cell
        for edge in added_edges:
            if any(self.occupancy[cell] != 0 for cell in self.edge_to_cells[edge]):
                self.edge_active_status[edge] = False

        # deactivate all edges affected by added obstacle cells
        for cell in added_cells:
            if cell in self.cell_to_edge:
                for edge in self.cell_to_edge[cell]:
                    self.edge_active_status[edge] = False

        self.edges = [edge for edge in self.edge_to_cells if self.edge_active_status[edge]]

        print("UPDATED EDGE STATUS")

        self.iteration += 1
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)

        self.record_graph(phase="online_update")
    
    def record_graph(self, phase="manual"):
        """Record all visibility candidates and their current active status."""
        if self.recorder is None:
            return None

        edges = sorted(self.edge_to_cells)
        save_progress = GenerationProgress(
            "Dynamic visibility graph", f"Saving graph to disk ({phase})", self.iteration
        )
        save_progress.start(f"edges: {len(edges)}")
        try:
            saved_artifacts = self.recorder.save(
                data_type="dynamic_vgraph",
                data={
                    "occupancy": self.occupancy.copy(),
                    "agv_radius": self.agv_radius,
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
                # plot=True,
                plot=False,
            )
        except Exception as error:
            save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts

        
