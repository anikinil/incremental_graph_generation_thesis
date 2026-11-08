import numpy as np
from math import ceil

from logs.progress import ProgressReportingAlgorithm
from algorithms.geometry import buffered_edge_cells, navigation_space


class DynamicVGraph(ProgressReportingAlgorithm):
    """
    Dynamic Visibility Graph, with the method of activating/deactivating edges affected by dynamic obstacles, partially based on:
    
    "FAR Planner: Fast, Attemptable Route Planner using Dynamic Visibility Update"
    Yang, Cao, Zhu, Oh, and Zhang (2022)
    """
    
    
    progress_name = "Dynamic visibility graph"

    def __init__(self, height, width, occupancy, robot_radius, access_points, interaction_points,
                 recorder=None, print_progress=True):
        
        self.height = height
        self.width = width
        self.occupancy = occupancy.copy()
        self.robot_radius = robot_radius
        self.num_buffer_segments = 0 # always 0, since it's the only way to avoid collisions (temporary)
        self.access_points = access_points
        self.interaction_points = interaction_points
        self.recorder = recorder
        self.print_progress = bool(print_progress)
        
        self.nodes = []
        self.edges = []
        
        # mappings between grid cells, graph edges, and nodes
        self.cell_to_edge = {}
        self.edge_to_cells = {}
        self.node_to_edges = {}
        
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
    
    
    def _determine_edge_cells(self, node1, node2):
        """Returns grid cells crossed by the buffered edge between two nodes."""

        return buffered_edge_cells(node1, node2, self.robot_radius, self.height, self.width)


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

        node_progress = self._create_progress("Collecting nodes")
        if node_progress is not None:
            node_progress.start()
            
        start_time = self._start_generation_time()

        # get all convex vertices of the obstacle polygons
        _, obstacle_space = navigation_space(self.occupancy, self.robot_radius, num_buffer_segments=self.num_buffer_segments)
        nodes = list(dict.fromkeys(self._extract_convex_vertices(obstacle_space) + list(self.interaction_points) + list(self.access_points)))

        if node_progress is not None:
            node_progress.finish(True, f"collected {len(nodes)} visibility nodes")

        edge_progress = self._create_progress("Mapping candidate edges")
        if edge_progress is not None:
            total_pairs = len(nodes) * (len(nodes) - 1) // 2
            edge_progress.start(f"target: {total_pairs} node pairs")
                
        # for each node pair determine which cells the edge between them crosses
        # store mappings between cells and edges in dictionaries for fast lookup
        if edge_progress is not None:
            pair_number = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)): # order nodes by index to avoid duplicate edges
                edge = (nodes[i], nodes[j])
                cells = self._determine_edge_cells(edge[0], edge[1])
                self.edge_to_cells[edge] = cells
                for node in edge:
                    if node not in self.node_to_edges:
                        self.node_to_edges[node] = set()
                    self.node_to_edges[node].add(edge)
                for cell in cells:
                    if cell not in self.cell_to_edge:
                        self.cell_to_edge[cell] = set()
                    self.cell_to_edge[cell].add(edge)
                self.edge_active_status[edge] = True # initialize edge as active

                if edge_progress is not None:
                    pair_number += 1
                    if pair_number % 100 == 0 or pair_number == total_pairs:
                        edge_progress.update(pair_number, total_pairs)

        # mark edges that cross obstacle cells as inactive in edge_active_status
        self._deactivate_edges_in_obstacle_cells()

        # construct the graph from the nodes and active edges
        for node in nodes:
            self.nodes.append(node)
        for edge in self.edge_to_cells.keys():
            if self.edge_active_status[edge]:
                self.edges.append(edge)
                
        self._record_generation_time(self._generation_seconds(start_time), "generation")

        if edge_progress is not None:
            edge_progress.finish(True, f"mapped {len(self.edge_to_cells)} candidate edges | active: {len(self.edges)}", completed=total_pairs, total=total_pairs)


    def apply_layout_updates(self, obstacle_updates, new_access_points, new_interaction_points):
        """Apply layout updates and update the visibility graph."""
        
        progress = self._create_progress("Updating roadmap")
        if progress is not None:
            update_steps = 5
            progress.start(f"target: {update_steps} stages")
            
        start_time = self._start_generation_time()

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
                
        if progress is not None:
            progress.update(1, update_steps, "layout updates applied")

        # activate edges affected by removed obstacle cells, if no remaining obstacle blocks them
        for cell in removed_cells:
            if cell in self.cell_to_edge:
                for edge in self.cell_to_edge[cell]:
                    if all(self.occupancy[edge_cell] == 0 for edge_cell in self.edge_to_cells[edge]):
                        self.edge_active_status[edge] = True
        if progress is not None:
            progress.update(2, update_steps, "released edges activated")

        self.access_points = new_access_points
        self.interaction_points = new_interaction_points

        if progress is not None:
            progress.update(3, update_steps, "special points updated")

        # TODO make incremental, only look at relebvant parts instead of recomputing whole obstacle space
        # get all nodes of the updated obstacle space and special points
        _, obstacle_space = navigation_space(self.occupancy, self.robot_radius, num_buffer_segments=self.num_buffer_segments)
        new_nodes = list(dict.fromkeys(self._extract_convex_vertices(obstacle_space) + list(self.interaction_points) + list(self.access_points)))

        # remove nodes that are no longer part of the obstacle space or special points
        new_node_set = set(new_nodes)
        removed_nodes = set(self.nodes) - new_node_set
        removed_edges = set()
        for node in removed_nodes:
            removed_edges.update(self.node_to_edges.get(node, set()))

        for edge in removed_edges:
            cells = self.edge_to_cells.pop(edge)
            self.edge_active_status.pop(edge)
            for edge_node in edge:
                self.node_to_edges[edge_node].remove(edge)
                if not self.node_to_edges[edge_node]:
                    del self.node_to_edges[edge_node]
            for cell in cells:
                self.cell_to_edge[cell].remove(edge)
                if not self.cell_to_edge[cell]:
                    del self.cell_to_edge[cell]

        self.nodes = [node for node in self.nodes if node not in removed_nodes]

        if progress is not None:
            progress.update(4, update_steps, f"removed {len(removed_nodes)} nodes")

        # add new obstacle and special point nodes with edges to all existing nodes
        current_node_set = set(self.nodes)
        added_nodes = [node for node in new_nodes if node not in current_node_set]
        added_edges = set()
        for node in added_nodes:
            for other_node in self.nodes:
                edge = (other_node, node)
                cells = self._determine_edge_cells(edge[0], edge[1])
                self.edge_to_cells[edge] = cells
                self.edge_active_status[edge] = True
                added_edges.add(edge)
                for edge_node in edge:
                    if edge_node not in self.node_to_edges:
                        self.node_to_edges[edge_node] = set()
                    self.node_to_edges[edge_node].add(edge)
                for cell in cells:
                    if cell not in self.cell_to_edge:
                        self.cell_to_edge[cell] = set()
                    self.cell_to_edge[cell].add(edge)
            self.nodes.append(node)
            
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

        if progress is not None:
            progress.update(5, update_steps, "edge status updated")

        self.iteration += 1
        
        self._record_generation_time(self._generation_seconds(start_time), "update")
        
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)

        self.record_graph(phase="online_update")
        if progress is not None:
            progress.finish(True, f"nodes: {len(self.nodes)} | active edges: {len(self.edges)}", completed=update_steps, total=update_steps)
    
    def record_graph(self, phase="manual"):
        """Record all visibility candidates and their current active status."""
        if self.recorder is None:
            return None

        edges = sorted(self.edge_to_cells)
        save_progress = self._create_progress(f"Saving graph to disk ({phase})")
        if save_progress is not None:
            save_progress.start(f"edges: {len(edges)}")
        try:
            saved_artifacts = self.recorder.save(
                data_type="dynamic_vgraph",
                data={
                    "occupancy": self.occupancy.copy(),
                    "robot_radius": self.robot_radius,
                    "access_points": sorted(self.access_points),
                    "interaction_points": sorted(self.interaction_points),
                    "height": self.height,
                    "width": self.width,
                    "nodes": sorted({tuple(node) for node in self.nodes}),
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

        
