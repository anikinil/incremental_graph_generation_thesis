
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, box

from artifacts.progress import GenerationProgress
from scenario.geometry import buffered_obstacle_space


class DRM:

    def __init__(self, rng, occupancy, access_points, io_points, bounds, max_dist, max_neighbors, agv_radius, buffer_segments, recorder=None):
        
        self.rng = rng
        self.occupancy = np.asarray(occupancy, dtype=np.uint8).copy()
        self.height, self.width = self.occupancy.shape
        self.bounds = tuple(map(float, bounds))
        self.max_dist = float(max_dist)
        self.max_neighbors = int(max_neighbors)
        self.agv_radius = float(agv_radius)
        self.buffer_segments = int(buffer_segments)
        self.recorder = recorder
        self.iteration = 0

        self.empty_space = box(*self.bounds)
        self.obstacle_space = None
        self.free_space = None
        self._refresh_free_and_obstacle_space()

        self.vertices = []
        self.edges = []
        self.vertex_active = [] # bool (indexed by vertex index)
        self.edge_active = {} # (edge_tuple, bool)
        self.adjacency = defaultdict(set) # (vertex_index, set(neighbour_indices))
        
        self.kdtree = None
 
        # tracks which roadmap vertices and edges are on which occupancy cells
        self.phi_n = defaultdict(set) # (cell, set(vertex_indices))
        self.phi_a = defaultdict(set) # (cell, set(edge_tuples))

        self.special_vertices = {} # (point, vertex_index)
        self.access_points = access_points
        self.io_points = io_points

    def _refresh_free_and_obstacle_space(self):
        self.obstacle_space = buffered_obstacle_space(self.occupancy, self.agv_radius, self.buffer_segments)
        self.free_space = self.empty_space.difference(self.obstacle_space)

    def _update_kdtree(self):
        self.kdtree = KDTree(np.asarray(self.vertices)) if self.vertices else None

    def _add_vertex(self, config):
        """Add one vertex and connect it to nearby vertices in the empty workspace."""
        config = np.asarray(config, dtype=np.float64)
        neighbours = []

        # find nearest neighbours of the new vertex
        if self.kdtree is not None: # on the first vertex, there is no kdtree yet, so skip this step
            count = min(self.max_neighbors, len(self.vertices)) # limit the number of neighbours to consider
            # get the indices of the nearest neighbours within max_dist
            distances, indices = self.kdtree.query(config, k=count, distance_upper_bound=self.max_dist)
            # ensure arrays so scalar returns from KDTree are handled uniformly
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
            neighbours = [int(idx) for dist, idx in zip(distances, indices) if np.isfinite(dist)]
            # if no neighbours are found within max_dist, connect to the single nearest neighbour regardless of distance
            if not neighbours:
                neighbours = [int(self.kdtree.query(config, k=1)[1])]

        # add the new vertex to the roadmap and initialize 
        index = len(self.vertices)
        self.vertices.append(config.copy())
        self.vertex_active.append(False)
        self.adjacency[index] = set()

        # connect the new vertex to its neighbours if the connecting edge is in free space
        for neighbour in neighbours:
            line = LineString([config, self.vertices[neighbour]])
            if not self.empty_space.covers(line):
                continue
            edge = tuple(sorted((index, neighbour)))
            self.edges.append(edge)
            self.adjacency[index].add(neighbour)
            self.adjacency[neighbour].add(index)
            self.edge_active[edge] = False # initialize edge as inactive, will be updated later

        self._update_kdtree()
        return index

    def _construction_step(self, num_vertices):
        """Sample the permanent roadmap in the obstacle-free workspace."""
        
        progress = GenerationProgress("DRM", "Generating empty-space roadmap", self.iteration)
        progress.start(f"target: {num_vertices} vertices")
        min_x, min_y, max_x, max_y = self.bounds
        for number in range(1, num_vertices + 1):
            config = np.array([self.rng.uniform(min_x, max_x), self.rng.uniform(min_y, max_y)], dtype=np.float64)
            self._add_vertex(config)
            if number % 100 == 0 or number == num_vertices:
                progress.update(number, num_vertices, f"vertices: {len(self.vertices)} | edges: {len(self.edges)}")
        progress.finish(True, f"generated {len(self.vertices)} vertices and {len(self.edges)} edges")

    def _remove_special_vertices(self, points):
        """Remove special-point vertices and their incident edges from the roadmap."""
        
        # determine the indices of the special vertices to be removed and remove them from special_vertices
        removed_indices = {self.special_vertices.pop(point) for point in points}
        if not removed_indices: # no special vertices have been removed, skip the rest of the function
            return

        # determine kept vertex indices and create a mapping from old indices to new indices
        kept_indices = [index for index in range(len(self.vertices)) if index not in removed_indices]
        new_kept_indices = {old_index: new_index for new_index, old_index in enumerate(kept_indices)}
        
        # update vertices and vertex_active status of remaining vertices according to the new indices
        self.vertices = [self.vertices[index] for index in kept_indices]
        self.vertex_active = [self.vertex_active[index] for index in kept_indices]

        # update adjacency of remaining vertices according to the new indices
        old_adjacency = self.adjacency
        self.adjacency = defaultdict(set)
        for old_index in kept_indices: # for each remaining vertex
            index = new_kept_indices[old_index]
            for neighbour in old_adjacency[old_index]: # for each neighbour of the remaining vertex
                if neighbour in new_kept_indices: # if the neighbour is also a remaining vertex
                    self.adjacency[index].add(new_kept_indices[neighbour]) # register the neighbour as adjacent to the remaining vertex

        # update edges and edge_active status of remaining edges according to the new indices
        old_edges = self.edges
        old_edge_active = self.edge_active
        self.edges = []
        for left, right in old_edges:
            if left in new_kept_indices and right in new_kept_indices:
                self.edges.append(tuple(sorted((new_kept_indices[left], new_kept_indices[right]))))
        self.edge_active = {}
        for left, right in old_edges:
            if left in new_kept_indices and right in new_kept_indices:
                edge_key = tuple(sorted((new_kept_indices[left], new_kept_indices[right])))
                self.edge_active[edge_key] = old_edge_active[(left, right)]

        # update phi_n and phi_a according to the new indices        
        remapped_phi_n = defaultdict(set)
        for cell, indices in self.phi_n.items():
            for index in indices:
                if index in new_kept_indices:
                    remapped_phi_n[cell].add(new_kept_indices[index])
        self.phi_n = remapped_phi_n
        remapped_phi_a = defaultdict(set)
        for cell, edges in self.phi_a.items():
            for left, right in edges:
                if left in new_kept_indices and right in new_kept_indices:
                    edge = tuple(sorted((new_kept_indices[left], new_kept_indices[right])))
                    remapped_phi_a[cell].add(edge)
        self.phi_a = remapped_phi_a

        # update special_vertices
        remapped_special_vertices = {}
        for point, index in self.special_vertices.items():
            remapped_special_vertices[point] = new_kept_indices[index]
        self.special_vertices = remapped_special_vertices
        
        self._update_kdtree()

    def _update_special_points(self, access_points, io_points):
        """Synchronize special-point terminals and return their vertices and edges."""
        self.access_points = list(access_points)
        self.io_points = list(io_points)
        current_points = list(dict.fromkeys(self.access_points + self.io_points))

        # determine which special points have been removed and remove their vertices and edges from the roadmap
        removed_points = set(self.special_vertices).difference(current_points)
        self._remove_special_vertices(removed_points)

        active_vertices = [] # list of active special vertices
        for point in current_points:
            index = self.special_vertices.get(point)
            if index is None:
                index = self._add_vertex(point)
                self.special_vertices[point] = index
                self._map_vertex_to_occupancy_cell(index) # map the new special point to its occupancy cell
                for neighbour in self.adjacency[index]:
                    self._map_edge_to_occupancy_cells(tuple(sorted((index, neighbour)))) # map the new special point's edges to their occupancy cells
            active_vertices.append(index)

        changed_edges = set()
        for index in active_vertices:
            for neighbour in self.adjacency[index]:
                changed_edges.add(tuple(sorted((index, neighbour))))
        return set(active_vertices), changed_edges

        
    def _get_intersecting_cells(self, geometry):
        """Return the set of occupancy cells that intersect the given geometry."""

        min_x, min_y, max_x, max_y = geometry.bounds
        first_row = max(0, int(np.floor(min_y)))
        last_row = min(self.height, int(np.ceil(max_y)))
        first_col = max(0, int(np.floor(min_x)))
        last_col = min(self.width, int(np.ceil(max_x)))

        intersecting_cells = set()
        for row in range(first_row, last_row):
            for col in range(first_col, last_col):
                cell = box(col, row, col + 1, row + 1)
                if geometry.intersects(cell):
                    intersecting_cells.add((row, col))
        return intersecting_cells

    def _get_vertex_footprint_cells(self, index):
        """Return the set of occupancy cells that intersect the given roadmap vertex's footprint."""
        footprint = Point(self.vertices[index])
        footprint = footprint.buffer(self.agv_radius, quad_segs=self.buffer_segments)
        return self._get_intersecting_cells(footprint)

    def _get_edge_footprint_cells(self, edge):
        """Return the set of occupancy cells that intersect the given roadmap edge's footprint."""
        path = LineString([self.vertices[edge[0]], self.vertices[edge[1]]])
        path = path.buffer(self.agv_radius, quad_segs=self.buffer_segments)
        cells = self._get_intersecting_cells(path)
        cells.difference_update(self._get_vertex_footprint_cells(edge[0]))
        cells.difference_update(self._get_vertex_footprint_cells(edge[1]))
        return cells

    def _map_vertex_to_occupancy_cell(self, index):
        """Map the given roadmap vertex to its footprint's occupancy cell."""
        for cell in self._get_vertex_footprint_cells(index):
            self.phi_n[cell].add(index)

    def _map_edge_to_occupancy_cells(self, edge):
        """Map the given roadmap edge to its footprint's occupancy cells."""
        for cell in self._get_edge_footprint_cells(edge):
            self.phi_a[cell].add(edge)

    def _build_workspace_to_roadmap_mapping(self):
        """Map all roadmap vertices and edges to their footprint's occupancy cells."""
        self.phi_n.clear()
        self.phi_a.clear()
        for index in range(len(self.vertices)):
            self._map_vertex_to_occupancy_cell(index)
        for edge in self.edges:
            self._map_edge_to_occupancy_cells(edge)

        self._refresh_active_status(range(len(self.vertices)), self.edges)
        # self.record_graph(phase="preprocessing")

    def preprocessing_step(self, num_vertices):
        self._construction_step(num_vertices)
        self._update_special_points(self.access_points, self.io_points)
        self._build_workspace_to_roadmap_mapping()

    def _refresh_active_status(self, vertices, edges):
        """Refresh the active/inactive status of the given roadmap vertices and edges."""
        
        for vertex_index in vertices:
            self.vertex_active[vertex_index] = self.free_space.covers(Point(self.vertices[vertex_index]))

        for edge in edges:
            edge_is_in_free_space = self.free_space.covers(LineString([self.vertices[edge[0]], self.vertices[edge[1]]]))
            both_vertices_are_active = (self.vertex_active[edge[0]] and self.vertex_active[edge[1]])

            self.edge_active[edge] = (both_vertices_are_active and edge_is_in_free_space)

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_io_points):
        """Apply changed cells, then refresh only their mapped roadmap items."""

        # update the occupancy grid and collect all changed cells
        changed_cells = set()
        for update in obstacle_updates:
            action = update["action"]
            if action == "add":
                cell = tuple(update["cell"])
                self.occupancy[cell] = update.get("obstacle_type", 1)
                changed_cells.add(cell)
            elif action == "remove":
                cell = tuple(update["cell"])
                self.occupancy[cell] = 0
                changed_cells.add(cell)
            elif action == "move":
                source = tuple(update["from_cell"])
                target = tuple(update["to_cell"])
                self.occupancy[target] = self.occupancy[source] or 1
                self.occupancy[source] = 0
                changed_cells.update((source, target))

        # update the free space
        self._refresh_free_and_obstacle_space()
        
        # update special points and get their vertices and edges
        special_vertices, special_edges = self._update_special_points(new_access_points, new_io_points)

        # update the active/inactive status of all vertices and edges affected by cell changes
        affected_vertices = set()
        affected_edges = set()
        for cell in changed_cells:
            affected_vertices.update(self.phi_n.get(cell, ()))
            affected_edges.update(self.phi_a.get(cell, ()))

        # add all edges connected to affected vertices to the affected set
        for index in affected_vertices:
            for neighbour in self.adjacency[index]:
                affected_edges.add(tuple(sorted((index, neighbour))))

        # refresh the active/inactive status of all affected vertices and edges, including special points
        self._refresh_active_status(affected_vertices | special_vertices, affected_edges | special_edges)
        
        # record the updated graph
        self.iteration += 1
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)
        self.record_graph(phase="online_update")

    def build_graph(self):
        """Return the roadmap as a NetworkX graph."""
        graph = nx.Graph()
        for index, config in enumerate(self.vertices):
            if self.vertex_active[index]:
                graph.add_node(index, pos=tuple(config), active=self.vertex_active[index])

        for edge in self.edges:
            if self.edge_active[edge]:
                graph.add_edge(edge[0], edge[1], active=self.edge_active[edge])
        return graph

    def record_graph(self, phase="manual"):
        """Save the roadmap for the existing DRM plotter."""
        if self.recorder is None:
            return None

        vertices = (np.asarray(self.vertices, dtype=np.float64) if self.vertices else np.empty((0, 2), dtype=np.float64))
        
        return self.recorder.save(
            data_type="drm_graph",
            data={
                "occupancy": self.occupancy.copy(),
                "agv_radius": self.agv_radius,
                "buffer_segments": self.buffer_segments,
                "access_points": sorted(self.access_points),
                "io_points": sorted(self.io_points),
                "bounds": self.bounds,
                "vertices": vertices,
                "vertex_active": np.asarray(self.vertex_active, dtype=bool),
                "edges": sorted(self.edges),
                "edge_active": {edge: self.edge_active[edge] for edge in sorted(self.edges)},
                "mapped_vertex_cells": sum(len(vertices) for vertices in self.phi_n.values()),
                "mapped_arc_cells": sum(len(edges) for edges in self.phi_a.values()),
                "phase": phase,
            },
            plot=True,
            iteration=self.iteration,
        )
