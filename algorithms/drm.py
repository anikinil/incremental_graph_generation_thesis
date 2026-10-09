
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, box
from shapely.ops import unary_union

from artifacts.progress import GenerationProgress
from algorithms.geometry import buffered_edge_cells


class DRM:
    """
    Based on:
    
    "A Framework for Real-time Path Planning in Changing Environments",
    Peter Leven, Seth Hutchinson (2002)
    """
    
    def __init__(self, rng, occupancy, access_points, interaction_points, height, width, max_dist, max_neighbors, agv_radius, num_buffer_segments, recorder=None):

        self.rng = rng
        self.occupancy = np.asarray(occupancy, dtype=np.uint8).copy()
        self.height = height
        self.width = width
        self.max_dist = float(max_dist)
        self.max_neighbors = int(max_neighbors)
        self.agv_radius = float(agv_radius)
        self.num_buffer_segments = int(num_buffer_segments)
        self.recorder = recorder
        self.iteration = 0

        self.obstacle_space = None
        self._refresh_obstacle_space()

        self.vertices = []
        # indicates whether a vertex is present in the roadmap (True) or has been removed (False)
        self.vertex_present = [] # bool (indexed by vertex index)
        self.edges = []
        self.vertex_active = [] # bool (indexed by vertex index)
        self.edge_active = {} # (edge_tuple, bool)
        self.adjacency = defaultdict(set) # (vertex_index, set(neighbour_indices))

        self.kdtree = None
        self.kdtree_indices = np.empty(0, dtype=int)
 
        # tracks which roadmap vertices and edges are on which occupancy cells
        self.phi_n = defaultdict(set) # (cell, set(vertex_indices))
        self.phi_a = defaultdict(set) # (cell, set(edge_tuples))

        self.special_vertices = {} # (point, vertex_index) mapping for special points
        
        self.access_points = access_points
        self.interaction_points = interaction_points
        
        self.wall_vertices = {} # (row, col) -> vertex_index
        self.affected_vertices = set()
        self.affected_edges = set()
        
        self.KING_MOVES = (
            (-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1),
        )

    def _rebuild_kdtree(self):
        """Rebuild the nearest-neighbour tree for the current roadmap vertices."""
        self.kdtree_indices = np.flatnonzero(self.vertex_present)
        if self.kdtree_indices.size == 0:
            self.kdtree = None
            return
        self.kdtree = KDTree(np.asarray([self.vertices[index] for index in self.kdtree_indices]))

    def _refresh_obstacle_space(self):
        """Rebuild the unbuffered workspace obstacle geometry."""
        occupied_cells = []
        for row in range(self.height):
            for col in range(self.width):
                cell = self.occupancy[row, col]
                if cell != 0:
                    occupied_cells.append(box(col, row, col + 1, row + 1)) # swap row and col to get x and y
        self.obstacle_space = unary_union(occupied_cells)

    def _is_config_collision_free(self, config):
        """Return whether the robot can occupy config without hitting an obstacle."""
        point = Point(config)
        if self.agv_radius == 0: # if the robot is a point, check if it is in free space
            return point.disjoint(self.obstacle_space)
        else: # else check if the distance from the robot's center to the nearest obstacle is at least the robot's radius
            return point.distance(self.obstacle_space) >= self.agv_radius

    def _is_path_collision_free(self, start, end):
        """Return whether the robot can traverse the path from start to end collision-free."""
        crossed_cells = buffered_edge_cells(start, end, self.agv_radius, self.height, self.width)
        return all(self.occupancy[row, col] == 0 for row, col in crossed_cells)

    def _get_candidate_neighbors(self, config):
        """Return nearby roadmap vertices in increasing distance order."""

        # if no vertices created yet, return empty list
        if self.kdtree is None or self.kdtree_indices.size == 0:
            return []

        # number of neighbors to query
        k = min(self.max_neighbors, self.kdtree_indices.size)
        # get distances and positions of the nearest candidates within max_dist
        distances, positions = self.kdtree.query(config, k=k, distance_upper_bound=self.max_dist)
        distances = np.atleast_1d(distances)
        positions = np.atleast_1d(positions)

        # filter out infinite distances (no neighbor found within max_dist)
        candidates = [int(self.kdtree_indices[position]) for distance, position in zip(distances, positions) if np.isfinite(distance)]
        # if no candidates found, return the nearest vertex regardless of distance
        if not candidates:
            nearest_position = int(self.kdtree.query(config, k=1)[1])
            return [int(self.kdtree_indices[nearest_position])]
        return candidates

    def _add_vertex(self, config):
        """Add one vertex and initialise its DRM-specific state."""
        config = np.asarray(config, dtype=np.float64)
        index = len(self.vertices)
        self.vertices.append(config.copy())
        self.vertex_present.append(True)
        self.vertex_active.append(False)
        self.adjacency[index] = set()
        return index

    def _add_edge(self, left, right):
        """Add one roadmap edge."""
        edge = tuple(sorted((left, right)))
        self.edges.append(edge)
        self.adjacency[left].add(right)
        self.adjacency[right].add(left)
        self.edge_active[edge] = False
        return edge

    def _get_incident_edges(self, vertices):
        """Return all roadmap edges incident to the given vertices."""
        edges = set()
        for index in vertices:
            for neighbour in self.adjacency[index]:
                edges.add(tuple(sorted((index, neighbour))))
        return edges
    
    def _remove_edge(self, edge):
        """Remove one roadmap edge and its occupancy-cell mapping."""
        self._remove_mapping(self.phi_a, edge, self._get_edge_footprint_cells(edge))

        self.edges.remove(edge)
        self.adjacency[edge[0]].discard(edge[1])
        self.adjacency[edge[1]].discard(edge[0])
        del self.edge_active[edge]
        return True

    def _remove_mapping(self, mapping, item, cells):
        """Remove an item from the occupancy-cell mapping for the given cells."""
        for cell in cells:
            mapped_items = mapping.get(cell)
            if mapped_items is not None:
                mapped_items.discard(item)
                if not mapped_items:
                    del mapping[cell]

    def _num_active_components(self):
        """Return the number of connected components in the active roadmap."""
        return nx.number_connected_components(self.build_graph())

    def _generate_random_free_config(self, max_attempts = 1000):
        """Generates a random configuration in the free space using rejection sampling."""
        for _ in range(max_attempts):
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            config = np.array([x, y], dtype=np.float64)
            if self._is_config_collision_free(config):
                return config
        raise RuntimeError(f"Failed to generate a random free configuration after {max_attempts} attempts.")
    
    def _repair_active_connectivity(self, min_num_vertices):
        """
        Add roadmap vertices until the active roadmap has one component.
        
        min_num_vertices: minimum number of vertices to add (None: add until connected)
        """
        progress = GenerationProgress("DRM", "Connecting active roadmap", self.iteration)
        progress.start()
        
        num_components = self._num_active_components()
        added_vertices = set()
        added_vertices_num = 0
        added_edges = set()
        while num_components > 1 or (min_num_vertices is not None and added_vertices_num < min_num_vertices):
            c = self._generate_random_free_config()
            candidate_indices = self._get_candidate_neighbors(c)
            c_idx = self._add_vertex(c)
            added_vertices_num += 1
            for n_idx in candidate_indices:
                self._add_edge(c_idx, n_idx)
            self._rebuild_kdtree()
            incident_edges = self._get_incident_edges({c_idx})
            self._refresh_active_status({c_idx}, incident_edges)
            num_components = self._num_active_components()
            added_vertices.add(c_idx)
            added_edges.update(incident_edges)
            if added_vertices_num % 100 == 0:
                progress.update(added_vertices_num, details=(f"components: {num_components} | vertices: {sum(self.vertex_present)} | "
                                                             f"edges: {len(self.edges)}"))
        progress.finish(True, (f"added vertices: {added_vertices_num} | vertices: {sum(self.vertex_present)} | "
                               f"edges: {len(self.edges)} | components: {num_components}"))
        return added_vertices, added_edges

    def _build_wall_graph(self):
        """
        Adds the free, wall-adjacent cell centres as an 8-connected graph.

        Similar to: "A Randomized Roadmap Method for Path and Manipulation Planning", Amato & Wu (1996) 
        """
        # collect the wall vertices from the occupancy grid and add them to the roadmap
        for row in range(self.height):
            for col in range(self.width):
                if self._is_adjacent_to_wall((row, col)):
                    # add the wall vertex to the roadmap
                    index = self._add_vertex(np.array([col + 0.5, row + 0.5], dtype=np.float64))
                    self.wall_vertices[(row, col)] = index

        self._connect_wall_vertices(self.wall_vertices)
        self._rebuild_kdtree()

    def _is_adjacent_to_wall(self, cell):
        """Return whether a cell is adjacent to a free cell."""
        row, col = cell
        if self.occupancy[row, col] == 0: # cell not free -> not a wall cell
            for row_offset, col_offset in self.KING_MOVES:
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                if (0 <= neighbour_row < self.height and # if neighbor in bounds
                    0 <= neighbour_col < self.width and
                    self.occupancy[neighbour_row, neighbour_col] != 0): # and occupied
                    return True # wall cell
        return False

    def _connect_wall_vertices(self, cells):
        """Add all missing valid King's Graph edges for the given wall cells."""
        new_edges = set()
        for row, col in cells:
            index = self.wall_vertices[(row, col)]
            for row_offset, col_offset in self.KING_MOVES:
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                neighbour_index = self.wall_vertices.get((neighbour_row, neighbour_col))
                if neighbour_index is None:
                    continue
                edge = tuple(sorted((index, neighbour_index)))
                if edge in self.edge_active: # edge already created by another wall cell
                    continue
                if (abs(row_offset) + abs(col_offset) == 2 and
                    (self.occupancy[row, neighbour_col] != 0 or self.occupancy[neighbour_row, col] != 0)):
                    continue # diagonal edge would cut a corner through an occupied cell
                new_edges.add(self._add_edge(index, neighbour_index))
        return new_edges

    def _get_wall_update_candidate_cells(self, changed_cells):
        """Return the set of wall-cell candidates affected by changes in occupancy."""
        # get the neighborhood of changed cells
        neighbourhood_cells = set()
        # for all changed cells
        for row, col in changed_cells:
            # for whole 8-neighborhood of changed cell (including the changed cell itself)
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    neighbour_row = row + row_offset
                    neighbour_col = col + col_offset
                    if 0 <= neighbour_row < self.height and 0 <= neighbour_col < self.width:
                        cell = (neighbour_row, neighbour_col)
                        # add free cells that may become wall cells and wall vertices that may need removal
                        if self.occupancy[cell] == 0 or cell in self.wall_vertices:
                            neighbourhood_cells.add(cell)

        return neighbourhood_cells

    def _update_wall_graph(self, changed_cells):
        """Update the wall graph based on changed occupancy cells."""
        
        # get the set of cells that are considered affected by the changes in occupancy
        potentially_affected_cells = self._get_wall_update_candidate_cells(changed_cells)
        
        # collect all wall vertices that are no longer on wall cells and remove them from the roadmap
        removed_indices = set()
        for cell in potentially_affected_cells:
            # if cell is a wall vertex but no longer a wall cell, mark it for removal
            if cell in self.wall_vertices and not self._is_adjacent_to_wall(cell):
                removed_indices.add(self.wall_vertices[cell])
        self._remove_vertices(removed_indices)

        # add new wall vertices for any newly created wall cells
        added_vertices = set()
        for cell in potentially_affected_cells:
            if self._is_adjacent_to_wall(cell) and cell not in self.wall_vertices:
                row, col = cell
                index = self._add_vertex(np.array([col + 0.5, row + 0.5], dtype=np.float64))
                self.wall_vertices[cell] = index
                added_vertices.add(index)
        if added_vertices:
            self._rebuild_kdtree()
        
        # remove edges incident to affected wall vertices
        affected_vertices = set()
        for cell in potentially_affected_cells:
            if cell in self.wall_vertices:
                affected_vertices.add(self.wall_vertices[cell])
        for index in affected_vertices:
            incident_edges = [tuple(sorted((index, neighbour))) for neighbour in self.adjacency[index]]
            for edge in incident_edges:
                self._remove_edge(edge)

        affected_wall_cells = [cell for cell, index in self.wall_vertices.items() if index in affected_vertices]
        new_edges = self._connect_wall_vertices(affected_wall_cells)
        wall_indices = set(self.wall_vertices.values())

        # get the indices of the permanent vertices that are not wall vertices, to connect the new wall vertices to them
        permanent_indices = np.asarray([index for index, present in enumerate(self.vertex_present) if present and index not in wall_indices], dtype=int)
        # convert the permanent vertices into a KDTree for efficient nearest-neighbor queries
        permanent_tree = (KDTree(np.asarray([self.vertices[index] for index in permanent_indices], dtype=np.float64)) 
            if permanent_indices.size > 0 else None)
        
        # connect the affected wall vertices to the permanent roadmap vertices
        candidate_count = min(self.max_neighbors, len(permanent_indices))
        for index in affected_vertices:
            if permanent_tree is not None:
                distances, positions = permanent_tree.query(self.vertices[index], k=candidate_count, distance_upper_bound=self.max_dist)
                for distance, position in zip(np.atleast_1d(distances), np.atleast_1d(positions)):
                    if np.isfinite(distance): # if the candidate is within max_dist
                        candidate_index = int(permanent_indices[int(position)])
                        # if the robot's swept footprint is obstacle-free, add the edge and move to the next wall vertex.
                        if self._is_path_collision_free(self.vertices[index], self.vertices[candidate_index]):
                            new_edges.add(self._add_edge(index, candidate_index))
                            break
                        
        self._map_roadmap_elements(added_vertices, new_edges)
        return affected_vertices, new_edges


    def _remove_vertices(self, removed_indices):
        """Remove vertices without changing the remaining roadmap indices."""
        removed_indices = {index for index in removed_indices if self.vertex_present[index]}
        if not removed_indices:
            return
        print(f"Removing {len(removed_indices)} vertices from the roadmap.")

        for edge in self._get_incident_edges(removed_indices):
            self._remove_edge(edge)
        for index in removed_indices:
            self._remove_mapping(self.phi_n, index, self._get_vertex_footprint_cells(index))
            self.vertex_present[index] = False
            self.vertex_active[index] = False

        self.special_vertices = {
            point: index for point, index in self.special_vertices.items() if index not in removed_indices
        }
        self.wall_vertices = {
            cell: index for cell, index in self.wall_vertices.items() if index not in removed_indices
        }
        self._rebuild_kdtree()
        print(f"Removed {len(removed_indices)} vertices. Remaining vertices: {sum(self.vertex_present)}. Remaining edges: {len(self.edges)}.")


    def _update_special_points(self, access_points, interaction_points, map_new=True):
        """Update the special points in the roadmap, optionally mapping newly added elements to occupancy cells."""
        self.access_points = list(access_points)
        self.interaction_points = list(interaction_points)
        current_points = list(dict.fromkeys(self.access_points + self.interaction_points))

        # determine which special points have been removed and remove their vertices and edges from the roadmap
        removed_points = set(self.special_vertices).difference(current_points)
        removed_indices = set()
        for point in removed_points:
            if point in self.special_vertices:
                removed_indices.add(self.special_vertices[point])
        self._remove_vertices(removed_indices)

        # add new special points to the roadmap
        count = 0
        total = len(current_points)
        new_vertices = set()
        for point in current_points:
            index = self.special_vertices.get(point)
            if index is None:
                print(f"Adding special points to roadmap: {count + 1}/{total} ({(count + 1) / total * 100:.2f}%) ({point})", end="\r", flush=True)
                candidate_indices = self._get_candidate_neighbors(point)
                index = self._add_vertex(point)
                for neighbour_index in candidate_indices:
                    self._add_edge(index, neighbour_index)
                self._rebuild_kdtree()
                self.special_vertices[point] = index
                new_vertices.add(index)
                count += 1
        if map_new:
            new_edges = self._get_incident_edges(new_vertices)
            self._map_roadmap_elements(new_vertices, new_edges)

    def _get_vertex_footprint_cells(self, index):
        """Return the set of cells that intersect the given roadmap vertex's footprint."""
        x, y = self.vertices[index]
        radius = self.agv_radius

        # bounding box of the circular footprint
        first_row = max(0, int(np.floor(y - radius)))
        last_row = min(self.height, int(np.ceil(y + radius)))
        first_col = max(0, int(np.floor(x - radius)))
        last_col = min(self.width, int(np.ceil(x + radius)))

        intersecting_cells = set()
        for row in range(first_row, last_row):
            for col in range(first_col, last_col):
                nearest_x = min(max(x, col), col + 1)
                nearest_y = min(max(y, row), row + 1)
                if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 < radius ** 2:
                    intersecting_cells.add((row, col))
        return intersecting_cells

    def _get_edge_footprint_cells(self, edge):
        """Return the set of cells that intersect the given roadmap edge's footprint."""
        # footprint = self._path_footprint(self.vertices[edge[0]], self.vertices[edge[1]])
        cells = set(buffered_edge_cells(self.vertices[edge[0]], self.vertices[edge[1]], self.agv_radius, self.height, self.width))
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

    def _map_roadmap_elements(self, vertices, edges):
        """Map the given roadmap vertices and edges to occupancy cells."""
        for index in vertices:
            self._map_vertex_to_occupancy_cell(index)
        for edge in edges:
            self._map_edge_to_occupancy_cells(edge)

    def _get_mapped_roadmap_items(self, cells):
        """Return the roadmap vertices and edges mapped to the given occupancy cells."""
        vertices = {index for cell in cells for index in self.phi_n.get(cell, ())}
        edges = {edge for cell in cells for edge in self.phi_a.get(cell, ())}
        return vertices, edges

    def _build_workspace_to_roadmap_mapping(self):
        """Map all roadmap vertices and edges to their footprint's occupancy cells."""
        self.phi_n.clear()
        self.phi_a.clear()
        self._map_roadmap_elements(np.flatnonzero(self.vertex_present), self.edges)
        self._refresh_active_status(np.flatnonzero(self.vertex_present), self.edges)

    def preprocessing_step(self, min_num_vertices=None):
        """Build the initial roadmap and map all vertices and edges to their occupancy cells."""
        self.affected_vertices.clear()
        self.affected_edges.clear()
        
        self._build_wall_graph()
        print("WALL GRAPH BUILT")
        # self._repair_active_connectivity()
        self._update_special_points(self.access_points, self.interaction_points, map_new=False)
        print("SPECIAL POINTS ADDED", flush=True)
        self._build_workspace_to_roadmap_mapping()
        print("WORKSPACE MAPPED TO ROADMAP", flush=True)

        repair_vertices, repair_edges = self._repair_active_connectivity(min_num_vertices)
        print(f"REPAIRED CONNECTIVITY")
        self._map_roadmap_elements(repair_vertices, repair_edges)

        self.record_graph(phase="preprocessing")

    def _refresh_active_status(self, vertices, edges):
        """Refresh roadmap status using the circular robot footprint against raw obstacles."""
        
        for vertex_index in vertices:
            if self.vertex_present[vertex_index]:
                self.vertex_active[vertex_index] = self._is_config_collision_free(self.vertices[vertex_index])

        for edge in edges:
            if edge in self.edge_active:
                edge_is_collision_free = self._is_path_collision_free(self.vertices[edge[0]], self.vertices[edge[1]])
                both_vertices_are_active = (self.vertex_active[edge[0]] and self.vertex_active[edge[1]])
                self.edge_active[edge] = (both_vertices_are_active and edge_is_collision_free)

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_interaction_points, min_num_vertices=None):
        """Apply changed cells and update only the affected roadmap regions."""

        self.affected_vertices.clear()
        self.affected_edges.clear()

        self.iteration += 1
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)
        
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
                
        # update the obstacle space
        self._refresh_obstacle_space()
        
        # determine which special points are new (for highlighting in plot)
        new_special_points = (set(new_access_points) | set(new_interaction_points)).difference(self.special_vertices)

        # update special points and map new elements before the wall update consults cell mappings
        self._update_special_points(new_access_points, new_interaction_points)

        # update the wall-graph vertices and edges affected by the edits
        affected_wall_vertices, new_wall_edges = self._update_wall_graph(changed_cells)

        # update the active/inactive status of all vertices and edges affected by cell changes
        affected_vertices, affected_edges = self._get_mapped_roadmap_items(changed_cells)

        # add all edges connected to affected vertices to the affected set
        affected_edges.update(self._get_incident_edges(affected_vertices))

        # determine which special vertices and edges are affected by the edits
        special_vertices = set(self.special_vertices.values())
        new_special_vertices = {
            self.special_vertices[point] for point in new_special_points if point in self.special_vertices
        }
        changed_special_edges = self._get_incident_edges(new_special_vertices)
        special_edges = self._get_incident_edges(special_vertices)

        # refresh the active/inactive status of all affected vertices and edges, including special points
        refreshed_vertices = affected_vertices | special_vertices | affected_wall_vertices
        refreshed_edges = affected_edges | special_edges | new_wall_edges
        self._refresh_active_status(refreshed_vertices, refreshed_edges)

        # repair the active roadmap connectivity if it has become disconnected
        repair_vertices, repair_edges = self._repair_active_connectivity(min_num_vertices)
        self._map_roadmap_elements(repair_vertices, repair_edges)

        # record the affected vertices and edges for highlighting
        self.affected_vertices = refreshed_vertices | repair_vertices
        self.affected_edges = affected_edges | new_wall_edges | changed_special_edges | repair_edges
        
        # record the updated graph
        self.record_graph(phase="online_update")

    def build_graph(self):
        """Return the roadmap of active vertices and edges as a NetworkX graph."""
        graph = nx.Graph()
        for index, config in enumerate(self.vertices):
            if self.vertex_present[index] and self.vertex_active[index]:
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

        save_progress = GenerationProgress("DRM", f"Saving graph to disk ({phase})", self.iteration)
        save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="drm_graph",
                data={
                    "occupancy": self.occupancy.copy(),
                    "agv_radius": self.agv_radius,
                    "num_buffer_segments": self.num_buffer_segments,
                    "access_points": sorted(self.access_points),
                    "interaction_points": sorted(self.interaction_points),
                    "height": self.height,
                    "width": self.width,
                    "vertices": vertices,
                    "vertex_present": np.asarray(self.vertex_present, dtype=bool),
                    "vertex_active": np.asarray(self.vertex_active, dtype=bool),
                    "edges": sorted(self.edges),
                    "edge_active": {edge: self.edge_active[edge] for edge in sorted(self.edges)},
                    "affected_vertices": sorted(self.affected_vertices),
                    "affected_edges": sorted(self.affected_edges),
                    "mapped_vertex_cells": sum(len(vertices) for vertices in self.phi_n.values()),
                    "mapped_arc_cells": sum(len(edges) for edges in self.phi_a.values()),
                    "phase": phase,
                },
                plot=True,
                iteration=self.iteration,
            )
        except Exception as error:
            save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts
