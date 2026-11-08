
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, box
from shapely.ops import unary_union

from logs.progress import ProgressReportingAlgorithm
from algorithms.geometry import buffered_edge_cells


class DRM(ProgressReportingAlgorithm):
    """
    Based on:
    
    "A Framework for Real-time Path Planning in Changing Environments",
    Peter Leven, Seth Hutchinson (2002)
    """
    
    progress_name = "DRM"

    def __init__(self, rng, occupancy, access_points, interaction_points, height, width,
                 max_dist, max_neighbors, robot_radius, num_buffer_segments, recorder=None,
                 print_progress=True):

        self.rng = rng
        self.occupancy = np.asarray(occupancy, dtype=np.uint8).copy()
        self.height = height
        self.width = width
        self.max_dist = float(max_dist)
        self.max_neighbors = int(max_neighbors)
        self.robot_radius = float(robot_radius)
        self.num_buffer_segments = int(num_buffer_segments)
        self.recorder = recorder
        self.iteration = 0
        self.print_progress = print_progress

        self.obstacle_space = None
        self._refresh_obstacle_space()

        self.nodes = []
        # indicates whether a node is present in the roadmap (True) or has been removed (False)
        self.node_present = [] # bool (indexed by node index)
        self.edges = []
        self.node_active = [] # bool (indexed by node index)
        self.edge_active = {} # (edge_tuple, bool)
        self.adjacency = defaultdict(set) # (node_index, set(neighbour_indices))

        self.kdtree = None
        self.kdtree_indices = np.empty(0, dtype=int)
 
        # tracks which roadmap nodes and edges are on which occupancy cells
        self.phi_n = defaultdict(set) # (cell, set(node_indices))
        self.phi_a = defaultdict(set) # (cell, set(edge_tuples))

        self.special_nodes = {} # (point, node_index) mapping for special points
        
        self.access_points = access_points
        self.interaction_points = interaction_points
        
        self.wall_nodes = {} # (row, col) -> node_index
        self.affected_nodes = set()
        self.affected_edges = set()
        
        self.KING_MOVES = (
            (-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1),
        )

    def _rebuild_kdtree(self):
        """Rebuild the nearest-neighbour tree for the current roadmap nodes."""
        self.kdtree_indices = np.flatnonzero(self.node_present)
        if self.kdtree_indices.size == 0:
            self.kdtree = None
            return
        self.kdtree = KDTree(np.asarray([self.nodes[index] for index in self.kdtree_indices]))

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
        if self.robot_radius == 0: # if the robot is a point, check if it is in free space
            return point.disjoint(self.obstacle_space)
        else: # else check if the distance from the robot's center to the nearest obstacle is at least the robot's radius
            return point.distance(self.obstacle_space) >= self.robot_radius

    def _is_path_collision_free(self, start, end):
        """Return whether the robot can traverse the path from start to end collision-free."""
        crossed_cells = buffered_edge_cells(start, end, self.robot_radius, self.height, self.width)
        return all(self.occupancy[row, col] == 0 for row, col in crossed_cells)

    def _get_candidate_neighbors(self, config):
        """Return nearby roadmap nodes in increasing distance order."""

        # if no nodes created yet, return empty list
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
        # if no candidates found, return the nearest node regardless of distance
        if not candidates:
            nearest_position = int(self.kdtree.query(config, k=1)[1])
            return [int(self.kdtree_indices[nearest_position])]
        return candidates

    def _add_node(self, config):
        """Add one node and initialise its DRM-specific state."""
        config = np.asarray(config, dtype=np.float64)
        index = len(self.nodes)
        self.nodes.append(config.copy())
        self.node_present.append(True)
        self.node_active.append(False)
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

    def _get_incident_edges(self, nodes):
        """Return all roadmap edges incident to the given nodes."""
        edges = set()
        for index in nodes:
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
    
    def _repair_active_connectivity(self, min_num_nodes):
        """
        Add roadmap nodes until the active roadmap has one component.
        
        min_num_nodes: minimum number of nodes to add (None: add until connected)
        """
        progress = self._create_progress("Connecting active roadmap")
        if progress is not None:
            progress.start()
        
        num_components = self._num_active_components()
        added_nodes = set()
        added_nodes_num = 0
        added_edges = set()
        while num_components > 1 or (min_num_nodes is not None and added_nodes_num < min_num_nodes):
            c = self._generate_random_free_config()
            candidate_indices = self._get_candidate_neighbors(c)
            c_idx = self._add_node(c)
            added_nodes_num += 1
            for n_idx in candidate_indices:
                self._add_edge(c_idx, n_idx)
            self._rebuild_kdtree()
            incident_edges = self._get_incident_edges({c_idx})
            self._refresh_active_status({c_idx}, incident_edges)
            num_components = self._num_active_components()
            added_nodes.add(c_idx)
            added_edges.update(incident_edges)
            if progress is not None and added_nodes_num % 100 == 0:
                progress.update(added_nodes_num, details=(f"components: {num_components} | nodes: {sum(self.node_present)} | "
                                                             f"edges: {len(self.edges)}"))
        if progress is not None:
            progress.finish(True, (f"added nodes: {added_nodes_num} | nodes: {sum(self.node_present)} | "
                                   f"edges: {len(self.edges)} | components: {num_components}"))
        return added_nodes, added_edges

    def _build_wall_graph(self):
        """
        Adds the free, wall-adjacent cell centres as an 8-connected graph.

        Similar to: "A Randomized Roadmap Method for Path and Manipulation Planning", Amato & Wu (1996) 
        """
        # collect the wall nodes from the occupancy grid and add them to the roadmap
        for row in range(self.height):
            for col in range(self.width):
                if self._is_adjacent_to_wall((row, col)):
                    # add the wall node to the roadmap
                    index = self._add_node(np.array([col + 0.5, row + 0.5], dtype=np.float64))
                    self.wall_nodes[(row, col)] = index

        self._connect_wall_nodes(self.wall_nodes)
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

    def _connect_wall_nodes(self, cells):
        """Add all missing valid King's Graph edges for the given wall cells."""
        new_edges = set()
        for row, col in cells:
            index = self.wall_nodes[(row, col)]
            for row_offset, col_offset in self.KING_MOVES:
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                neighbour_index = self.wall_nodes.get((neighbour_row, neighbour_col))
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
                        # add free cells that may become wall cells and wall nodes that may need removal
                        if self.occupancy[cell] == 0 or cell in self.wall_nodes:
                            neighbourhood_cells.add(cell)

        return neighbourhood_cells

    def _update_wall_graph(self, changed_cells):
        """Update the wall graph based on changed occupancy cells."""
        
        # get the set of cells that are considered affected by the changes in occupancy
        potentially_affected_cells = self._get_wall_update_candidate_cells(changed_cells)
        
        # collect all wall nodes that are no longer on wall cells and remove them from the roadmap
        removed_indices = set()
        for cell in potentially_affected_cells:
            # if cell is a wall node but no longer a wall cell, mark it for removal
            if cell in self.wall_nodes and not self._is_adjacent_to_wall(cell):
                removed_indices.add(self.wall_nodes[cell])
        self._remove_nodes(removed_indices)

        # add new wall nodes for any newly created wall cells
        added_nodes = set()
        for cell in potentially_affected_cells:
            if self._is_adjacent_to_wall(cell) and cell not in self.wall_nodes:
                row, col = cell
                index = self._add_node(np.array([col + 0.5, row + 0.5], dtype=np.float64))
                self.wall_nodes[cell] = index
                added_nodes.add(index)
        if added_nodes:
            self._rebuild_kdtree()
        
        # remove edges incident to affected wall nodes
        affected_nodes = set()
        for cell in potentially_affected_cells:
            if cell in self.wall_nodes:
                affected_nodes.add(self.wall_nodes[cell])
        for index in affected_nodes:
            incident_edges = [tuple(sorted((index, neighbour))) for neighbour in self.adjacency[index]]
            for edge in incident_edges:
                self._remove_edge(edge)

        affected_wall_cells = [cell for cell, index in self.wall_nodes.items() if index in affected_nodes]
        new_edges = self._connect_wall_nodes(affected_wall_cells)
        wall_indices = set(self.wall_nodes.values())

        # get the indices of the permanent nodes that are not wall nodes, to connect the new wall nodes to them
        permanent_indices = np.asarray([index for index, present in enumerate(self.node_present) if present and index not in wall_indices], dtype=int)
        # convert the permanent nodes into a KDTree for efficient nearest-neighbor queries
        permanent_tree = (KDTree(np.asarray([self.nodes[index] for index in permanent_indices], dtype=np.float64)) if permanent_indices.size > 0 else None)
        
        # connect the affected wall nodes to the permanent roadmap nodes
        candidate_count = min(self.max_neighbors, len(permanent_indices))
        for index in affected_nodes:
            if permanent_tree is not None:
                distances, positions = permanent_tree.query(self.nodes[index], k=candidate_count, distance_upper_bound=self.max_dist)
                for distance, position in zip(np.atleast_1d(distances), np.atleast_1d(positions)):
                    if np.isfinite(distance): # if the candidate is within max_dist
                        candidate_index = int(permanent_indices[int(position)])
                        # if the robot's swept footprint is obstacle-free, add the edge and move to the next wall node.
                        if self._is_path_collision_free(self.nodes[index], self.nodes[candidate_index]):
                            new_edges.add(self._add_edge(index, candidate_index))
                            break
                        
        self._map_roadmap_elements(added_nodes, new_edges)
        return affected_nodes, new_edges


    def _remove_nodes(self, removed_indices):
        """Remove nodes without changing the remaining roadmap indices."""
        removed_indices = {index for index in removed_indices if self.node_present[index]}
        if not removed_indices:
            return
        for edge in self._get_incident_edges(removed_indices):
            self._remove_edge(edge)
        for index in removed_indices:
            self._remove_mapping(self.phi_n, index, self._get_node_footprint_cells(index))
            self.node_present[index] = False
            self.node_active[index] = False

        self.special_nodes = {
            point: index for point, index in self.special_nodes.items() if index not in removed_indices
        }
        self.wall_nodes = {
            cell: index for cell, index in self.wall_nodes.items() if index not in removed_indices
        }
        self._rebuild_kdtree()


    def _update_special_points(self, access_points, interaction_points, map_new=True):
        """Update the special points in the roadmap, optionally mapping newly added elements to occupancy cells."""
        progress = self._create_progress("Updating special points")
        self.access_points = list(access_points)
        self.interaction_points = list(interaction_points)
        current_points = list(dict.fromkeys(self.access_points + self.interaction_points))
        if progress is not None:
            total_special_points = len(current_points)
            progress.start(f"target: {total_special_points} special points")
            processed_special_points = 0

        # determine which special points have been removed and remove their nodes and edges from the roadmap
        removed_points = set(self.special_nodes).difference(current_points)
        removed_indices = set()
        for point in removed_points:
            if point in self.special_nodes:
                removed_indices.add(self.special_nodes[point])
        self._remove_nodes(removed_indices)

        # add new special points to the roadmap
        new_nodes = set()
        for point in current_points:
            index = self.special_nodes.get(point)
            if index is None:
                candidate_indices = self._get_candidate_neighbors(point)
                index = self._add_node(point)
                for neighbour_index in candidate_indices:
                    self._add_edge(index, neighbour_index)
                self._rebuild_kdtree()
                self.special_nodes[point] = index
                new_nodes.add(index)
            if progress is not None:
                processed_special_points += 1
                if (processed_special_points % 10 == 0
                        or processed_special_points == total_special_points):
                    progress.update(processed_special_points, total_special_points)
        if map_new:
            new_edges = self._get_incident_edges(new_nodes)
            self._map_roadmap_elements(new_nodes, new_edges)
        if progress is not None:
            progress.finish(
                True,
                f"nodes: {sum(self.node_present)} | edges: {len(self.edges)}",
                completed=total_special_points,
                total=total_special_points,
            )

    def _get_node_footprint_cells(self, index):
        """Return the set of cells that intersect the given roadmap node's footprint."""
        x, y = self.nodes[index]
        radius = self.robot_radius

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
        # footprint = self._path_footprint(self.nodes[edge[0]], self.nodes[edge[1]])
        cells = set(buffered_edge_cells(self.nodes[edge[0]], self.nodes[edge[1]], self.robot_radius, self.height, self.width))
        cells.difference_update(self._get_node_footprint_cells(edge[0]))
        cells.difference_update(self._get_node_footprint_cells(edge[1]))
        return cells

    def _map_node_to_occupancy_cell(self, index):
        """Map the given roadmap node to its footprint's occupancy cell."""
        for cell in self._get_node_footprint_cells(index):
            self.phi_n[cell].add(index)

    def _map_edge_to_occupancy_cells(self, edge):
        """Map the given roadmap edge to its footprint's occupancy cells."""
        for cell in self._get_edge_footprint_cells(edge):
            self.phi_a[cell].add(edge)

    def _map_roadmap_elements(self, nodes, edges):
        """Map the given roadmap nodes and edges to occupancy cells."""
        for index in nodes:
            self._map_node_to_occupancy_cell(index)
        for edge in edges:
            self._map_edge_to_occupancy_cells(edge)

    def _get_mapped_roadmap_items(self, cells):
        """Return the roadmap nodes and edges mapped to the given occupancy cells."""
        nodes = {index for cell in cells for index in self.phi_n.get(cell, ())}
        edges = {edge for cell in cells for edge in self.phi_a.get(cell, ())}
        return nodes, edges

    def _build_workspace_to_roadmap_mapping(self):
        """Map all roadmap nodes and edges to their footprint's occupancy cells."""
        self.phi_n.clear()
        self.phi_a.clear()
        self._map_roadmap_elements(np.flatnonzero(self.node_present), self.edges)
        self._refresh_active_status(np.flatnonzero(self.node_present), self.edges)

    def preprocessing_step(self, min_num_nodes=None):
        """Build the initial roadmap and map all nodes and edges to their occupancy cells."""
        start_time = self._start_generation_time()
        self.affected_nodes.clear()
        self.affected_edges.clear()
        progress = self._create_progress("Preprocessing roadmap")
        if progress is not None:
            preprocessing_steps = 4
            progress.start(f"target: {preprocessing_steps} stages")
        
        self._build_wall_graph()
        if progress is not None:
            progress.update(1, preprocessing_steps, "wall graph built")
        # self._repair_active_connectivity()
        self._update_special_points(self.access_points, self.interaction_points, map_new=False)
        if progress is not None:
            progress.update(2, preprocessing_steps, "special points updated")
        self._build_workspace_to_roadmap_mapping()
        if progress is not None:
            progress.update(3, preprocessing_steps, "workspace mapped to roadmap")

        repair_nodes, repair_edges = self._repair_active_connectivity(min_num_nodes)
        self._map_roadmap_elements(repair_nodes, repair_edges)
        if progress is not None:
            progress.update(4, preprocessing_steps, "connectivity repaired")

        self.build_graph()
        
        self._record_generation_time(self._generation_seconds(start_time), "generation")
        
        self.record_graph(phase="preprocessing")
        
        if progress is not None:
            progress.finish(True, completed=preprocessing_steps, total=preprocessing_steps)

    def _refresh_active_status(self, nodes, edges):
        """Refresh roadmap status using the circular robot footprint against raw obstacles."""
        
        for node_index in nodes:
            if self.node_present[node_index]:
                self.node_active[node_index] = self._is_config_collision_free(self.nodes[node_index])

        for edge in edges:
            if edge in self.edge_active:
                edge_is_collision_free = self._is_path_collision_free(self.nodes[edge[0]], self.nodes[edge[1]])
                both_nodes_are_active = (self.node_active[edge[0]] and self.node_active[edge[1]])
                self.edge_active[edge] = (both_nodes_are_active and edge_is_collision_free)

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_interaction_points, min_num_nodes=None):
        """Apply changed cells and update only the affected roadmap regions."""
        start_time = self._start_generation_time()

        self.affected_nodes.clear()
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
        new_special_points = (set(new_access_points) | set(new_interaction_points)).difference(self.special_nodes)

        # update special points and map new elements before the wall update consults cell mappings
        self._update_special_points(new_access_points, new_interaction_points)

        # update the wall-graph nodes and edges affected by the edits
        affected_wall_nodes, new_wall_edges = self._update_wall_graph(changed_cells)

        # update the active/inactive status of all nodes and edges affected by cell changes
        affected_nodes, affected_edges = self._get_mapped_roadmap_items(changed_cells)

        # add all edges connected to affected nodes to the affected set
        affected_edges.update(self._get_incident_edges(affected_nodes))

        # determine which special nodes and edges are affected by the edits
        special_nodes = set(self.special_nodes.values())
        new_special_nodes = {
            self.special_nodes[point] for point in new_special_points if point in self.special_nodes
        }
        changed_special_edges = self._get_incident_edges(new_special_nodes)
        special_edges = self._get_incident_edges(special_nodes)

        # refresh the active/inactive status of all affected nodes and edges, including special points
        refreshed_nodes = affected_nodes | special_nodes | affected_wall_nodes
        refreshed_edges = affected_edges | special_edges | new_wall_edges
        self._refresh_active_status(refreshed_nodes, refreshed_edges)

        # repair the active roadmap connectivity if it has become disconnected
        repair_nodes, repair_edges = self._repair_active_connectivity(min_num_nodes)
        self._map_roadmap_elements(repair_nodes, repair_edges)

        # record the affected nodes and edges for highlighting
        self.affected_nodes = refreshed_nodes | repair_nodes
        self.affected_edges = affected_edges | new_wall_edges | changed_special_edges | repair_edges
        
        # record the updated graph
        self.build_graph()
        
        self._record_generation_time(self._generation_seconds(start_time), "update")
        self.record_graph(phase="online_update")

    def build_graph(self):
        """Return the roadmap of active nodes and edges as a NetworkX graph."""
        graph = nx.Graph()
        for index, config in enumerate(self.nodes):
            if self.node_present[index] and self.node_active[index]:
                graph.add_node(index, pos=tuple(config), active=self.node_active[index])

        for edge in self.edges:
            if self.edge_active[edge]:
                graph.add_edge(edge[0], edge[1], active=self.edge_active[edge])
        return graph

    def record_graph(self, phase="manual"):
        """Save the roadmap for the existing DRM plotter."""
        if self.recorder is None:
            return None

        nodes = (np.asarray(self.nodes, dtype=np.float64) if self.nodes else np.empty((0, 2), dtype=np.float64))

        save_progress = self._create_progress(f"Saving graph to disk ({phase})")
        if save_progress is not None:
            save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="drm_graph",
                data={
                    "occupancy": self.occupancy.copy(),
                    "robot_radius": self.robot_radius,
                    "num_buffer_segments": self.num_buffer_segments,
                    "access_points": sorted(self.access_points),
                    "interaction_points": sorted(self.interaction_points),
                    "height": self.height,
                    "width": self.width,
                    "nodes": nodes,
                    "node_present": np.asarray(self.node_present, dtype=bool),
                    "node_active": np.asarray(self.node_active, dtype=bool),
                    "edges": sorted(self.edges),
                    "edge_active": {edge: self.edge_active[edge] for edge in sorted(self.edges)},
                    "affected_nodes": sorted(self.affected_nodes),
                    "affected_edges": sorted(self.affected_edges),
                    "mapped_node_cells": sum(len(nodes) for nodes in self.phi_n.values()),
                    "mapped_arc_cells": sum(len(edges) for edges in self.phi_a.values()),
                    "phase": phase,
                },
                plot=True,
                iteration=self.iteration,
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
