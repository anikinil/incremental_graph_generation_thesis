
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from artifacts.progress import GenerationProgress
from algorithms.geometry import buffered_edge_cells


class DRM:
    """
    Based on:
    
    "A Framework for Real-time Path Planning in Changing Environments",
    Peter Leven, Seth Hutchinson (2002)
    """
    
    # TODO use agv_radius + eps for collision checking, like in PRM
    

    def __init__(self, rng, occupancy, access_points, io_points, height, width, max_dist, max_neighbors, agv_radius, num_buffer_segments, recorder=None):

        self.rng = rng
        self.occupancy = np.asarray(occupancy, dtype=np.uint8).copy()
        self.height = height
        self.width = width
        self.max_dist = float(max_dist)
        self.max_neighbors = int(max_neighbors)
        self.agv_radius = float(agv_radius)
        self.edge_clearance_epsilon = 1e-6 # small epsilon to avoid floating-point precision issues when checking if the robot's swept footprint is collision-free
        self.num_buffer_segments = int(num_buffer_segments)
        self.recorder = recorder
        self.iteration = 0

        self.empty_space = box(0, 0, self.width, self.height)
        self.obstacle_space = None
        self._refresh_obstacle_space()

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
        
        self.wall_vertices = {} # (row, col) -> vertex_index
        self.affected_vertices = set()
        self.affected_edges = set()
        
        self.KING_MOVES = (
            (-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1),
        )

    def _refresh_obstacle_space(self):
        """Rebuild the unbuffered workspace obstacle geometry."""
        occupied_cells = []
        for row in range(self.height):
            for col in range(self.width):
                cell = self.occupancy[row, col]
                if cell != 0:
                    occupied_cells.append(box(col, row, col + 1, row + 1)) # swap row and col to get x and y
        self.obstacle_space = unary_union(occupied_cells)

    def _point_footprint(self, config):
        """Return a conservative polygonal envelope of the robot footprint at config."""
        point = Point(config)
        buffered_point = point.buffer(self._mapping_buffer_radius(), self.num_buffer_segments)
        return buffered_point

    # def _path_footprint(self, start, end):
    #     """Return a conservative polygonal envelope of the robot's swept footprint."""
    #     path = LineString([start, end])
    #     return path.buffer(self._mapping_buffer_radius(), quad_segs=self.num_buffer_segments)

    def _mapping_buffer_radius(self):
        """Return a radius whose polygonal Shapely buffer contains the true circular footprint."""
        segments = self.num_buffer_segments
        return self.agv_radius / np.cos(np.pi / (4 * segments))

    def _is_config_collision_free(self, config):
        """Return whether the robot can occupy config without hitting an obstacle."""
        point = Point(config)
        if self.agv_radius == 0: # if the robot is a point, check if it is in free space
            return point.disjoint(self.obstacle_space)
        else: # else check if the distance from the robot's center to the nearest obstacle is at least the robot's radius
            return point.distance(self.obstacle_space) >= self.agv_radius

    def _is_path_collision_free(self, start, end):
        """Return whether the robot can traverse the path from start to end collision-free."""
        path = LineString([start, end])
        if not self.empty_space.covers(path):
            return False
        if self.agv_radius == 0:
            return path.disjoint(self.obstacle_space)
        return path.distance(self.obstacle_space) >= self.agv_radius

    def _update_kdtree(self):
        self.kdtree = KDTree(np.asarray(self.vertices)) if self.vertices else None

    def _add_vertex(self, config, connect=True, update_kdtree=True):
        """Add one vertex and connect it to nearby vertices in the empty workspace."""
        config = np.asarray(config, dtype=np.float64)
        neighbours = []

        # find nearest neighbours of the new vertex
        if connect and self.kdtree is not None: # on the first vertex, there is no kdtree yet, so skip this step
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
            # if self.empty_space.covers(line):
            self._add_edge(index, neighbour)

        if update_kdtree:
            self._update_kdtree()
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

        for cell in self._get_edge_footprint_cells(edge):
            mapped_edges = self.phi_a.get(cell)
            if mapped_edges is not None:
                mapped_edges.discard(edge)
                if not mapped_edges:
                    del self.phi_a[cell]

        self.edges.remove(edge)
        self.adjacency[edge[0]].discard(edge[1])
        self.adjacency[edge[1]].discard(edge[0])
        del self.edge_active[edge]
        return True

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
            config = self._generate_random_free_config()            
            # index = self._add_vertex(config, update_kdtree=False)
            index = self._add_vertex(config)
            incident_edges = self._get_incident_edges({index})
            self._refresh_active_status({index}, incident_edges)
            num_components = self._num_active_components()
            added_vertices.add(index)
            added_edges.update(incident_edges)
            added_vertices_num += 1
            if added_vertices_num % 100 == 0:
                progress.update(added_vertices_num, details=(f"components: {num_components} | vertices: {len(self.vertices)} | "
                                                             f"edges: {len(self.edges)}"))
        progress.finish(True, (f"added vertices: {added_vertices_num} | vertices: {len(self.vertices)} | "
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
                if self._is_wall_cell((row, col)):
                    index = self._add_vertex(np.array([col + 0.5, row + 0.5], dtype=np.float64), connect=False, update_kdtree=False)
                    self.wall_vertices[(row, col)] = index

        self._update_kdtree()

        # add the King's Graph edges between the wall vertices
        for (row, col), index in self.wall_vertices.items(): # for all wall vertices
            for row_offset, col_offset in self.KING_MOVES: # for all neighbors
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                neighbour_cell = (neighbour_row, neighbour_col)
                neighbour_index = self.wall_vertices.get(neighbour_cell)
                if neighbour_index is not None and neighbour_index > index: # if neighbour is a wall vertex and not already processed
                    if (abs(row_offset) + abs(col_offset) == 2 # if neighbour is diagonal
                        # and would need to cut a corner to pass through one or two adjacent obstacle cells
                        and (self.occupancy[row, neighbour_col] != 0 or self.occupancy[neighbour_row, col] != 0)):
                            continue # ignore

                    self._add_edge(index, neighbour_index)

    def _is_wall_cell(self, cell):
        """Return whether a cell is adjacent to a free cell."""
        row, col = cell
        if self.occupancy[row, col] == 0: # cell not free -> not a wall cell
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset != 0 or col_offset != 0: # skip the cell itself
                        neighbour_row = row + row_offset
                        neighbour_col = col + col_offset
                        if (0 <= neighbour_row < self.height and # if neighbor in bounds
                            0 <= neighbour_col < self.width and
                            self.occupancy[neighbour_row, neighbour_col] != 0): # and occupied
                            return True # wall cell
        return False

    def _get_wall_update_candidate_cells(self, changed_cells):
        """Return the set cells that may be wall cells affected by changes in occupancy."""
        # get the neighborhood of changed cells
        neighbordood_cells = set()
        # for all changed cells
        for row, col in changed_cells:
            # for whole 8-neighborhood of changed cell (including the changed cell itself)
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    neighbour_row = row + row_offset
                    neighbour_col = col + col_offset
                    if 0 <= neighbour_row < self.height and 0 <= neighbour_col < self.width:
                        neighbordood_cells.add((neighbour_row, neighbour_col)) # add neihgbor to potentailly affected wall cells
        # get the wall cells that are affected by the changed layout
        index_to_wall_cell = {index: cell for cell, index in self.wall_vertices.items()}
        wall_cells = set()
        # for all changed cells
        for cell in changed_cells:
            # check if the cell is a wall vertex and get its index
            for index in self.phi_n.get(cell, ()):
                wall_cell = index_to_wall_cell.get(index)
                if wall_cell is not None:
                    wall_cells.add(wall_cell)
            # check if the cell is part of any wall edge and get the indices of those edges
            for edge in self.phi_a.get(cell, ()):
                for index in edge:
                    wall_cell = index_to_wall_cell.get(index)
                    if wall_cell is not None:
                        wall_cells.add(wall_cell)

        return neighbordood_cells | wall_cells

    def _update_wall_graph(self, changed_cells):
        """Update the wall graph based on changed occupancy cells."""
        
        # get the set of cells that are considered affected by the changes in occupancy
        potentially_affected_cells = self._get_wall_update_candidate_cells(changed_cells)
        
        # collect all wall vertices that are no longer on wall cells and remove them from the roadmap
        removed_indices = set()
        for cell in potentially_affected_cells:
            # if cell is a wall vertex but no longer a wall cell, mark it for removal
            if cell in self.wall_vertices and not self._is_wall_cell(cell):
                removed_indices.add(self.wall_vertices[cell])
        if removed_indices:
            self._remove_vertices(removed_indices)

        # add new wall vertices for any newly created wall cells
        added_vertices = set()
        for cell in potentially_affected_cells:
            if self._is_wall_cell(cell) and cell not in self.wall_vertices:
                row, col = cell
                index = self._add_vertex(np.array([col + 0.5, row + 0.5], dtype=np.float64), connect=False, update_kdtree=False)
                self.wall_vertices[cell] = index
                added_vertices.add(index)
        if added_vertices:
            self._update_kdtree()
        
        # remove edges incident to affected wall vertices
        affected_vertices = set()
        for cell in potentially_affected_cells:
            if cell in self.wall_vertices:
                affected_vertices.add(self.wall_vertices[cell])
        for index in affected_vertices:
            incident_edges = [tuple(sorted((index, neighbour))) for neighbour in self.adjacency[index]]
            for edge in incident_edges:
                self._remove_edge(edge)

        # add new edges for newly created wall vertices
        new_edges = set()
        for (row, col), index in self.wall_vertices.items():
            if index in affected_vertices:
                for row_offset, col_offset in self.KING_MOVES:
                    neighbour_row = row + row_offset
                    neighbour_col = col + col_offset
                    neighbour_index = self.wall_vertices.get((neighbour_row, neighbour_col))
                    if neighbour_index is None:
                        continue
                    edge = tuple(sorted((index, neighbour_index)))
                    # check if edge already exists in the roadmap, to avoid duplicates
                    if edge in self.edge_active: # check in edge_active, since it is a set (O(1) lookup)
                        continue
                    # if the edge is diagonal, check if it would cut a corner through an occupied cell and if yes skip it
                    if (abs(row_offset) + abs(col_offset) == 2 and
                        (self.occupancy[row, neighbour_col] != 0 or self.occupancy[neighbour_row, col] != 0)):
                        continue
                    new_edges.add(self._add_edge(index, neighbour_index))
        wall_indices = set(self.wall_vertices.values())

        # get the indices of the permanent vertices that are not wall vertices, to connect the new wall vertices to them
        permanent_indices = []
        for index in range(len(self.vertices)):
            if index not in wall_indices:
                permanent_indices.append(index)
        permanent_indices = np.asarray(permanent_indices, dtype=int)
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
                        
        # map the added vertices and edges to occupancy cells
        for index in added_vertices:
            self._map_vertex_to_occupancy_cell(index)
        for edge in new_edges:
            self._map_edge_to_occupancy_cells(edge)
        return affected_vertices, new_edges


    def _remove_vertices(self, removed_indices):
        """Remove vertices and reindex the remaining roadmap."""
        removed_indices = set(removed_indices)
        print("Removing vertices")
        if not removed_indices:
            print("No vertices to remove.")
            return
        print(f"Removing {len(removed_indices)} vertices from the roadmap.")
        # get indices of vertices to keep and create a mapping from old to new indices
        kept_indices = [index for index in range(len(self.vertices)) if index not in removed_indices]
        index_map = {old_index: new_index for new_index, old_index in enumerate(kept_indices)}

        # reindex vertices
        self.vertices = [self.vertices[index] for index in kept_indices]
        self.vertex_active = [self.vertex_active[index] for index in kept_indices]

        # reindex adjacency list
        old_adjacency = self.adjacency
        self.adjacency = defaultdict(set)
        for old_index in kept_indices:
            new_index = index_map[old_index]
            for neighbour in old_adjacency[old_index]:
                if neighbour in index_map:
                    self.adjacency[new_index].add(index_map[neighbour])

        # reindex edges and their active status
        old_edges = self.edges
        old_edge_active = self.edge_active
        self.edges = []
        self.edge_active = {}
        for left, right in old_edges:
            if left in index_map and right in index_map:
                edge = tuple(sorted((index_map[left], index_map[right])))
                self.edges.append(edge)
                self.edge_active[edge] = old_edge_active.get((left, right), False)

        # reindex the occupancy-cell mappings for vertices and edges
        remapped_phi_n = defaultdict(set)
        for cell, indices in self.phi_n.items():
            remapped_phi_n[cell] = {index_map[index] for index in indices if index in index_map}
        self.phi_n = remapped_phi_n
        remapped_phi_a = defaultdict(set)
        for cell, edges in self.phi_a.items():
            for left, right in edges:
                if left in index_map and right in index_map:
                    remapped_phi_a[cell].add(tuple(sorted((index_map[left], index_map[right]))))
        self.phi_a = remapped_phi_a

        # reindex special vertices
        special_vertices_remapped = {}
        for point, index in self.special_vertices.items():
            if index in index_map:
                special_vertices_remapped[point] = index_map[index]
        self.special_vertices = special_vertices_remapped

        # reindex wall vertices
        wall_vertices_remapped = {}
        for cell, index in self.wall_vertices.items():
            if index in index_map:
                wall_vertices_remapped[cell] = index_map[index]
        self.wall_vertices = wall_vertices_remapped
        self._update_kdtree()
        print(f"Removed {len(removed_indices)} vertices. Remaining vertices: {len(self.vertices)}. Remaining edges: {len(self.edges)}.")

    def _remove_special_vertices(self, points):
        """Remove special-point vertices and their incident edges from the roadmap."""
        removed_indices = set()
        for point in points:
            if point in self.special_vertices:
                removed_indices.add(self.special_vertices[point])
        self._remove_vertices(removed_indices)

    def _update_special_points(self, access_points, io_points, map_new=True):
        """Update the special points in the roadmap, optionally mapping newly added elements to occupancy cells."""
        self.access_points = list(access_points)
        self.io_points = list(io_points)
        current_points = list(dict.fromkeys(self.access_points + self.io_points))

        # determine which special points have been removed and remove their vertices and edges from the roadmap
        removed_points = set(self.special_vertices).difference(current_points)
        self._remove_special_vertices(removed_points)

        # add new special points to the roadmap and map them to occupancy cells (if requested)
        count = 0
        total = len(current_points)
        for point in current_points:
            index = self.special_vertices.get(point)
            if index is None:
                print(f"Adding special points to roadmap: {count + 1}/{total} ({(count + 1) / total * 100:.2f}%) ({point})", end="\r", flush=True)
                index = self._add_vertex(point)
                self.special_vertices[point] = index
                count += 1
                if map_new:
                    self._map_vertex_to_occupancy_cell(index)
                    for edge in self._get_incident_edges({index}):
                        self._map_edge_to_occupancy_cells(edge)

    def _get_intersecting_cells(self, geometry):
        """Return the set of cells that intersect the given geometry."""

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
    
    def _determine_edge_cells(self, vertex1, vertex2):
        """Returns grid cells crossed by the buffered edge between two vertices."""

        return buffered_edge_cells(vertex1, vertex2, self.agv_radius, self.height, self.width, self.edge_clearance_epsilon)
    

    def _get_vertex_footprint_cells(self, index):
        """Return the set of cells that intersect the given roadmap vertex's footprint."""
        return self._get_intersecting_cells(self._point_footprint(self.vertices[index]))

    def _get_edge_footprint_cells(self, edge):
        """Return the set of cells that intersect the given roadmap edge's footprint."""
        # footprint = self._path_footprint(self.vertices[edge[0]], self.vertices[edge[1]])
        # cells = self._get_intersecting_cells(footprint)
        cells = set(self._determine_edge_cells(self.vertices[edge[0]], self.vertices[edge[1]]))
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
        # self.phi_n.clear()
        # self.phi_a.clear()
        for index in range(len(self.vertices)):
            self._map_vertex_to_occupancy_cell(index)
        for edge in self.edges:
            self._map_edge_to_occupancy_cells(edge)

        self._refresh_active_status(range(len(self.vertices)), self.edges)

    def preprocessing_step(self, min_num_vertices=None):
        """Build the initial roadmap and map all vertices and edges to their occupancy cells."""
        self.affected_vertices.clear()
        self.affected_edges.clear()
        
        self._build_wall_graph()
        print("WALL GRAPH BUILT")
        # self._repair_active_connectivity()
        self._update_special_points(self.access_points, self.io_points, map_new=False)
        print("SPECIAL POINTS ADDED", flush=True)
        self._build_workspace_to_roadmap_mapping()
        print("WORKSPACE MAPPED TO ROADMAP", flush=True)

        repair_vertices, repair_edges = self._repair_active_connectivity(min_num_vertices)
        print(f"REPAIRED CONNECTIVITY")
        for index in repair_vertices:
            self._map_vertex_to_occupancy_cell(index)
        for edge in repair_edges:
            self._map_edge_to_occupancy_cells(edge)

        self.record_graph(phase="preprocessing")

    def _refresh_active_status(self, vertices, edges):
        """Refresh roadmap status using the circular robot footprint against raw obstacles."""
        
        for vertex_index in vertices:
            self.vertex_active[vertex_index] = self._is_config_collision_free(self.vertices[vertex_index])

        for edge in edges:
            edge_is_collision_free = self._is_path_collision_free(self.vertices[edge[0]], self.vertices[edge[1]])
            both_vertices_are_active = (self.vertex_active[edge[0]] and self.vertex_active[edge[1]])
            self.edge_active[edge] = (both_vertices_are_active and edge_is_collision_free)

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_io_points, min_num_vertices=None):
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
        new_special_points = (set(new_access_points) | set(new_io_points)).difference(self.special_vertices)

        # update special points and map new elements before the wall update consults cell mappings
        self._update_special_points(new_access_points, new_io_points)

        # update the wall-graph vertices and edges affected by the edits
        affected_wall_vertices, new_wall_edges = self._update_wall_graph(changed_cells)

        # determine which special vertices and edges are affected by the edits
        special_vertices = set(self.special_vertices.values())
        new_special_vertices = {
            self.special_vertices[point]
            for point in new_special_points
            if point in self.special_vertices
        }
        changed_special_edges = self._get_incident_edges(new_special_vertices)
        special_edges = self._get_incident_edges(special_vertices)

        # update the active/inactive status of all vertices and edges affected by cell changes
        affected_vertices = set()
        affected_edges = set()
        for cell in changed_cells:
            affected_vertices.update(self.phi_n.get(cell, ()))
            affected_edges.update(self.phi_a.get(cell, ()))

        # add all edges connected to affected vertices to the affected set
        affected_edges.update(self._get_incident_edges(affected_vertices))

        # refresh the active/inactive status of all affected vertices and edges, including special points
        refreshed_vertices = affected_vertices | special_vertices | affected_wall_vertices
        refreshed_edges = affected_edges | special_edges | new_wall_edges
        self._refresh_active_status(refreshed_vertices, refreshed_edges)

        # repair the active roadmap connectivity if it has become disconnected
        repair_vertices, repair_edges = self._repair_active_connectivity(min_num_vertices)
        for index in repair_vertices:
            self._map_vertex_to_occupancy_cell(index) # map the new vertices to their cells
        for edge in repair_edges:
            self._map_edge_to_occupancy_cells(edge) # map the new edges to their cells

        # record the affected vertices and edges for highlighting
        self.affected_vertices = refreshed_vertices | repair_vertices
        self.affected_edges = affected_edges | new_wall_edges | changed_special_edges | repair_edges
        
        # record the updated graph
        self.record_graph(phase="online_update")

    def build_graph(self):
        """Return the roadmap of active vertices and edges as a NetworkX graph."""
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
                    "io_points": sorted(self.io_points),
                    "height": self.height,
                    "width": self.width,
                    "vertices": vertices,
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
