from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from shapely.geometry import Point
import networkx as nx
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from logs.progress import ProgressReportingAlgorithm
from algorithms.geometry import navigation_space


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for efficient component tracking.
    (R. E. Tarjan, "Efficiency of a Good But Not Linear Set Union Algorithm", 1975)
    """
    
    def __init__(self, n):
        """
        Initializes the Union-Find structure with n elements (0 to n-1).
        Instead of using make
        """
        self.parent = np.arange(n) # each node is its own parent initially
        self.rank = np.zeros(n, dtype=int) # rank for union by rank optimization

    def ensure_capacity(self, size):
        """Expands storage so element indices below size are valid."""
        old_size = len(self.parent)
        if size <= old_size:
            return

        new_size = max(size, old_size + 100)
        new_parent = np.arange(new_size)
        new_parent[:old_size] = self.parent
        self.parent = new_parent

        new_rank = np.zeros(new_size, dtype=int)
        new_rank[:old_size] = self.rank
        self.rank = new_rank
        
    def find(self, x):
        """Finds the root of the set containing x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Unions two sets. Returns True if they were in different sets."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # union by rank: attach the smaller tree under the larger one
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def get_num_components(self, num_nodes):
        """Gets the number of connected components among the first num_nodes nodes."""
        roots = set()
        for i in range(num_nodes):
            roots.add(self.find(i))
        return len(roots)


class PRM(ProgressReportingAlgorithm):
    """
    "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces",
    Lydia E. Kavraki, Petr Svestka, Jean-Claude Latombe and Mark H. Overmars (1996)
    """
    
    progress_name = "PRM"

    def __init__(self, rng, occupancy, access_points, interaction_points, height, width,
                 eps, max_dist, max_neighbors, robot_radius,
                 num_buffer_segments, recorder=None, print_progress=True):

        self.rng = rng # random number generator
        self.occupancy = occupancy # occupancy grid: 0 = free, 1 = simple obstacle, 2 = unit load
        self.height, self.width = height, width # grid dimensions
        self.eps = eps # discretization step for local planner
        self.free_space = None
        self.access_points = access_points # access points (y, x)
        self.interaction_points = interaction_points # interaction points (y, x)
        # self.special_points = np.array(np.concatenate([interaction_points, access_points])) # interaction points ++ access points
        self.special_points = np.asarray([*interaction_points, *access_points], dtype=float) # interaction points ++ access points
        self.max_dist = max_dist # maximum distance to consider for neighbors
        self.max_neighbors = max_neighbors # maximum number of neighbors to consider for connection
        self.robot_radius = robot_radius # radius of the robot for collision costraints
        if self.robot_radius >= 0.5:
            raise ValueError(f"Robot's radius {self.robot_radius} is too large for the occupancy grid (max 0.5).")
        self.buffer = robot_radius # radius of the robot for collision costraints
        self.num_buffer_segments = num_buffer_segments # segments per rounded-corner quadrant
        self.recorder = recorder
        self.print_progress = bool(print_progress)
        self.iteration = 0
        
        # initialize with empty nodes (pre-allocate capacity)
        self.nodes = np.empty((100, 2), dtype=np.float64)
        self.num_nodes = 0  # Track actual number of nodes
        
        # KDTree for efficient neighbor search (will be built on first addition)
        self.kdtree = None
        
        # unionind for efficient component tracking (will be expanded as needed)
        self.union_find = UnionFind(100)
        
        # track failures and attempts per node
        # self.failure_count = defaultdict(int)
        # self.attempt_count = defaultdict(int)
        
        # save adjacency, to build graph at the end
        self.adjacency = defaultdict(set)

        # store paths for edges created during random-bounce walks
        self.edge_paths = {}

        self.wall_nodes = {} # (row, col) -> node_index
        self.graph = None

        self.KING_MOVES = (
            (-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1),
        )

    def _generate_random_free_config(self, max_attempts = 1000):
        """Generates a random configuration in the free space using rejection sampling."""
        
        for _ in range(max_attempts):
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            point = Point(x, y)
            
            if self.free_space.covers(point):
                return np.array([x, y], dtype=np.float64)
        raise RuntimeError(f"Failed to generate a random free configuration after {max_attempts} attempts.")


    def _local_planner(self, config1, config2):
        """Checks if a straight-line path between two configurations is collision-free."""
        
        # discretize the straight line between config1 and config2
        distance = self._distance_function(config1, config2)
        num_steps = int(np.ceil(distance / self.eps))
        t_vals = np.linspace(0, 1, num_steps + 1)
        
        # interpolation
        delta = config2 - config1
        configs = config1[np.newaxis, :] + t_vals[:, np.newaxis] * delta[np.newaxis, :]

        # check each discretized configuration
        for config in configs:
            point = Point(config[0], config[1])  # shapely uses (x, y)
            # if a point in obstacle space, reject path
            if not self.free_space.covers(point):
                return False
        return True
    
    def _distance_function(self, config1, config2):
        """Returns euclidean distance between configurations."""
        return np.linalg.norm(config2 - config1)

    def _get_candidate_neighbors(self, config):
        """Returns candidates from current roadmap in increasing distance order."""

        # if no nodes created yet, return empty list
        if self.kdtree is None or self.num_nodes == 0:
            return []

        # number of neighbors to query
        k = min(self.max_neighbors, self.num_nodes)
        # get distances and indices of the nearest candidates within max_dist
        distances, indices = self.kdtree.query(config, k=k, distance_upper_bound=self.max_dist)
        # make sure distances and indices are 1D arrays (query returns scalars if k=1)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        # return indices of candidates that are within max_dist and are valid nodes
        return [int(index) for distance, index in zip(distances, indices) if np.isfinite(distance)]

    def _update_kdtree(self):
        """Rebuilds KDTree with current nodes."""
        if self.num_nodes > 0:
            self.kdtree = KDTree(self.nodes[:self.num_nodes])
        else:
            self.kdtree = None
    
    def _add_node(self, config):
        """Adds a new node and updates data structures."""
        
        new_idx = self.num_nodes
        
        # expand nodes array if needed
        if new_idx >= len(self.nodes):
            new_nodes = np.empty((len(self.nodes) + 100, 2), dtype=np.float64)
            new_nodes[:self.num_nodes] = self.nodes[:self.num_nodes]
            self.nodes = new_nodes
        
        self.nodes[new_idx] = config
        self.num_nodes += 1

        self.union_find.ensure_capacity(self.num_nodes)
        
        return new_idx

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
        return False # no occupied neighbour found -> not a wall cell

    def _build_wall_graph(self):
        """
        Adds the free, wall-adjacent cell centres as an 8-connected graph.

        Similar to: "A Randomized Roadmap Method for Path and Manipulation Planning", Amato & Wu (1996) 
        """

        # add the free, wall-adjacent cell centres to the roadmap
        for row in range(self.height):
            for col in range(self.width):
                if self._is_adjacent_to_wall((row, col)):
                    config = np.array([col + 0.5, row + 0.5], dtype=np.float64)
                    self.wall_nodes[(row, col)] = self._add_node(config)

        # add the King's Graph edges between the wall nodes
        for (row, col), node_idx in self.wall_nodes.items(): # for all wall nodes
            for row_offset, col_offset in self.KING_MOVES: # for all neighbours
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                neighbour_idx = self.wall_nodes.get((neighbour_row, neighbour_col))
                if neighbour_idx is not None and neighbour_idx > node_idx: # if neighbour is a wall node and not already processed
                    # if the edge is diagonal, check if it would cut a corner through an occupied cell and if yes skip it
                    if (abs(row_offset) + abs(col_offset) == 2 and
                        (self.occupancy[row, neighbour_col] != 0 or self.occupancy[neighbour_row, col] != 0)):
                        continue

                    # # the local path must also respect the buffered obstacle space
                    # path = LineString([self.nodes[node_idx], self.nodes[neighbour_idx]])
                    # if self.free_space.covers(path): # line between the two nodes is collision-free
                    self._connect_nodes(node_idx, neighbour_idx)

        # build the tree after all wall nodes have been added
        self._update_kdtree()

    def _num_components(self):
        """Returns the number of connected components in the roadmap."""
        return self.union_find.get_num_components(self.num_nodes)

    def construction_step(self, min_num_nodes, avoid_cycles):
        """Builds initial roadmap without special points."""
        progress = self._create_progress("Generating initial roadmap")
        if progress is not None:
        
            progress.start(f"target: {min_num_nodes} sampling attempts after wall graph")
        start_time = self._start_generation_time()

        # get free space as shapely Polygon for sampling and local planner checks
        self.free_space, _ = navigation_space(self.occupancy, self.buffer, self.num_buffer_segments)
        
        # build wall graph
        self._build_wall_graph()
        
        # connect special points to the roadmap
        self._connect_special_points(avoid_cycles=avoid_cycles)
        
        added_nodes_num = 0
        num_components = self._num_components()
        while num_components > 1 or (min_num_nodes is not None and added_nodes_num < min_num_nodes):
            
            # generate random free configuration
            c = self._generate_random_free_config()

            # get candidate neighbors of c
            candidate_indices = self._get_candidate_neighbors(c)

            # add every collision-free sampled configuration to the roadmap
            c_idx = self._add_node(c)
            added_nodes_num += 1

            # try to connect in order of increasing distance
            for n_idx in candidate_indices:
                # if not already in same component (avoids cycles)
                if not avoid_cycles or self.union_find.find(c_idx) != self.union_find.find(n_idx):
                    # # track attempts
                    # self.attempt_count[c_idx] += 1
                    # self.attempt_count[n_idx] += 1

                    # check if can be connected via local planner (\delta(c, n) = True)
                    if self._local_planner(self.nodes[c_idx], self.nodes[n_idx]):
                        # if yes, connect new node to neighbor and update component status
                        self._connect_nodes(c_idx, n_idx)
                    # else:
                    #     # else, mark failure and continue
                    #     self.failure_count[c_idx] += 1
                    #     self.failure_count[n_idx] += 1

            # update KDTree after added nodes
            self._update_kdtree()
            num_components = self._num_components()

            if progress is not None and added_nodes_num % 100 == 0:
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                progress.update(added_nodes_num, details=(f"components: {num_components} | nodes: {self.num_nodes} | "
                                                         f"edges: {num_edges}"))
        
        if progress is not None:
            num_edges = sum(len(v) for v in self.adjacency.values()) // 2
            progress.finish(True, (f"added nodes: {added_nodes_num} | nodes: {self.num_nodes} | "
                                   f"edges: {num_edges} | components: {num_components}"))

        self.graph = self.build_graph()
        self._record_generation_time(self._generation_seconds(start_time), "generation")

    
    # def _random_bounce_walk(self, start_config, walk_length, max_bounces):
    #     """
    #     Generates configurations via random-bounce walk.
    #     Picks random direction an moves until hitting obstacle, or max steps reached.
    #     """
    #     # warehouse bounds
    #     current = start_config.copy()

    #     # all checked configurations
    #     path = [current.copy()]
    #     remaining_steps = int(np.ceil(walk_length / step_size))
    #     bounces = 0
    #     # stop when walk length or max bounces reached
    #     while remaining_steps > 0 and bounces < max_bounces:
    #         angle = self.rng.uniform(0, 2 * np.pi)
    #         direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            
    #         # move in the chosen direction until hitting obstacle or max steps reached
    #         while remaining_steps > 0:
    #             next_pos = current + direction * step_size
    #             # stop if left warehouse bounds
    #             if not (0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height):
    #                 break
    #             next_point = Point(next_pos[0], next_pos[1])
    #             # stop if hit obstacle
    #             if not self.free_space.covers(next_point):
    #                 break
    #             current = next_pos
    #             remaining_steps -= 1
    #         # append the last valid position (before hitting something, or reaching max steps) to the path
    #         path.append(current.copy())
    #         bounces += 1

    #     return current, np.asarray(path)

    def _connect_nodes(self, node_a, node_b, path=None):
        """Connects two nodes and update data structures."""
        # store new edge
        self.adjacency[node_a].add(node_b)
        self.adjacency[node_b].add(node_a)
        # store union for efficient component check
        self.union_find.union(node_a, node_b)

        if path is not None:
            edge = tuple(sorted((node_a, node_b)))
            edge_path = np.asarray(path, dtype=np.float64)
            self.edge_paths[edge] = edge_path if edge == (node_a, node_b) else edge_path[::-1].copy()

    def _connect_special_points(self, avoid_cycles=True):
        """Attempts to connect special points (access + interaction points) to the roadmap."""

        progress = self._create_progress("Connecting special points")
        if progress is not None:
            total_special_points = len(self.special_points)
            progress.start(f"target: {total_special_points} special points")
            processed_special_points = 0

        # for each special point
        for config in self.special_points:

            # get candidate neighbors of the special point
            candidate_indices = self._get_candidate_neighbors(config)

            # add special point to nodes
            node_idx = self._add_node(config)

            for neighbor_idx in candidate_indices:
                # skip if already in same component
                if not avoid_cycles or self.union_find.find(node_idx) != self.union_find.find(neighbor_idx):

                    # # check if can be connected via local planner
                    # self.attempt_count[node_idx] += 1
                    # self.attempt_count[neighbor_idx] += 1

                    if self._local_planner(self.nodes[node_idx], self.nodes[neighbor_idx]):
                        # if yes, connect new node to neighbor
                        self._connect_nodes(node_idx, neighbor_idx)
                    # else:
                    #     # else, mark failure and continue
                    #     self.failure_count[node_idx] += 1
                    #     self.failure_count[neighbor_idx] += 1

            self._update_kdtree()

            if progress is not None:
                processed_special_points += 1
                if (processed_special_points % 10 == 0
                        or processed_special_points == total_special_points):
                    num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                    progress.update(
                        processed_special_points,
                        total_special_points,
                        f"nodes: {self.num_nodes} | edges: {num_edges}",
                    )

        if progress is not None:
            num_edges = sum(len(v) for v in self.adjacency.values()) // 2
            progress.finish(
                True,
                f"processed {total_special_points} special points | nodes: {self.num_nodes} | edges: {num_edges}",
                completed=total_special_points,
                total=total_special_points,
            )

        
    def smoothing_step(self):
        # J.-P. Laumond, M. Taix, and P. Jacobs, “A motion planner for car-like robots based on a global/local approach,” Proc. IEEE Internat. Workshop Intell. Robot Syst., pp. 765-773. 1990
        # S. Berchtold and B. Glavina, “A scalable optimizer for automatically generated manipulator motions,” Proc. IEEE/RSJ/GI Int. Conj Intelligent Robots and Systems, Miinchen, Germany, pp. 1796-1802, 1994
        return # TODO implement
        
    
    def build_graph(self):
        """Converts the roadmap to a NetworkX graph with straight edges only."""
        G = nx.Graph()
        
        # add all nodes to the graph
        for i in range(self.num_nodes):
            G.add_node(i, pos=tuple(self.nodes[i]))
        
        next_path_node_idx = self.num_nodes
        added_edges = set() # track edges already added to avoid duplicates
        # for every node, add edges to its neighbors
        for node_idx, neighbors in self.adjacency.items():
            for neighbor_idx in neighbors:
                edge = tuple(sorted([node_idx, neighbor_idx])) # sort to avoid duplicates
                if edge in added_edges:
                    continue # skip if already added
                # check if there is a stored path for this edge (from random-bounce walk)
                path = self.edge_paths.get(edge)
                if path is None:
                    # if no path stored, add straight edge
                    G.add_edge(node_idx, neighbor_idx)
                else:
                    # if path is stored, add intermediate nodes for the path and connect them
                    path = np.asarray(path, dtype=np.float64)
                    path_node_indices = [edge[0]] # nodes to add
                    previous_config = self.nodes[edge[0]] # track previous node to avoid duplicates

                    # turn path nodes created during bounce walks into graph nodes (exclude endpoints)
                    for config in path[1:-1]:
                        if np.array_equal(config, previous_config): # skip, if same as previous node (hit the same wall twice)
                            continue
                        G.add_node(next_path_node_idx, pos=tuple(config), bounce_turn=True)
                        path_node_indices.append(next_path_node_idx)
                        next_path_node_idx += 1
                        previous_config = config
                    # connect the path nodes
                    path_node_indices.append(edge[1])
                    for i in range(len(path_node_indices) - 1):
                        left = path_node_indices[i]
                        right = path_node_indices[i + 1]
                        G.add_edge(left, right)

                added_edges.add(edge)

        return G

    # for visualization during generation, TODO remove later
    def _expanded_graph_snapshot(self):
        """Returns the straight-edge graph representation."""
        graph = self.graph if self.graph is not None else self.build_graph()
        node_ids = sorted(graph.nodes)
        node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
        nodes = np.asarray([graph.nodes[node_id]["pos"] for node_id in node_ids], dtype=np.float64)
        edges = sorted(tuple(sorted((node_id_to_index[left], node_id_to_index[right]))) for left, right in graph.edges)
        return nodes, edges

    def record_graph(self):
        """Records the current roadmap graph."""
        if self.recorder is None:
            return None

        nodes, edges = self._expanded_graph_snapshot()
        save_progress = self._create_progress("Saving graph to disk")
        if save_progress is not None:
            save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="prm_graph",
                data={
                    "occupancy": self.occupancy,
                    "robot_radius": self.robot_radius,
                    "num_buffer_segments": self.num_buffer_segments,
                    "access_points": self.access_points,
                    "interaction_points": self.interaction_points,
                    "height": self.height,
                    "width": self.width,
                    "nodes": nodes,
                    "edges": edges,
                    "num_nodes": self.num_nodes,
                },
                plot=True,
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
