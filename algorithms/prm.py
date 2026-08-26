from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from shapely.geometry import Point
import networkx as nx
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from artifacts.progress import GenerationProgress


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for efficient component tracking.
    (R. E. Tarjan, "Efficiency of a Good But Not Linear Set Union Algorithm", 1975)
    """
    
    def __init__(self, n):
        self.parent = np.arange(n) # each node is its own parent initially
        self.rank = np.zeros(n, dtype=int) # rank for union by rank optimization
        
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
        
        # union by rank
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
        return len(set(self.find(i) for i in range(num_nodes)))


class PRM:
    """
    "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces",
    Lydia E. Kavraki, Petr Svestka, Jean-Claude Latombe and Mark H. Overmars (1996)
    """
    
    def __init__(self, rng, free_space, obstacle_space, occupancy, access_points, io_points, bounds,
                 eps, max_dist, max_neighbors, agv_radius,
                 buffer_segments, recorder=None):

        self.rng = rng # random number generator
        self.free_space = free_space # free space (shapely Polygon)
        self.obstacle_space = obstacle_space # obstacle space (shapely Polygon)
        self.occupancy = occupancy # occupancy grid: 0 = free, 1 = simple obstacle, 2 = unit load
        self.access_points = access_points # access points (y, x)
        self.io_points = io_points # io points (y, x)
        self.special_points = np.array(np.concatenate([io_points, access_points])) # access points ++ io points
        self.bounds = bounds # x min, y min, x max, y max
        self.eps = eps # step size for path discretization
        self.max_dist = max_dist # maximum distance to consider for neighbors
        self.max_neighbors = max_neighbors # maximum number of neighbors to consider for connection
        self.agv_radius = agv_radius # radius of the AGV for collision costraints
        self.buffer_segments = buffer_segments # segments per rounded-corner quadrant
        self.recorder = recorder
        self.iteration = 0
        
        # initialize with empty nodes (pre-allocate capacity)
        self.nodes = np.empty((100, 2), dtype=np.float64)
        self.num_nodes = 0  # Track actual number of nodes
        
        # KDTree for efficient neighbor search (will be built on first addition)
        self.kdtree = None
        
        # unionind for efficient component tracking (will be expanded as needed)
        self.union_find = UnionFind(100)
        
        # track failures and attempts per node
        self.failure_count = defaultdict(int)
        self.attempt_count = defaultdict(int)
        
        # save adjacency, to build graph at the end
        self.adjacency = defaultdict(set)

        # store paths for edges created during random-bounce walks
        self.edge_paths = {}

    def _generate_random_free_config(self, max_attempts = 1000):
        """Generates a random configuration in the free space using rejection sampling."""
        
        x_min, y_min, x_max, y_max = self.bounds
        for _ in range(max_attempts):
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            point = Point(x, y)
            
            if self.free_space.covers(point):
                return np.array([x, y], dtype=np.float64)
        raise RuntimeError(f"Failed to generate a random free configuration after {max_attempts} attempts.")


    def _local_planner(self, config1, config2):
        """Checks if a straight-line path between two configurations is collision-free."""
        
        # discretize the straight line between config1 and config2
        delta = config2 - config1
        distance = np.linalg.norm(delta)
        num_steps = int(np.ceil(distance / self.eps))
        t_vals = np.linspace(0, 1, num_steps + 1)
        
        # interpolation
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
        return float(np.linalg.norm(config2 - config1))

    def _get_candidate_neighbors(self, config):
        """Returns candidates from current roadmap in increasing distance order."""

        # if no vertices created yet, return empty list
        if self.kdtree is None or self.num_nodes == 0:
            return []

        # number of neighbors to query
        k = min(self.max_neighbors, self.num_nodes)
        # get distances and indices of the nearest candidates within max_dist
        distances, indices = self.kdtree.query(config, k=k, distance_upper_bound=self.max_dist)
        # make sure distances and indices are 1D arrays (query returns scalars if k=1)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        # filter out infinite distances (no neighbor found within max_dist)
        candidates = [int(index) for distance, index in zip(distances, indices) if np.isfinite(distance)]
        return candidates

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
        
        # expand union-find if needed
        if new_idx >= len(self.union_find.parent):
            old_size = len(self.union_find.parent)
            new_size = old_size + 100
            
            new_parent = np.arange(new_size)
            new_parent[:old_size] = self.union_find.parent
            self.union_find.parent = new_parent
            
            new_rank = np.zeros(new_size, dtype=int)
            new_rank[:old_size] = self.union_find.rank
            self.union_find.rank = new_rank
        
        return new_idx
    
    def construction_step(self, num_nodes):
        """Builds initial roadmap without special points."""
        
        # progress tracking
        progress = GenerationProgress("PRM", "Generating initial roadmap", self.iteration)
        progress.start(f"target: {num_nodes} sampling attempts")
        
        nodes_added = 0
        for iteration in range(num_nodes):
            
            # generate random free configuration
            c = self._generate_random_free_config()

            # get candidate neighbors of c
            candidate_indices = self._get_candidate_neighbors(c)

            # add c to nodes
            c_idx = self._add_node(c)
            nodes_added += 1

            # try to connect in order of increasing distance
            for n_idx in candidate_indices:

                # if not already in same component (avoids cycles)
                if not self.union_find.find(c_idx) == self.union_find.find(n_idx):

                    # track attempts
                    self.attempt_count[c_idx] += 1
                    self.attempt_count[n_idx] += 1

                    # check if can be connected via local planner (\delta(c, n) = True)
                    if self._local_planner(self.nodes[c_idx], self.nodes[n_idx]):
                        # if yes, connect new node to neighbor and update component status
                        self._connect_nodes(c_idx, n_idx)
                    else:
                        # else, mark failure and continue
                        self.failure_count[c_idx] += 1
                        self.failure_count[n_idx] += 1

            # update KDTree after added nodes
            self._update_kdtree()

            # progress updates
            if (iteration + 1) % 100 == 0 or iteration + 1 == num_nodes:
                num_components = self.union_find.get_num_components(self.num_nodes)
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                progress.update(
                    iteration + 1,
                    num_nodes,
                    f"nodes: {self.num_nodes}/{num_nodes} ({self.num_nodes / num_nodes * 100:.1f}%) | "
                    f"edges: {num_edges} | components: {num_components}",
                )

        # progress finish
        num_components = self.union_find.get_num_components(self.num_nodes)
        num_edges = sum(len(v) for v in self.adjacency.values()) // 2
        progress.finish(True, f"generated {self.num_nodes} nodes and {num_edges} edges | components: {num_components}")
    
    def _random_bounce_walk(self, start_config, walk_length, max_bounces):
        """
        Generates configurations via random-bounce walk.
        Picks random direction an moves until hitting obstacle, or max steps reached.
        """
        # warehouse bounds
        x_min, y_min, x_max, y_max = self.bounds
        current = start_config.copy()

        # all checked configurations
        path = [current.copy()]
        remaining_steps = int(np.ceil(walk_length / self.eps))
        bounces = 0
        # stop when walk length or max bounces reached
        while remaining_steps > 0 and bounces < max_bounces:
            angle = self.rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            
            # move in the chosen direction until hitting obstacle or max steps reached
            while remaining_steps > 0:
                next_pos = current + direction * self.eps
                # stop if left warehouse bounds
                if not (x_min <= next_pos[0] <= x_max and y_min <= next_pos[1] <= y_max):
                    break
                next_point = Point(next_pos[0], next_pos[1])
                # stop if hit obstacle
                if not self.free_space.covers(next_point):
                    break
                current = next_pos
                remaining_steps -= 1
            # append the last valid position (before hitting something, or reaching max steps) to the path
            path.append(current.copy())
            bounces += 1

        return current, np.asarray(path)

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

    def connect_special_points(self):
        """Attempts to connect special points (access + IO points) to the roadmap."""

        # progress tracking
        total_special_points = len(self.special_points)
        progress = GenerationProgress("PRM", "Connecting special points", self.iteration)
        progress.start(f"target: {total_special_points} special points")

        # for each special point
        for point_number, config in enumerate(self.special_points, start=1):

            # get candidate neighbors of the special point
            candidate_indices = self._get_candidate_neighbors(config)

            # add special point to nodes
            node_idx = self._add_node(config)

            for neighbor_idx in candidate_indices:
                # skip if already in same component
                if not self.union_find.find(node_idx) == self.union_find.find(neighbor_idx):

                    # check if can be connected via local planner
                    self.attempt_count[node_idx] += 1
                    self.attempt_count[neighbor_idx] += 1

                    if self._local_planner(self.nodes[node_idx], self.nodes[neighbor_idx]):
                        # if yes, connect new node to neighbor
                        self._connect_nodes(node_idx, neighbor_idx)
                    else:
                        # else, mark failure and continue
                        self.failure_count[node_idx] += 1
                        self.failure_count[neighbor_idx] += 1

            self._update_kdtree()

            if point_number % 10 == 0 or point_number == total_special_points:
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                progress.update(point_number, total_special_points, f"nodes: {self.num_nodes} | edges: {num_edges}")

        num_edges = sum(len(v) for v in self.adjacency.values()) // 2
        progress.finish(True, f"processed {total_special_points} special points | nodes: {self.num_nodes} | edges: {num_edges}")

    def expansion_step(self, bounce_walk_length, max_bounces):
        """
        Expansion step: densifies difficult regions until all components and special points are connected.
        """
        progress = GenerationProgress("PRM", "Expanding roadmap", self.iteration)
        progress.start("until all components are connected")

        num_exp_nodes = self.num_nodes

        weights = np.array([(self.failure_count[i] / self.attempt_count[i] if self.attempt_count[i] > 0 else 0.0) for i in range(num_exp_nodes)], dtype=np.float64)

        # normalize weights to sum to 1 (or make uniform if all weights are zero)
        total_weight = weights.sum()
        if total_weight > 0:
            weights /= total_weight
        else:
            weights = np.full(num_exp_nodes, 1.0 / num_exp_nodes)

        expansions = 0
        while True:
            num_components = self.union_find.get_num_components(self.num_nodes)

            # if all points are connected, map generation is complete
            if num_components == 1:
                break

            # progress updates
            if expansions % 10 == 0:
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                progress.update(expansions, details=f"nodes: {self.num_nodes} | edges: {num_edges} | components: {num_components}")

            # select node proportionally to failure ratio
            node_idx = self.rng.choice(num_exp_nodes, p=weights)
            # print(f"Selected node {self.nodes[node_idx]} for expansion (failure ratio: {weights[node_idx]:.4f})")
            
            # expand this node with random-bounce walk
            config, bounce_path = self._random_bounce_walk(self.nodes[node_idx], walk_length=bounce_walk_length, max_bounces=max_bounces)

            # select candidates to connect the endpoint of the bounce walk to
            candidate_indices = self._get_candidate_neighbors(config)

            # add the endpoint of the bounce walk to the roadmap with multi-edge path to the starting node of the bounce walk
            new_idx = self._add_node(config)
            self._connect_nodes(node_idx, new_idx, path=bounce_path)

            # update tree
            self._update_kdtree()

            # try the endpoint against the previously selected candidates, track attempts and failures
            for neighbor_idx in candidate_indices:
                if not self.union_find.find(new_idx) == self.union_find.find(neighbor_idx):

                    self.attempt_count[new_idx] += 1
                    self.attempt_count[neighbor_idx] += 1
                    if self._local_planner(self.nodes[new_idx], self.nodes[neighbor_idx]):
                        self._connect_nodes(new_idx, neighbor_idx)
                    else:
                        self.failure_count[new_idx] += 1
                        self.failure_count[neighbor_idx] += 1

            expansions += 1
            num_edges = sum(len(v) for v in self.adjacency.values()) // 2
            progress.update(expansions, details=f"nodes: {self.num_nodes} | edges: {num_edges} | " f"components: {self.union_find.get_num_components(self.num_nodes)}")
                # self.visualize(title=f"Expansion step {expansions} | Nodes: {self.num_nodes} | Components: {self.union_find.get_num_components(self.num_nodes)}")

            if self.recorder is not None and expansions % 1000 == 0:
                nodes, edges = self._expanded_graph_snapshot()
                self.recorder.save(data_type="prm_graph", data={
                    "occupancy": self.occupancy,
                    "agv_radius": self.agv_radius,
                    "buffer_segments": self.buffer_segments,
                    "access_points": self.access_points,
                    "io_points": self.io_points,
                    "bounds": self.bounds,
                    "nodes": nodes,
                    "edges": edges,
                    "num_nodes": self.num_nodes,
                }, plot=True, algorithm="prm", iteration=self.iteration)

        # update KDTree
        self._update_kdtree()
        
        num_components = self.union_find.get_num_components(self.num_nodes)
        num_edges = sum(len(v) for v in self.adjacency.values()) // 2
        success = num_components == 1
        progress.finish(success, f"expansions: {expansions} | nodes: {self.num_nodes} | edges: {num_edges} | components: {num_components}")
        return success
        
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
                    # connect the path nodes to 
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
        graph = self.build_graph()
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
        return self.recorder.save(
            data_type="prm_graph",
            data={
                "occupancy": self.occupancy,
                "agv_radius": self.agv_radius,
                "buffer_segments": self.buffer_segments,
                "access_points": self.access_points,
                "io_points": self.io_points,
                "bounds": self.bounds,
                "nodes": nodes,
                "edges": edges,
                "num_nodes": self.num_nodes,
            },
            plot=True,
        )
