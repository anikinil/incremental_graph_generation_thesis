from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import LineString, Point, Polygon
import networkx as nx
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for efficient component tracking.
    (R. E. Tarjan, "Efficiency of a Good But Not Linear Set Union Algorithm", 1975)
    """
    
    def __init__(self, n):
        self.parent = np.arange(n) # each node is its own parent initially
        self.rank = np.zeros(n, dtype=int) # rank for union by rank optimization
        self.num_nodes = 0
        
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
    
    def get_num_components(self):
        """Gets the number of connected components."""
        return len(set(self.find(i) for i in range(self.num_nodes)))


class PRM:
    # TODO citation
    
    def __init__(self, rng, free_space, obstacle_space, occupancy, access_points, io_points, bounds,
                 eps = 0.1, max_dist = 5.0, max_neighbors = 30, agv_radius = 0.0):

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
    
    def _generate_random_free_config(self, max_attempts = 100):
        """Generates a random configuration in the free space using rejection sampling."""
        
        x_min, y_min, x_max, y_max = self.bounds
        for _ in range(max_attempts):
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            point = Point(x, y)
            
            if self.free_space.contains(point):
                return np.array([x, y], dtype=np.float64)
        
        return None
    
    def _discretize_path(self, config1, config2):
        """Discretizes the straight-line path between two configurations into steps of size eps."""
        
        delta = config2 - config1
        distance = np.linalg.norm(delta)
        
        if distance == 0:
            return np.array([config1])
        
        num_steps = int(np.ceil(distance / self.eps))
        t_vals = np.linspace(0, 1, num_steps + 1)
        
        # interpolation
        configs = config1[np.newaxis, :] + t_vals[:, np.newaxis] * delta[np.newaxis, :]
        return configs
    
    def _local_planner(self, config1, config2):
        """Checks if a straight-line path between two configurations is collision-free."""
        
        configs = self._discretize_path(config1, config2)
        
        # check each discretized configuration
        for config in configs:
            point = Point(config[0], config[1])
            # if a point in obstacle space, reject path
            if not self.free_space.contains(point):
                return False
        return True
    
    def _distance_function(self, config1, config2):
        """Returns euclidean distance between configurations."""
        return float(np.linalg.norm(config2 - config1))
    
    def _get_candidate_neighbors(self, new_config):
        """Returns candidate neighbors for a new configuration using KDTree."""
        
        if self.kdtree is None or self.num_nodes == 0:
            return []
        
        # query all neighbors within max_dist
        k = min(self.max_neighbors, self.num_nodes)
        distances, indices = self.kdtree.query(new_config, k=k, 
                                                distance_upper_bound=self.max_dist)
        
        # filter out invalid results (distance > max_dist gives inf)
        if isinstance(distances, np.ndarray):
            valid_mask = np.isfinite(distances)
            return indices[valid_mask].tolist()
        elif np.isfinite(distances):
            return [indices]
        else:
            return []
    
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
            
            self.union_find.num_nodes = new_idx + 1
            
            new_parent = np.arange(new_size)
            new_parent[:old_size] = self.union_find.parent
            self.union_find.parent = new_parent
            
            new_rank = np.zeros(new_size, dtype=int)
            new_rank[:old_size] = self.union_find.rank
            self.union_find.rank = new_rank
        
        return new_idx
    
    def construction_step(self, num_nodes):
        """Builds initial roadmap without special points."""
        
        print("==== CONSTRUCTION STEP ====")
        
        nodes_added = 0
        for iteration in range(num_nodes):
            
            # generate random free configuration
            config = self._generate_random_free_config()
            if config is None:
                continue
            
            # add new configuration to nodes
            node_idx = self._add_node(config)
            nodes_added += 1
            
            # rebuild KDTree periodically for accurate neighbor queries
            if node_idx % 100 == 0 and node_idx > 0:
                self._update_kdtree()
            
            # get candidate neighbors
            candidate_indices = self._get_candidate_neighbors(config)
            # try to connect in order of increasing distance
            for neighbor_idx in candidate_indices:
                # track attempt
                self.attempt_count[node_idx] += 1
                self.attempt_count[neighbor_idx] += 1
                
                # skip if already in same component (avoids cycles)
                if self.union_find.find(node_idx) == self.union_find.find(neighbor_idx):
                    continue
                
                # check if can be connected via local planner
                if self._local_planner(self.nodes[node_idx], self.nodes[neighbor_idx]):
                    # if yes, connect new node to neighbor
                    self._connect_nodes(node_idx, neighbor_idx)
                else:
                    # else, mark failure and coninue
                    self.failure_count[node_idx] += 1
                    self.failure_count[neighbor_idx] += 1
            
            # print consruction iteration stats
            if (iteration + 1) % 100 == 0:
                num_components = self.union_find.get_num_components()
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                print(f"  Nodes: {self.num_nodes}/{num_nodes} | Edges: {num_edges} | Components: {num_components}")
        
        # update KDTree
        self._update_kdtree()
        
        print(f"Construction step complete: {self.num_nodes} nodes")
    
    def _random_bounce_walk(self, start_config, walk_length = 5.0, max_bounces = 20):
        """
        Generates configurations via random-bounce walk.
        Picks random direction an moves until hitting obstacle, or max steps reached, then repeats.
        """
        x_min, y_min, x_max, y_max = self.bounds
        current = start_config.copy()
        
        visited = []
        # for every bounce
        for bounce in range(max_bounces):
            # pick random direction
            angle = self.rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            
            # move until hitting obstacle or reaching max number of steps allowed in one bounce
            step_size = self.eps
            num_steps = int(walk_length / step_size)
            for step in range(num_steps):
                next_pos = current + direction * step_size
                
                # check if within bounds
                if not (x_min <= next_pos[0] <= x_max and y_min <= next_pos[1] <= y_max):
                    break
                
                # check if in free space
                next_point = Point(next_pos[0], next_pos[1])
                if not self.free_space.contains(next_point):
                    break
                
                # move to next step
                current = next_pos
            
            visited.append(current.copy())
        return visited
    
    def _connect_nodes(self, node_a, node_b):
        """Connects two nodes and update data structures."""
        # store new edge
        self.adjacency[node_a].add(node_b)
        self.adjacency[node_b].add(node_a)
        # store unnion for efficient component check
        self.union_find.union(node_a, node_b)
    
    def expansion_step(self, bounce_walk_length = 5.0, max_bounces = 10):
        """
        Expansion step: densifies difficult regions until all components and special points are connected.
        """
        print("\n=== EXPANSION STEP ===")
        
        expansions = 0
        while True:
            num_components = self.union_find.get_num_components()
            if num_components <= 1:
                break
            
            if expansions % 10 == 0:
                num_edges = sum(len(v) for v in self.adjacency.values()) // 2
                print(f"  Expansions: {expansions} | Nodes: {self.num_nodes} | Edges: {num_edges} | Components: {num_components}")
            
            # if num_components == 2:
            #     self.visualize(title=f"Expansion step {expansions} | Nodes: {self.num_nodes} | Edges: {num_edges} | Components: {num_components}")
            
            # compute weights for node selection based on failure ratio (higher weight -> more likely)
            weights = np.zeros(self.num_nodes, dtype=np.float64)
            for i in range(self.num_nodes):
                if self.attempt_count[i] > 0:
                    weights[i] = self.failure_count[i] / (self.attempt_count[i] + 1)
                else:
                    weights[i] = 5.0  # TODO improve
                # if i >= self.num_nodes - len(self.special_points):
                #     weights[i] *= 2.0  # Prioritize special points
            
            # normalize weights
            total_weight = weights.sum()
            if total_weight > 0:
                weights = weights / total_weight
            else:
                # if no difficult nodes, select uniformly
                weights = np.ones(self.num_nodes) / self.num_nodes
            
            # select node proportionally to failure ratio
            node_idx = self.rng.choice(self.num_nodes, p=weights)
            # print(f"Selected node {self.nodes[node_idx]} for expansion (failure ratio: {weights[node_idx]:.4f})")
            
            # expand this node with random-bounce walk
            bounce_points = self._random_bounce_walk(self.nodes[node_idx], walk_length=bounce_walk_length, max_bounces=max_bounces)
            
            prev_idx = node_idx
            # new_connections = []
            for config in bounce_points:
                new_idx = self._add_node(config)
                self._connect_nodes(new_idx, prev_idx)
                
                # rebuild KDTree periodically for accurate neighbor queries
                if new_idx % 100 == 0:
                    self._update_kdtree()
                
                candidate_indices = self._get_candidate_neighbors(config)
                
                for neighbor_idx in candidate_indices:
                    # mark attempt
                    self.attempt_count[new_idx] += 1
                    self.attempt_count[neighbor_idx] += 1
                    
                    # skip if already in same component
                    if self.union_find.find(new_idx) == self.union_find.find(neighbor_idx):
                        continue
                    
                    # check if can be connected via local planner
                    if self._local_planner(self.nodes[new_idx], self.nodes[neighbor_idx]):
                        # if yes, connect nodes
                        self._connect_nodes(new_idx, neighbor_idx)
                        # new_connections.append((new_idx, neighbor_idx))
                    else:
                        # if not, mark failure
                        self.failure_count[new_idx] += 1
                        self.failure_count[neighbor_idx] += 1
                
                # go to next bounce
                prev_idx = new_idx
                
            expansions += 1
        
        # update KDTree
        self._update_kdtree()
        
        print(f"Expansion step complete: {self.num_nodes} nodes")
        
    def smoothing_step(self):
        # J.-P. Laumond, M. Taix, and P. Jacobs, “A motion planner for car-like robots based on a global/local approach,” Proc. IEEE Internat. Workshop Intell. Robot Syst., pp. 765-773. 1990
        # S. Berchtold and B. Glavina, “A scalable optimizer for automatically generated manipulator motions,” Proc. IEEE/RSJ/GI Int. Conj Intelligent Robots and Systems, Miinchen, Germany, pp. 1796-1802, 1994
        return # TODO implement
        
    
    def build_graph(self):
        """Converts roadmap to a NetworkX graph."""
        G = nx.Graph()
        
        for i in range(self.num_nodes):
            G.add_node(i, pos=tuple(self.nodes[i]))
        
        # add edges from adjacency list
        added_edges = set()
        for node_idx, neighbors in self.adjacency.items():
            for neighbor_idx in neighbors:
                edge = tuple(sorted([node_idx, neighbor_idx]))
                if edge not in added_edges:
                    distance = self._distance_function(self.nodes[node_idx],  self.nodes[neighbor_idx])
                    G.add_edge(node_idx, neighbor_idx, weight=distance)
                    added_edges.add(edge)
        
        return G
    
    def visualize(self, title = None):
        _, ax = plt.subplots(figsize=(25, 20))

        width = self.bounds[2] - self.bounds[0]
        height = self.bounds[3] - self.bounds[1]

        BUFFER_COLOR = [0.8, 0.8, 0.8, 1] # light grey
        FREE_SPACE_COLOR = [0, 0, 0, 0]  # transparent
        SIMPLE_OBSTACLE_COLOR = [0.3, 0.3, 0.3, 1] # dark grey
        UNIT_LOAD_COLOR = [0.5, 0.5, 0.5, 1] # grey

        # draw obstacle/free space first (light grey)
        gpd.GeoSeries(self.obstacle_space).plot(ax=ax, color=BUFFER_COLOR, zorder=0)
        gpd.GeoSeries(self.free_space).plot(ax=ax, color=FREE_SPACE_COLOR[0:4], zorder=0)
        
        # draw wall buffer
        ax.add_patch(plt.Rectangle((0, 0), width, self.agv_radius, color=BUFFER_COLOR, zorder=0))
        ax.add_patch(plt.Rectangle((0, height - self.agv_radius), width, self.agv_radius, color=BUFFER_COLOR, zorder=0))
        ax.add_patch(plt.Rectangle((0, 0), self.agv_radius, height, color=BUFFER_COLOR, zorder=0))
        ax.add_patch(plt.Rectangle((width - self.agv_radius, 0), self.agv_radius, height, color=BUFFER_COLOR, zorder=0))

        G = self.build_graph()

        # draw occupancy grid on top
        # [free space (transparent), simple obstacle (dark grey), unit load (grey)]
        cmap = ListedColormap([FREE_SPACE_COLOR, SIMPLE_OBSTACLE_COLOR, UNIT_LOAD_COLOR])
        ax.pcolormesh(self.occupancy, cmap=cmap, edgecolors="black", linewidth=0.5, shading="flat", zorder=1)
        
        # draw edges
        for u, v in G.edges:
            x1, y1 = G.nodes[u]["pos"]
            x2, y2 = G.nodes[v]["pos"]
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=1, zorder=2)

        # draw nodes (color by component)
        components = list(nx.connected_components(G))
        colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
        
        for comp_idx, component in enumerate(components):
            x_coords = [self.nodes[n][0] for n in component]
            y_coords = [self.nodes[n][1] for n in component]
            ax.scatter(x_coords, y_coords, s=15, color=[colors[comp_idx]], zorder=3, alpha=0.7, edgecolors='black', linewidth=1)
            
        # # mark last N added nodes
        # if self.num_nodes > 0:
        #     last_nodes = [self.nodes[i] for i in range(max(0, self.num_nodes - 50), self.num_nodes)]
        #     x_coords = [node[0] for node in last_nodes]
        #     y_coords = [node[1] for node in last_nodes]
        #     colors = plt.cm.autumn(np.linspace(0, 1, len(last_nodes)))
        #     for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        #         ax.scatter(x, y, s=100, color=[colors[i]], zorder=4, linewidth=1.0)

        # access points
        ax.scatter([p[0] for p in self.access_points], [p[1] for p in self.access_points], s=30, color="red", zorder=3)

        # IO points
        ax.scatter([p[0] for p in self.io_points], [p[1] for p in self.io_points], s=50, color="green", zorder=3)

        # grid boundaries
        for x in range(width + 1):
            ax.axvline(x, color="black", linewidth=1, zorder=4)
        for y in range(height + 1):
            ax.axhline(y, color="black", linewidth=1, zorder=4)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.margins(0)
        ax.invert_yaxis()

        plt.title(title if title else "PRM Roadmap")
        plt.show()
   

def main():
    from load_scenario import load_scenario_as_shapely
    scenario = load_scenario_as_shapely()
    
    rng = np.random.default_rng(42)
    
    # parameters
    
    # num_construction_nodes = int((width * height) / occupancy_rate)
    num_construction_nodes = int(scenario["width"] * scenario["height"]) * 5
    max_dist = 0.5 # maximum distance to consider for neighbors
    max_neighbors = 15 # maximum number of neighbors to consider for connection
    eps = 0.1 # step size for path discretization
    
    bounce_walk_length = 0.5
    max_bounces = 50
    
    # build PRM
    bounds = (0, 0, scenario["width"], scenario["height"])
    prm = PRM(
        rng,
        scenario["free_space"],
        scenario["obstacle_space"],
        scenario["occupancy"],
        scenario["access_points"],
        scenario["io_points"],
        bounds,
        eps=eps,
        max_dist=max_dist,
        max_neighbors=max_neighbors,
        agv_radius=scenario["agv_radius"]
    )
        
    # construction step (no special points)
    prm.construction_step(num_construction_nodes)
    prm.visualize(title="PRM Roadmap After Construction Step")
    
    # add special points (access and IO points)
    for point in scenario["access_points"]:
        prm._add_node(np.array(point))
    for point in scenario["io_points"]:
        prm._add_node(np.array(point))
    prm._update_kdtree()
    
    # expand (with special points)
    prm.expansion_step(bounce_walk_length, max_bounces)
    prm.visualize(title="PRM Roadmap After Expansion Step")
    
    # perform smoothing
    prm.smoothing_step()
    prm.visualize(title="PRM Roadmap After Smoothing Step")
    

if __name__ == "__main__":
    main()