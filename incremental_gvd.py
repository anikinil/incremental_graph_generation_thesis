"""
Lau, Sprunk, Burgard
"Efficient grid-based spatial representations for robot navigation in dynamic environments"
(Robotics and Autonomous Systems, 2013)
"""

import math
import sys

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from collections import defaultdict, deque

from matplotlib.path import Path
from matplotlib.patches import PathPatch
        
import shapely



BUFFER_COLOR = [0.8, 0.8, 0.8, 1] # light grey
BUFFER_RES = 2

SIMPLE_OBSTACLE_COLOR = [0.3, 0.3, 0.3, 1] # dark grey
UNIT_LOAD_COLOR = [0.5, 0.5, 0.5, 1] # grey

VORO_SKELETON_COLOR = [0.3, 0.3, 0.5, 0.7] # blue
AFFECTED_VORO_CELLS_COLOR = [0.8, 0.2, 0.2, 0.7] # red overlay
AFFECTED_DISTANCES_COLOR = [0.9, 0.1, 0.1, 0.3] # red
BRIDGE_COLOR = [0.2, 0.5, 0.2, 0.7] # green


EDGE_COLOR = [1.0, 0.6, 0.2, 1] # orange
VERTEX_COLOR = [0.9, 0.4, 0.1, 1] # dark orange


class BucketPriorityQueue:
    
    def __init__(self, max_distance):
        
        # dictionary of lists for each distance value
        self.buckets = defaultdict(list)
        # bounded by the maximum possible squared distance in the grid
        self.max_distance = max_distance
        # current minimum distance in the queue
        self.current_min_distance = np.inf
    
    def insert(self, cell, distance):
        
        distance = int(distance)  # cast distance to integer
        if distance < self.current_min_distance: # if new cell has smaller distance, update current min distance
            self.current_min_distance = distance
        self.buckets[distance].append(cell) # add cell to the appropriate bucket based on its distance
    
    def pop(self):
        
        # from min distance
        while self.current_min_distance <= self.max_distance:
            # if there are cells in the current min distance bucket
            if self.buckets[self.current_min_distance]:
                # pop a cell from the bucket and return it
                cell = self.buckets[self.current_min_distance].pop()
                return cell, self.current_min_distance
            # update current min distance to next non-empty bucket
            self.current_min_distance += 1
        # if no cells left in any bucket, popping from an empty queue: raise error (should not happen)
        raise IndexError("Priority queue is empty.")
    
    def is_empty(self):
        
        # if all buchets are empty, the queue is empty
        return all(len(cells) == 0 for cells in self.buckets.values())


class IncrementalGVD:
    
    def __init__(self, rng, occupancy_grid, scale_factor, access_points, io_points, agv_radius):
        self.rng = rng

        self.occupancy_grid = occupancy_grid
        self.height, self.width = occupancy_grid.shape
        
        self.scale_factor = scale_factor
        
        self.access_points = access_points
        self.io_points = io_points
        
        self.agv_radius = agv_radius
                
        # increase resolution and add outer walls
        self.increase_resolution()
        self.add_outer_walls()
        
        # distance map: stores distances to closest obstacle for each cell
        self.D = np.full((self.height, self.width), np.inf, dtype=np.float32)
        
        # obstacle reference map: stores (y, x) of closest occupied cell for each cell
        self.obst = np.empty((self.height, self.width), dtype=object)
        print()
        for i in range(self.height):
            for j in range(self.width):
                print(f"\r{' ' * 100}\rInitializing obstacle reference for cell ({i}, {j})", end="", flush=True)
                self.obst[i, j] = (-1, -1)
        print()
        
        # voronoi map: True if cell is on Voronoi diagram
        self.voro = np.zeros((self.height, self.width), dtype=bool)
        print(self.voro.shape)
        
        # queue for wavefront propagation
        self.open = BucketPriorityQueue(max_distance=self.height ** 2 + self.width ** 2)  # max possible squared distance in grid
        
        # flag for raise/lower wavefronts
        self.toRaise = np.zeros((self.height, self.width), dtype=bool)
        
        # track affected cells after dynamic updates for visualization
        self.affected_distance_cells = set() 
        self.affected_voro_cells = set()
        
        self.graph = None # { "vertices": [vid_1: (y, x), vid_2: (y, x), ...], "edges": [(vid_1, vid_2), ...] }
        
        # save voronoi cells added to connect special points to Voronoi skeleton, to recalculate if affected by dynamic updates
        # {sp_1: [(y, x), (y, x), ...], sp_2: [...], ...}
        self.special_point_bridges = np.zeros((self.height, self.width), dtype=bool)
    
        self.iteration = 0 # increased after every layout update
    
    def increase_resolution(self):
        
        old_height, old_width = self.height, self.width
        
        self.height *= self.scale_factor
        self.width *= self.scale_factor
        
        # create a new high-resolution occupancy grid
        new_occupancy_grid = np.zeros((self.height, self.width), dtype=self.occupancy_grid.dtype)
        
        # fill in the high-resolution grid based on the original occupancy values
        for y in range(old_height):
            for x in range(old_width):
                value = self.occupancy_grid[y, x]
                new_occupancy_grid[y*self.scale_factor:(y+1)*self.scale_factor, x*self.scale_factor:(x+1)*self.scale_factor] = value
        
        self.occupancy_grid = new_occupancy_grid
        
        # scale positions of access points and IO points
        self.access_points = [(y * self.scale_factor, x * self.scale_factor, dir) for (y, x, dir) in self.access_points]
        self.io_points = [(y * self.scale_factor, x * self.scale_factor) for (y, x) in self.io_points]
        
        # scale agv_radius
        self.agv_radius *= self.scale_factor
        
    
    def add_outer_walls(self):
        
        # add outer walls to the occupancy grid
        self.height += 2
        self.width += 2
        new_occupancy = np.ones((self.height, self.width), dtype=self.occupancy_grid.dtype)
        new_occupancy[1:-1, 1:-1] = self.occupancy_grid
        self.occupancy_grid = new_occupancy
        
        # adjust access points and IO points
        self.access_points = [(y + 1, x + 1, dir) for (y, x, dir) in self.access_points]
        self.io_points = [(y + 1, x + 1) for (y, x) in self.io_points]
        
    
    def distance(self, p1, p2):
        dy = p1[0] - p2[0]
        dx = p1[1] - p2[1]
        return dy**2 + dx**2
    
    def Adj(self, s):
        y, x = s
        neighbors = []
        if y > 0:
            neighbors.append((y-1, x))  # up
        if y < self.height-1:
            neighbors.append((y+1, x))  # down
        if x > 0:
            neighbors.append((y, x-1))  # left
        if x < self.width-1:
            neighbors.append((y, x+1))  # right
        return neighbors
        
        # """Get 8-connected neighbors (including diagonals)."""
        # neighbors = []
        # for dy in [-1, 0, 1]:
        #     for dx in [-1, 0, 1]:
        #         if dy == 0 and dx == 0:
        #             continue
        #         ny, nx = y + dy, x + dx
        #         if 0 <= ny < self.height and 0 <= nx < self.width:
        #             neighbors.append((ny, nx))
        # return neighbors
    
    def computeDistanceMap(self):
        
        # initialize and enqueue occupied cells
        print()
        for y in range(self.height):
            for x in range(self.width):
                s = (y, x)
                print(f"\r{' ' * 100}\rInitializing cell ({y}, {x}) with occupancy {self.occupancy_grid[y, x]}", end="", flush=True)
                if self.occupancy_grid[y, x] > 0:  # if an object is in the cell
                    self.D[y, x] = 0 # distance to itself is zero
                    self.obst[y, x] = s # is its own closest obstacle
                    self.open.insert(s, 0) # enqueue for wavefront propagation: distance=0, cell=s
                        
        # propagate wavefronts
        print()
        # dist = 0
        while not self.open.is_empty():
            # if self.open.current_min_distance > dist:
            #     dist = self.open.current_min_distance
            #     self.visualize_distance_map(title=dist)
            print(f"\r{' ' * 100}\rProcessing queue with current min distance {self.open.current_min_distance}", end="", flush=True)
            s, _ = self.open.pop() # pop cell with smallest distance
            self.lower(s) # propagate to neighbors and update their distances if closer to obstacle
        print(f"\r{' ' * 100}\rDistance map complete.")
        
        # self.prune_voronoi(connectivity=8) # prune
            
    def lower(self, s):
        
        # get the obstacle reference of the current cell
        s_obst = self.obst[s[0], s[1]]
        
        # for each neighbor of s
        for n in self.Adj(s):
            # calculate distance from neighbor to obstacle
            if not self.toRaise[n[0], n[1]]:  # if neighbor not marked to raise
                d = self.distance(s_obst, n) # calculate distance between current cell's closest obstacle and neighbor cell
                # update neighbor's distance if new distance is smaller than its current distance
                if d < self.D[n[0], n[1]]:
                    # update neighbor's distance
                    self.D[n[0], n[1]] = d
                    # update neighbor's obstacle reference to the same as current cell
                    self.obst[n[0], n[1]] = s_obst
                    # enqueue neighbor to continue wavefront propagation
                    self.open.insert(n, d)
                    
                    self.affected_distance_cells.add(n) # track affected cells for visualization
                else:
                    # threshold = max(self.agv_radius, 1)
                    threshold = self.agv_radius
                    self.checkVoro(s, n, threshold=threshold)  # check Voronoi condition between current cell and neighbors
    
    def setObstacle(self, s, value):
        
        # add new cell to occupancy grid
        self.occupancy_grid[s[0], s[1]] = value
        
        # initialize distance and obstacle reference for new cell
        self.obst[s[0], s[1]] = s
        self.D[s[0], s[1]] = 0
        
        # enqueue new cell for wavefront propagation
        self.open.insert(s, 0)
        
        # track affected cells
        self.affected_distance_cells.add(s)
        self.affected_voro_cells.add(s) 
    
    def removeObstacle(self, s):
        
        # remove obstacle from occupancy grid, while keeping track of its type (for move operations)
        value = self.occupancy_grid[s[0], s[1]]
        self.occupancy_grid[s[0], s[1]] = 0
        
        # clearCell(s): set distance to inf, obstacle reference to (-1, -1) (clear)
        self.D[s[0], s[1]] = np.inf
        self.obst[s[0], s[1]] = (-1, -1)
        
        # enqueue for raise wavefront propagation
        self.toRaise[s[0], s[1]] = True
        
        # add to open queue as source of propagation for raise wavefront
        self.open.insert(s, 0)
        
        # track affected cells
        self.affected_distance_cells.add(s)
        self.affected_voro_cells.add(s) 

        return value  # return the type of obstacle that was removed (for move operations)
    
    def updateDistanceMap(self):

        while not self.open.is_empty():
            # pop cell with smallest distance
            s, _ = self.open.pop()

            # if cell is marked for raise (its closest obstacle was removed), propagate raise wavefront
            if self.toRaise[s[0], s[1]]:
                self.raise_(s)
            else:
                # else: if cell is not marked for raise and its closes obstacle is still valid
                obstacle_ref = self.obst[s[0], s[1]]
                if self.occupancy_grid[obstacle_ref[0], obstacle_ref[1]] > 0:
                    # exclude this cell from voronoi diagram and propagate lower wavefront to update neighbors
                    self.voro[s[0], s[1]] = False
                    if s in self.affected_voro_cells:
                        self.affected_voro_cells.remove(s) # Voronoi cell can be deleted
                    self.lower(s) 
        
        # # determine which cells were affected
        # changed_mask = (previous_D != self.D)
        # # store as tuples
        # self.affected_distance_cells = {(int(y), int(x)) for y, x in np.argwhere(changed_mask)}

    def raise_(self, s):
        
        # for every neighbor of the current cell
        for n in self.Adj(s):
            obstacle_ref = self.obst[n[0], n[1]]
            
            if obstacle_ref != (-1, -1) and not self.toRaise[n[0], n[1]]:  # if neighbor not already cleared or marked for raise
                
                # enqueue with current distance
                self.open.insert(n, self.D[n[0], n[1]])
                
                self.affected_distance_cells.add(n) # track affected cells for visualization
                
                # check if obstacle reference is still valid
                if self.occupancy_grid[obstacle_ref[0], obstacle_ref[1]] == 0: # if neighbor's closest obstacle is no longer occupied
                    # clear neighbor and mark for raise
                    self.D[n[0], n[1]] = np.inf
                    self.obst[n[0], n[1]] = (-1, -1)
                    self.toRaise[n[0], n[1]] = True
                    
        # after processing neighbors, mark current cell as no longer needing raise
        self.toRaise[s[0], s[1]] = False
    
    def checkVoro(self, s, n, threshold):
        
        threshold = threshold ** 2 # all distances are compared squared
        
        obst_s = self.obst[s[0], s[1]]
        obst_n = self.obst[n[0], n[1]]
        
        threshold = 1

        if (self.D[s[0], s[1]] > threshold or self.D[n[0], n[1]] > threshold) \
            and obst_n != (-1, -1) \
            and obst_s != obst_n \
            and obst_s not in self.Adj(obst_n):
        
            # calculate distances
            dist_s_to_obst_n = self.distance(s, obst_n)
            dist_n_to_obst_s = self.distance(n, obst_s)
            
            # add cell with smaller violation
            if dist_s_to_obst_n <= dist_n_to_obst_s:
                self.voro[s] = True
                self.affected_voro_cells.add(s)
            
            if dist_n_to_obst_s <= dist_s_to_obst_n:
                self.voro[n] = True
                self.affected_voro_cells.add(n)

    def apply_obstacle_updates(self, updates, new_access_points, new_io_points):

        # clear affected cells
        self.affected_voro_cells.clear()
        self.affected_distance_cells.clear()

        scaled_updates = []
        for update in updates:
            action = update["action"]
            if action =='add':
                (y, x), value = update["cell"], update["obstacle_type"]
                for x_ in range(self.scale_factor):
                    for y_ in range(self.scale_factor):
                        scaled_updates.append({"action": action, "cell": (y * self.scale_factor + y_, x * self.scale_factor + x_), "obstacle_type": value})
            if action == 'remove':
                y, x = update["cell"]
                for x_ in range(self.scale_factor):
                    for y_ in range(self.scale_factor):
                        scaled_updates.append({"action": action, "cell": (y * self.scale_factor + y_, x * self.scale_factor + x_)})
            elif action == 'move':
                (y_from, x_from), (y_to, x_to) = update["from_cell"], update["to_cell"]
                for x_ in range(self.scale_factor):
                    for y_ in range(self.scale_factor):
                        scaled_updates.append({"action": action, "from_cell": (y_from * self.scale_factor + y_, x_from * self.scale_factor + x_), "to_cell": (y_to * self.scale_factor + y_, x_to * self.scale_factor + x_)})

        # adjust changes to the wall offset
        translated_updates = []
        for update in scaled_updates:
            action = update["action"]
            if action =='add':
                (y, x), value = update["cell"], update["obstacle_type"]
                translated_updates.append({"action": action, "cell": (y + 1, x + 1), "obstacle_type": value})
            if action == 'remove':
                y, x = update["cell"]
                translated_updates.append({"action": action, "cell": (y + 1, x + 1)})
            elif action == 'move':
                (y_from, x_from), (y_to, x_to) = update["from_cell"], update["to_cell"]
                translated_updates.append({"action": action, "from_cell": (y_from + 1, x_from + 1), "to_cell": (y_to + 1, x_to + 1)})

        # apply updates to the occupancy grid
        for update in translated_updates:
            action = update["action"]
            if action == 'add':
                s, value = update["cell"], update["obstacle_type"]
                self.setObstacle(s, value)
            elif action == 'remove':
                s = update["cell"]
                self.removeObstacle(s)
            elif action == 'move':
                s_from, s_to = update["from_cell"], update["to_cell"]
                value = self.removeObstacle(s_from)
                self.setObstacle(s_to, value)
                
        # adjust and update access points
        self.access_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1, dir) for (y, x, dir) in new_access_points]
        # adjust and update IO points (probably usually unchanged)
        self.io_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1) for (y, x) in new_io_points]

        self.special_point_bridges.fill(False)  # clear special bridges, as they need to be recalculated now
        
        self.iteration += 1


    def build_special_point_bridges(self):
        """
        Connects special points to the Voronoi cells by casting a ray of cells 
        from the special point in the direction of the dir label to the closest GVD cell.
        
        NOTE theory, why that seems to alway work: Access point rays always meet a Voronoi cell before an obstacle.
        There is always a Voronoi cell between two obstacles, so there is always a Voronoi cell between the access point's obstacle
        and any other, the ray is headed towards.
        """
        
        dirs = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1)} # dir: (dy, dx)
        
        connected = 0
        for ap in self.access_points:
            
            y, x, dir = ap
            # determine the direction vector based on the dir label
            dir_vec = dirs[dir]
            while True:
                
                try:
                    # get current cell
                    on_voro = self.voro[int(y), int(x)]
                except IndexError:
                    self.visualize_gvd(title="Voronoi Diagram with Special Points Connected")
                    print(F"Access point {ap} ray went out of bounds at ({y}, {x}). Stopping ray.")

                # if on a Voronoi cell, stop the ray
                if on_voro:
                    connected += 1
                    break
                
                # else: add the cell to the bridge
                self.special_point_bridges[int(y), int(x)] = True
                
                # move in the direction of the dir label
                y += dir_vec[0]
                x += dir_vec[1]
                
        print(f"Connected {connected}/{len(self.access_points)} access points to Voronoi skeleton.")
        
        # try to connect similarly to access points, but try each direction and pick the best one:
        # no obstacles encountered, shortest distance
        connected = 0
        for i, iop in enumerate(self.io_points):
            y, x = iop
            found_connection = False
            best_bridge_cells = None
            best_distance = np.inf
            
            for dir_vec in dirs.values():
                curr_x, curr_y = x, y
                current_bridge_cells = []
                while True:
                    on_voro = self.voro[int(curr_y), int(curr_x)]
                    if on_voro: # if on skeleton, stop the ray
                        found_connection = True
                        current_distance = len(current_bridge_cells)
                        if current_distance < best_distance:
                            best_distance = current_distance
                            best_bridge_cells = current_bridge_cells.copy()
                        break
                    # if hit an obstacle, stop searching in this direction
                    if self.occupancy_grid[int(curr_y), int(curr_x)] > 0:
                        break
            
                    # add the current cell to the bridge cells list
                    current_bridge_cells.append((int(curr_y), int(curr_x)))
                    
                    # move in the direction of the dir label
                    curr_y += dir_vec[0]
                    curr_x += dir_vec[1]
                    
            if found_connection:
                connected += 1
                # add the best bridge cells to the Voronoi diagram
                for cell in best_bridge_cells:
                    self.special_point_bridges[cell] = True
            else: 
                print(f"Warning: Could not connect IO point {iop} to Voronoi skeleton in any direction.")
            
        print(f"Connected {connected}/{len(self.io_points)} IO points to Voronoi skeleton.")
        self.visualize_gvd(title="Voronoi Diagram with Special Points Connected")

    def clear_bridges(self):
        self.special_point_bridges.fill(False)

    def _special_point_on_cell(self, cell):
        """
        Checks if a special point is located on the given cell (r, c).
        """
        for sp in self.access_points + self.io_points:
            if (int(sp[0]), int(sp[1])) == cell:
                return True
        return False
    
    # ============= DEBUGGING ===================
    
    def _get_connected_components(self):
        if not self.graph or not self.graph["vertices"]:
            return []
        vertices = set(self.graph["vertices"].keys())
        adjacency = defaultdict(set)
        for v1, v2 in self.graph["edges"]:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        components = []
        while vertices:
            start = vertices.pop()
            component = {start}
            queue = deque([start])
            while queue:
                vertex = queue.popleft()
                for neighbor in adjacency[vertex]:
                    if neighbor in vertices:
                        vertices.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    def _is_graph_connected(self):
        components = self._get_connected_components()
        if len(components) == 1:
            return True
        else:
            return False
    
    def _visualize_graph_components(self):
        components = self._get_connected_components()

        fig, ax = self.draw_grid()

        # edges
        edge_segments = []
        for v1, v2 in self.graph["edges"]:
            y1, x1 = self.graph["vertices"][v1]
            y2, x2 = self.graph["vertices"][v2]
            edge_segments.append([(x1, y1), (x2, y2)])

        edge_collection = LineCollection(edge_segments, colors="blue", linewidths=3, zorder=2)
        ax.add_collection(edge_collection)

        # assign each vertex its component
        vertex_component = {}
        for component_id, component in enumerate(components):
            for vertex_id in component:
                vertex_component[vertex_id] = component_id

        # plot
        for component_id, component in enumerate(components):
            vertices = np.array([(self.graph["vertices"][v][1], self.graph["vertices"][v][0]) for v in component])
            ax.scatter(vertices[:, 0], vertices[:, 1], s=50, zorder=3, label=f"Component {component_id}")

        ax.legend()
        plt.title(f"Graph Components ({len(components)})")
        plt.show()

    # ================================================

    def build_graph(self):
        """
        "Sparse 3D Topological Graphs for Micro-Aerial Vehicle Planning",
        Helen Oleynikova, Zachary Taylor, Roland Siegwart, Juan Nieto (2018)
        
        https://github.com/ethz-asl/mav_voxblox_planning/blob/master/voxblox_skeleton/src/skeleton_generator.cpp
        
        """
        # self.thin() # not working: deleting essential endpoints. TODO remove, or find better solution
        # perhaps modify "Building Skeleton Models via 3D Medial Surface/Axis Thinning Algorithms", T.-C. Lee, R. Kashyap, C.-N. Chu (1994)
        self.build_special_point_bridges()
        bridge_grid = self.special_point_bridges.astype(np.uint8)  # 0 = free cell, 1 = special point bridge
        
        # 0 = free cell, 1 = skeleton cell (on free cell), 2 = occupied cell
        voronoi_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        voronoi_grid[self.voro] = 1
        obstacle_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        obstacle_grid[self.occupancy_grid > 0] = 2
        grid = voronoi_grid + obstacle_grid + bridge_grid  # 0 = free, 1 = skeleton/bridge, 2 = occupied
        
        print(f"Building graph from skeleton with {np.sum(voronoi_grid)} skeleton cells...")

        def neighbors(cell):
            """Returns 8-connected neighbors of a cell that are also part of the skeleton."""
            r, c = cell
            neighbors = []
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if (0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 1):
                    neighbors.append((nr, nc))
            return neighbors

        height, width = grid.shape
            
        # neighbor cell offsets for 8-connectivity
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        # calculate the degree of each skeleton cell (number of neighbors that are also skeleton cells)
        skeleton_points = np.argwhere(grid == 1)
        degree = {}
        for r, c in skeleton_points:
            degree[(r, c)] = len(neighbors((r, c)))

        # the coordinates of skeleton cells with degree!=2 are saved as vertex cells (all of them will later be graph nodes)
        # For block-stacking storage scenario: also include skeleton cells that are on the same cell as a special point,
        # to later easily connect the special points to these verteces.
        vertex_cells = [(r, c) for (r, c), deg in degree.items() if deg != 2 or self._special_point_on_cell((r, c))]
        
        # create mappings between vertex_id and vertex cell coordinates and vice versa
        vertex_id_to_cell_coords = {} # maps from vertex_id to cell coordinates of vertex cells
        cell_coords_to_vertex_id = {} # maps from cell coordinates of vertex cells to vertex_id
        for vertex_id, cell in enumerate(vertex_cells):
            vertex_id_to_cell_coords[vertex_id] = cell
            cell_coords_to_vertex_id[cell] = vertex_id

        # maps from skeleton cell coordinates to the vertex_id of the graph vertex that reached it
        cell_coords_to_origin_id = {}

        # queue for flood-fill propagation from each vertex to trace skeleton paths
        floodfill_queue = deque()

        # initialize the flood-fill queue with the neighbors of each vertex cell
        for id, coords in vertex_id_to_cell_coords.items():
            
            # cells that are already graph vertices are assigned to themselves as their origin
            cell_coords_to_origin_id[coords] = id

            for neighbor in neighbors(coords):
                floodfill_queue.append((neighbor, id))

        # graph edges: touple of vertex_ids (sorted to avoid duplicate edges)
        edges = set()

        # while there are cells to process in the flood-fill queue
        while floodfill_queue:
            
            print(f"\r{' ' * 100}\rBuilding graph. Flood-fill queue size: {len(floodfill_queue)}", end="", flush=True)
            
            # pop the next cell and its source vertex_id from the queue
            curr_cell_coords, source_vertex_id = floodfill_queue.popleft()
            
            # stores the vertex_id of the vertex that reached this cell for later edge creation
            connected_vertex_id = -1
            
            # get the vertex_id of the vertex that reached this cell, or -1 if it hasn't been reached yet
            origin = cell_coords_to_origin_id.get(curr_cell_coords, -1)

            # if the current cell is not a vertex cell
            if curr_cell_coords not in cell_coords_to_vertex_id:

                # if already visited by this flood fill, move on to the next cell in queue
                if origin == source_vertex_id:
                    continue

                # if already visited by a different flood fill
                if origin != -1:
                    # two different vertices have become connected
                    connected_vertex_id = origin
                else: # cell has not been visited yet
                    # mark the cell as visited by this flood fill
                    cell_coords_to_origin_id[curr_cell_coords] = source_vertex_id

                    # add the neighbors of the current cell to the flood-fill queue
                    for neighbor in neighbors(curr_cell_coords):
                        floodfill_queue.append((neighbor, source_vertex_id))
            else: # cell IS a vertex cell
                
                # if the current cell is a vertex cell, get its vertex_id
                connected_vertex_id = cell_coords_to_vertex_id[curr_cell_coords]
                
                # if the connected vertex is the same as the source vertex, skip to the next cell in queue
                if connected_vertex_id == source_vertex_id:
                    continue
        
            # a connection between two different vertices was found
            if connected_vertex_id >= 0:
                edges.add(tuple(sorted((source_vertex_id, connected_vertex_id))))

        print(f"\r{' ' * 100}\rGraph building complete. Found {len(vertex_id_to_cell_coords)} vertices and {len(edges)} edges.")

        # shift the coordinates of the vertex cells by 0.5 to place them in the center of the cell
        vertex_id_to_cell_coords = {vid: (y + 0.5, x + 0.5) for vid, (y, x) in vertex_id_to_cell_coords.items()}

        self.graph = {
            "vertices": vertex_id_to_cell_coords,
            "edges": edges,
        }
        
        self._connect_special_points_to_graph()  # connect access points and IO points to the graph
    
        # check that graph is connected
        if self._is_graph_connected():
            print("Graph is connected.")
            print("\n[MAP GENERATION SUCCESSFUL]")
        else:
            print("Generated graph is NOT connected!")
            print("\n[MAP GENERATION FAILED]")
            self._visualize_graph_components()
            self.visualize_graph(title="Disconnected Graph")
            # raise ValueError("Generated graph is not connected!")
        
    
    def _connect_special_points_to_graph(self):
        """
        Connects special points to the graph vertex located on the same Voronoi cell.
        """
        print(f"Connecting special points to graph vertices...")
        
        grid = self.voro.astype(np.uint8) + self.special_point_bridges.astype(np.uint8)  # 0 = free cell, 1 = skeleton/bridge, 2 = occupied
        
        connected = 0
        for sp in self.access_points + self.io_points:
            y, x = sp[0], sp[1]
            if grid[int(y), int(x)] == 1:  # if the special point is on a Voronoi/bridge cell
                # find the vertex_id of the graph vertex located on the same Voronoi cell
                for vertex_id, (vy, vx) in self.graph["vertices"].items():
                    if int(vy) == int(y) and int(vx) == int(x):
                        # create vertex for the special point in the graph
                        sp_vertex_id = len(self.graph["vertices"])
                        self.graph["vertices"][sp_vertex_id] = (y, x)
                        # connect the special point vertex to the graph vertex
                        self.graph["edges"].add(tuple((vertex_id, sp_vertex_id)))
                        connected += 1
                        break
            else:
                raise ValueError(f"Special point at ({y}, {x}) is not on a Voronoi cell.")
        print(f"Connected {connected}/{len(self.access_points) + len(self.io_points)} special points to graph vertices.")
    
    def draw_affected_outline(self, ax):
        """Draw an outline exactly along the boundaries of affected cells."""

        affected = np.zeros((self.height, self.width), dtype=bool)
        LINE_WIDTH = 2
        COLOR = (0.1, 0.1, 0.1, 1)

        for row, col in self.affected_distance_cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                affected[row, col] = True

        for row, col in zip(*np.where(affected)):
            # left boundary
            if col == 0 or not affected[row, col - 1]:
                ax.plot([col, col], [row, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # right boundary
            if col == self.width - 1 or not affected[row, col + 1]:
                ax.plot([col + 1, col + 1], [row, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # bottom boundary
            if row == 0 or not affected[row - 1, col]:
                ax.plot([col, col + 1], [row, row], color=COLOR, linewidth=LINE_WIDTH, zorder=50)

            # top boundary
            if row == self.height - 1 or not affected[row + 1, col]:
                ax.plot([col, col + 1], [row + 1, row + 1], color=COLOR, linewidth=LINE_WIDTH, zorder=50)
    
    def draw_obstacles(self):

        fig, ax = plt.subplots()
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # buffer zone
        from shapely.geometry import box
        from shapely.ops import unary_union
        occupied = self.occupancy_grid > 0
        ys, xs = np.where(occupied)
        obstacle_polygons = [box(x, y, x + 1, y + 1) for y, x in zip(ys, xs)]
        obstacles = unary_union(obstacle_polygons)
        buffer_zone = obstacles.buffer(self.agv_radius, resolution=BUFFER_RES, join_style=shapely.BufferJoinStyle.round)
        buffer_zone = buffer_zone.intersection(box(0, 0, self.width, self.height))
        polygons = (list(buffer_zone.geoms) if hasattr(buffer_zone, "geoms") else [buffer_zone])
        for poly in polygons:
            vertices = []
            codes = []
            exterior = np.asarray(poly.exterior.coords)
            vertices.extend(exterior)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY])
            for hole in poly.interiors:
                hole = np.asarray(hole.coords)
                vertices.extend(hole)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
            ax.add_patch(PathPatch( Path(vertices, codes), facecolor=BUFFER_COLOR[:3], alpha=BUFFER_COLOR[3], edgecolor="none", zorder=0))
        
        # occupancy
        vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
        vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR
    
        ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)
    
        return fig, ax
        
    
    def visualize_distance_map(self, title=None):
        """Draws obstacles and D"""
        
        fig, ax = self.draw_obstacles()
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # distance map
        
        # D_normalized = self.D / np.max(self.D) if np.max(self.D) > 0 else self.D
        # D_colormap = plt.cm.plasma(D_normalized)
        # vis_map[self.D < np.inf] = D_colormap[self.D < np.inf]
        
        finite = np.isfinite(self.D)

        D_colormap = np.zeros((*self.D.shape, 4))

        if np.any(finite):
            max_D = np.max(self.D[finite])
            D_normalized = np.zeros_like(self.D, dtype=float)
            D_normalized[finite] = self.D[finite] / max_D if max_D > 0 else 0

            D_colormap[finite] = plt.cm.plasma(D_normalized[finite])

        ax.imshow(D_colormap, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)

        ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)
        
        # affected cells outline
        if self.iteration > 0:
            self.draw_affected_outline(ax)
        
        # distance map legend
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # norm = Normalize(vmin=0, vmax=np.max(self.D) if np.max(self.D) > 0 else 1)
        # norm = Normalize(vmin=0, vmax=146 if np.max(self.D) > 0 else 1)
        # sm = ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.1)
        # cbar = fig.colorbar(sm, cax=cax)
        # cbar.set_label('Distance to Nearest Obstacle', rotation=270, labelpad=15)
        
        ax.set_xticks(np.arange(1, self.width, self.scale_factor))
        ax.set_yticks(np.arange(1, self.height, self.scale_factor))
        
        # remove tick marks and labels
        ax.tick_params(axis='both', which='both', length=0) 
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_axisbelow(False)
        ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect('equal')
        ax.margins(0)
        
        # if title:
        #     ax.set_title(title, fontsize=20, pad=5)
                
        fig.savefig(f"distance_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
        
        # filename = f"gvd_steps/gvd_{title}.pdf"
        # fig.savefig(filename, format="pdf", bbox_inches="tight")
        
        plt.show()
    
    def visualize_voro(self, title=None):
        """Draw obstacles and voro"""

        fig, ax = self.draw_obstacles()
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # voro map
        vis_map[self.voro] = VORO_SKELETON_COLOR
        
        # affected cells outline
        if self.iteration > 0:
            # self.draw_affected_outline(ax)
            # affected distance cells
            if self.iteration > 0:
                changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
                if self.affected_distance_cells:
                    cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
                    if cells.size > 0:
                        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                        changed_distance_mask[cells[:, 0], cells[:, 1]] = True
                vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR
            # affected Voronoi cells
            if self.iteration > 0:
                affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
                if self.affected_voro_cells:
                    cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
                    if cells.size > 0:
                        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                        affected_voro_mask[cells[:, 0], cells[:, 1]] = True
                vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR
        
        ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
        ax.set_xticks(np.arange(1, self.width, self.scale_factor))
        ax.set_yticks(np.arange(1, self.height, self.scale_factor))
        
        # remove tick marks and labels
        ax.tick_params(axis='both', which='both', length=0) 
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_axisbelow(False)
        ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect('equal')
        ax.margins(0)
        
        # if title:
        #     ax.set_title(title, fontsize=20, pad=5)
                
        fig.savefig(f"voro_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
        
        plt.show()
    
    def visualize_distance_map_and_voro(self, title=None):
        """Draw obstacles, D, and voro as two subplots in one figure."""
        
        fig = plt.figure(figsize=(7, 10))

        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.1)

        ax1 = fig.add_subplot(gs[0, 0])  # distance map
        cax = fig.add_subplot(gs[0, 1])  # colorbar
        ax2 = fig.add_subplot(gs[1, 0])  # voro map

        # obstacles

        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
        vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR
    
        ax1.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)
        ax2.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=2)
        
        # distance map

        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # distance map
        D_normalized = self.D / np.max(self.D) if np.max(self.D) > 0 else self.D
        D_colormap = plt.cm.plasma(D_normalized)
        vis_map[self.D < np.inf] = D_colormap[self.D < np.inf]

        ax1.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", alpha=0.8, zorder=1)
        
        # affected cells outline
        if self.iteration > 0:
            self.draw_affected_outline(ax1)
        
        # distance map legend
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=0, vmax=np.max(self.D) if np.max(self.D) > 0 else 1)

        sm = ScalarMappable(cmap=plt.cm.plasma, norm=norm)

        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Distance to Nearest Obstacle")
        
        ax1.set_xticks(np.arange(1, self.width, self.scale_factor))
        ax1.set_yticks(np.arange(1, self.height, self.scale_factor))
        
        # remove tick marks and labels
        ax1.tick_params(axis='both', which='both', length=0) 
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
        ax1.set_axisbelow(False)
        ax1.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(self.height, 0)
        ax1.set_aspect('equal')
        ax1.margins(0)
        
        # voro
        
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # voro map
        vis_map[self.voro] = VORO_SKELETON_COLOR
        
        # affected cells outline
        if self.iteration > 0:
            # self.draw_affected_outline(ax)
            # affected distance cells
            if self.iteration > 0:
                changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
                if self.affected_distance_cells:
                    cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
                    if cells.size > 0:
                        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                        changed_distance_mask[cells[:, 0], cells[:, 1]] = True
                vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR
            # affected Voronoi cells
            if self.iteration > 0:
                affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
                if self.affected_voro_cells:
                    cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
                    if cells.size > 0:
                        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                        affected_voro_mask[cells[:, 0], cells[:, 1]] = True
                vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR
        
        ax2.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
        ax2.set_xticks(np.arange(1, self.width, self.scale_factor))
        ax2.set_yticks(np.arange(1, self.height, self.scale_factor))
        
        # remove tick marks and labels
        ax2.tick_params(axis='both', which='both', length=0) 
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        ax2.set_axisbelow(False)
        ax2.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
        ax2.set_xlim(0, self.width)
        ax2.set_ylim(self.height, 0)
        ax2.set_aspect('equal')
        ax2.margins(0)
        
        fig.canvas.draw()

        ax1.annotate("", xy=(0.5, 0.48), xytext=(0.5, 0.52), xycoords=fig.transFigure, arrowprops=dict(arrowstyle="->", linewidth=2))
        
        plt.savefig(f"distance_and_voro_map_{self.iteration}_{title}.pdf", format="pdf", bbox_inches="tight")
        
        plt.show()
    
    def draw_grid(self, title=None):
        from matplotlib.patches import Patch
        
        fig, ax = self.draw_obstacles()
                
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # buffer zone
        from shapely.geometry import box
        from shapely.ops import unary_union
        occupied = self.occupancy_grid > 0
        ys, xs = np.where(occupied)
        obstacle_polygons = [box(x, y, x + 1, y + 1) for y, x in zip(ys, xs)]
        obstacles = unary_union(obstacle_polygons)
        buffer_zone = obstacles.buffer(self.agv_radius, resolution=BUFFER_RES, join_style=shapely.BufferJoinStyle.round)
        buffer_zone = buffer_zone.intersection(box(0, 0, self.width, self.height))
        polygons = (list(buffer_zone.geoms) if hasattr(buffer_zone, "geoms") else [buffer_zone])
        for poly in polygons:
            vertices = []
            codes = []
            exterior = np.asarray(poly.exterior.coords)
            vertices.extend(exterior)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY])
            for hole in poly.interiors:
                hole = np.asarray(hole.coords)
                vertices.extend(hole)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
            ax.add_patch(PathPatch( Path(vertices, codes), facecolor=BUFFER_COLOR[:3], alpha=BUFFER_COLOR[3], edgecolor="none", zorder=0))
        
        # occupancy
        vis_map[self.occupancy_grid == 1] = SIMPLE_OBSTACLE_COLOR
        vis_map[self.occupancy_grid == 2] = UNIT_LOAD_COLOR

        # affected distance cells
        if self.iteration > 0:
            changed_distance_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_distance_cells:
                cells = np.asarray(list(self.affected_distance_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    changed_distance_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[changed_distance_mask] = AFFECTED_DISTANCES_COLOR

        # Voronoi skeleton
        vis_map[self.voro] = VORO_SKELETON_COLOR

        # affected Voronoi cells
        if self.iteration > 0:
            affected_voro_mask = np.zeros((self.height, self.width), dtype=bool)
            if self.affected_voro_cells:
                cells = np.asarray(list(self.affected_voro_cells), dtype=np.intp)
                if cells.size > 0:
                    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)]
                    affected_voro_mask[cells[:, 0], cells[:, 1]] = True
            vis_map[affected_voro_mask] = AFFECTED_VORO_CELLS_COLOR

        # bridges are the final cell overlay.
        vis_map[self.special_point_bridges] = BRIDGE_COLOR
        ax.imshow(vis_map, origin="lower", extent=(0, self.width, 0, self.height), interpolation="none", zorder=1)
        
        # grid and axes
        ax.set_xticks(np.arange(1, self.width, self.scale_factor))
        ax.set_yticks(np.arange(1, self.height, self.scale_factor))
        ax.set_axisbelow(False)
        ax.grid(True, color='black', linestyle='-', linewidth=0.5, zorder=40)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect('equal')
        ax.margins(0)

        # special points
        if self.access_points:
            access = np.asarray([(p[1], p[0]) for p in self.access_points], dtype=float)
            ax.scatter(access[:, 0], access[:, 1], s=30, color="red", zorder=20, label="Access points")
        if self.io_points:
            io = np.asarray([(p[1], p[0]) for p in self.io_points], dtype=float)
            ax.scatter(io[:, 0], io[:, 1], s=50, color="green", zorder=20, label="IO points")

        # legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=AFFECTED_DISTANCES_COLOR, edgecolor="none"))
        handles.append(Patch(facecolor=AFFECTED_VORO_CELLS_COLOR, edgecolor="none"))
        handles.append(Patch(facecolor=BRIDGE_COLOR, edgecolor="none"))
        handles.append(Patch(facecolor=VORO_SKELETON_COLOR, edgecolor="none"))
        labels.append("Changed distance cells")
        labels.append("Affected Voronoi cells")
        labels.append("Bridges")
        labels.append("Voronoi skeleton")
        ax.legend(handles, labels, loc="upper right")
        
        ax.set_title(title)

        return fig, ax

    def visualize_gvd(self, title=None):
        fig, ax = self.draw_grid(title=title or "Incremental GVD")
        
        plt.show()

    def visualize_graph(self, title=None):
        
        fig, ax = self.draw_grid(title=title or "Graph from Voronoi Diagram")

        edge_segments = []
        for vertex1, vertex2 in self.graph["edges"]:
            y1, x1 = self.graph["vertices"][vertex1]
            y2, x2 = self.graph["vertices"][vertex2]
            edge_segments.append([(x1, y1), (x2, y2)])

        ax.add_collection(LineCollection(edge_segments, colors=EDGE_COLOR, linewidths=3, zorder=10, label="Graph edges"))

        vertices = np.asarray([(x, y) for y, x in self.graph["vertices"].values()], dtype=float)
        ax.scatter(vertices[:, 0], vertices[:, 1], color=VERTEX_COLOR, linewidth=1, s=50, zorder=11, label="Graph vertices")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

        # fig.tight_layout()
        plt.show()


# TODO: edge splitting (obstacle avoidance) -- necessary?
#         - probably YES (edges can cross buffer zone, at least for CORNER_SEGMENTS=0)


if __name__ == "__main__":
    
    
    from load_scenario import load_scenario_as_numpy
    
    rng = np.random.default_rng(42)
    
    # load initial scenario and add outer walls
    scenario_0 = load_scenario_as_numpy(0)
    scenario_1 = load_scenario_as_numpy(1)
    scenario_2 = load_scenario_as_numpy(2)
    scenario_3 = load_scenario_as_numpy(3)
    
    if len(sys.argv) > 1:
        scale_factor = int(sys.argv[1])
    else:
        # one Voronoi cell must fit between two obstacles, inflated by the AGV radius in a one unit-load wide passage
        scale_factor = math.ceil(1 / (1 - 2 * (scenario_0["agv_radius"])))
    
    # ---- iteration 0 ----
    
    # initialize spatial representation
    print("\nInitialize and compute GVD...")
    incr_gvd = IncrementalGVD(
        rng=rng,
        occupancy_grid=scenario_0["occupancy"],
        access_points=scenario_0["access_points"],
        io_points=scenario_0["io_points"],
        agv_radius=scenario_0["agv_radius"],
        scale_factor=scale_factor,
    )
    
    # compute initial distance map and Voronoi diagram
    incr_gvd.computeDistanceMap()
    # incr_gvd.visualize_distance_map(title="Initial Distance Map")
    # incr_gvd.visualize_voro(title="Voronoi Diagram After Changes")
    incr_gvd.visualize_distance_map_and_voro(title="dist_and_voro")
    # incr_gvd.visualize_gvd(title="Incremental GVD Results")
    
    # build Voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph(title="Incremental GVD Graph")
    
    # ---- iteration 1 ----
    
    # apply updates to the occupancy grid
    print("\nApplying updates to the occupancy grid...")
    incr_gvd.apply_obstacle_updates(scenario_1["updates"], scenario_1["access_points"], scenario_1["io_points"])
    
    # recompute distance map and Voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    incr_gvd.visualize_distance_map_and_voro(title="dist_and_voro")
    
    
    # build Voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph(title="Incremental GVD Graph After Changes")
    
    # ---- iteration 2 ----
    
    # apply updates to the occupancy grid
    print("\nApplying updates to the occupancy grid...")
    incr_gvd.apply_obstacle_updates(scenario_2["updates"], scenario_2["access_points"], scenario_2["io_points"])
    
    # recompute distance map and Voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    incr_gvd.visualize_distance_map_and_voro(title="dist_and_voro")

    # build Voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph(title="Incremental GVD Graph After Changes")

    # # ---- iteration 3 ----
    
    # apply updates to the occupancy grid
    print("\nApplying updates to the occupancy grid...")
    incr_gvd.apply_obstacle_updates(scenario_3["updates"], scenario_3["access_points"], scenario_3["io_points"])
    
    # recompute distance map and Voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    incr_gvd.visualize_distance_map(title="Distance Map After Changes")
    incr_gvd.visualize_gvd(title="Incremental GVD Results After Changes")

    # build Voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph(title="Incremental GVD Graph After Changes")