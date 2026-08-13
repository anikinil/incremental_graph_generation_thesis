"""
Lau, Sprunk, Burgard
"Efficient grid-based spatial representations for robot navigation in dynamic environments"
(Robotics and Autonomous Systems, 2013)
"""

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from collections import defaultdict, deque

from matplotlib.path import Path
from matplotlib.patches import PathPatch
        
from load_scenario import load_obstacle_collision_zone
        

import shapely


BUFFER_COLOR = [0.8, 0.8, 0.8, 1] # light grey
FREE_SPACE_COLOR = [0, 0, 0, 0]  # transparent
SIMPLE_OBSTACLE_COLOR = [0.3, 0.3, 0.3, 1] # dark grey
UNIT_LOAD_COLOR = [0.5, 0.5, 0.5, 1] # grey


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
        
        
        self.affected_cells = set()  # track affected cells after dynamic updates for visualization
        
        # initialize with static brushfire algorithm
        self.computeDistanceMap()
        
        self.graph = None # { "vertices": [vid_1: (y, x), vid_2: (y, x), ...], "edges": [(vid_1, vid_2), ...] }
        
        # save voronoi cells added to connect special points to Voronoi skeleton, to recalculate if affected by dynamic updates
        # {sp_1: [(y, x), (y, x), ...], sp_2: [...], ...}
        self.special_point_bridges = {}
    
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
        
        # initialize occupied cells
        print()
        for y in range(self.height):
            for x in range(self.width):
                s = (y, x)
                print(f"\r{' ' * 100}\rInitializing cell ({y}, {x}) with occupancy {self.occupancy_grid[y, x]}", end="", flush=True)
                if self.occupancy_grid[y, x] > 0:  # if cell occupied
                    self.D[y, x] = 0 # distance to itself is zero
                    self.obst[y, x] = s # is its own closest obstacle
                    self.open.insert(s, 0) # enqueue for wavefront propagation: distance=0, cell=s
                # else leave distance as inf and reference as (-1, -1)
                # TODO why do they set it to inf here?
                        
        # propagate wavefronts
        print()
        while not self.open.is_empty():
            print(f"\r{' ' * 100}\rProcessing queue with current min distance {self.open.current_min_distance}", end="", flush=True)
            # if self.open.is_empty(): # if no more cells to process: done
                # break
            s, _ = self.open.pop() # pop cell with smallest distance
            self.lower(s) # propagate to neighbors and update their distances if closer to obstacle
        print(f"\r{' ' * 100}\rDistance map complete.")
    
    def lower(self, s):
        
        # get the obstacle reference of the current cell
        s_obst = self.obst[s[0], s[1]]
        
        # for each neighbor of s
        for n in self.Adj(s):
            # calculate distance from neighbor to obstacle
            if not self.toRaise[n[0], n[1]]:  # if neighbor not marked to raise
                d = self.distance(s_obst, n) # calculate distance between current cell and neighbor
                # update neighbor's distance if new distance is smaller than its current distance
                if d < self.D[n[0], n[1]]:
                    # update neighbor's distance
                    self.D[n[0], n[1]] = d
                    # update neighbor's obstacle reference to the same as current cell
                    self.obst[n[0], n[1]] = s_obst
                    # enqueue neighbor to continue wavefront propagation
                    self.open.insert(n, d)
                    self.affected_cells.add(n)  # track affected cells for visualization
                else:
                    threshold = max(self.agv_radius, 1)
                    self.checkVoro(s, n, threshold=threshold)  # check Voronoi condition between current cell and neighbors
    
    def setObstacle(self, s, value):
        
        # add new cell to occupancy grid
        self.occupancy_grid[s[0], s[1]] = value
        
        # initialize distance and obstacle reference for new cell
        self.obst[s[0], s[1]] = s
        self.D[s[0], s[1]] = 0
        
        # enqueue new cell for wavefront propagation
        self.open.insert(s, 0)
        self.affected_cells.add(s)
    
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
        self.affected_cells.add(s)  # track affected cells for visualization
        
        return value  # return the type of obstacle that was removed (for move operations)
    
    def updateDistanceMap(self):
                
        # while there are cells to process
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
                    self.lower(s) 

    def raise_(self, s):
        
        # for every neighbor of the current cell
        for n in self.Adj(s):
            obstacle_ref = self.obst[n[0], n[1]]
            
            if obstacle_ref != (-1, -1) and not self.toRaise[n[0], n[1]]:  # if neighbor not already cleared or marked for raise
                
                # enqueue with current distance
                self.open.insert(n, self.D[n[0], n[1]])
                self.affected_cells.add(n)
                
                # check if obstacle reference is still valid
                if self.occupancy_grid[obstacle_ref[0], obstacle_ref[1]] == 0: # if neighbor's closest obstacle is no longer occupied
                    # clear neighbor and mark for raise
                    self.D[n[0], n[1]] = np.inf
                    self.obst[n[0], n[1]] = (-1, -1)
                    self.toRaise[n[0], n[1]] = True
        
        # after processing neighbors, mark current cell as no longer needing raise
        self.toRaise[s[0], s[1]] = False
    
    def checkVoro(self, s, n, threshold = 1.0):
        
        obst_s = self.obst[s[0], s[1]]
        obst_n = self.obst[n[0], n[1]]
        
        if (self.D[s[0], s[1]] > threshold or self.D[n[0], n[1]] > threshold) \
            and obst_n != (-1, -1) \
            and obst_s != (-1, -1) \
            and obst_s != obst_n \
            and obst_s not in self.Adj(obst_n):
        
            # calculate distances
            dist_s_to_obst_n = self.distance(s, obst_n)
            dist_n_to_obst_s = self.distance(n, obst_s)
            
            # add cell with smaller violation
            if dist_s_to_obst_n <= dist_n_to_obst_s:
                self.voro[s] = True
            
            if dist_n_to_obst_s <= dist_s_to_obst_n:
                self.voro[n] = True

    def apply_updates(self, updates, new_access_points, new_io_points):
        
        # clear bridges from special points that are affected by the updates
        # (need to clear the whole bridge, not just the affected cells, so they can be reconnected properly)
        for sp in list(self.special_point_bridges.keys()):
            if self._special_point_affected_by_update(sp):
                bridge_cells = self.special_point_bridges[sp]
                for cell in bridge_cells:
                    self.voro[cell] = False  # remove bridge cells from Voronoi diagram
                del self.special_point_bridges[sp]  # remove the bridge record for this special point
        
        self.affected_cells.clear()  # clear affected cells
        
        # scale update coords to the grid resolution
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
        
    def _special_point_affected_by_update(self, special_point):
        """
        Checks if a special point is affected by the dynamic updates.
        A special point is affected, if any cell of its Voronoi bridge is in the affected cells set.
        """
        if special_point not in self.special_point_bridges:
            return True  # if no bridge exists, the point has not been connected yet and must be processed anyway
        bridge_cells = self.special_point_bridges[special_point]
        for cell in bridge_cells:
            if cell in self.affected_cells:
                return True  # at least one cell of the bridge is affected
        return False  # none of the bridge cells are affected


    def connect_special_points_to_skeleton(self):
        """
        Connects specifal points to the Voronoi cells by building a ray of cells 
        from the special point in the direction of the dir label to the closest GVD cell.
        
        NOTE theory, why that seems to alway work: Access point rays always meet a Voronoi cell before an obstacle.
        There is always a Voronoi cell between two obstacles, so there is always a Voronoi cell between the access point's obstacle
        and any other, the ray is headed towards.
        """
        
        dirs = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1)} # dir: (dy, dx)
        
        connected = 0
        for ap in self.access_points:
            
            x, y, dir = ap
            # determine the direction vector based on the dir label
            dir_vec = dirs[dir]
            bridge_cells = []  # to store the cells added to the Voronoi diagram for this special point
            while True:
                # get current cell
                on_voro = self.voro[int(y), int(x)]
                # if on a Voronoi cell, stop the ray
                if on_voro:
                    connected += 1
                    break
                # else: add the current cell to the Voronoi diagram
                self.voro[int(y), int(x)] = True
                
                # add the current cell to the bridge cells list
                bridge_cells.append((int(y), int(x)))
                
                # move in the direction of the dir label
                y += dir_vec[0]
                x += dir_vec[1]
                
            # save the bridge cells for this special point
            self.special_point_bridges[ap] = bridge_cells
        
        # try to connect similarly to access points, but try each direction and pick the best one:
        # no obstacles encountered, shortest distance
        connected = 0
        for i, iop in enumerate(self.io_points):
            x, y = iop
            bridge_cells = [] # cells added to the Voronoi diagram for this special point
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
                self.special_point_bridges[iop] = best_bridge_cells
                # add the best bridge cells to the Voronoi diagram
                for cell in best_bridge_cells:
                    self.voro[cell] = True
            else: 
                print(f"Warning: Could not connect IO point {iop} to Voronoi skeleton in any direction.")
            
        print(f"Connected {connected}/{len(self.io_points)} IO points to Voronoi skeleton.")
            

    def _special_point_on_cell(self, cell):
        """
        Checks if a special point is located on the given cell (r, c).
        """
        r, c = cell
        for sp in self.access_points + self.io_points:
            if (int(sp[0]), int(sp[1])) == (c, r):
                return True
        return False

    def build_graph(self):
        """
        "Sparse 3D Topological Graphs for Micro-Aerial Vehicle Planning",
        Helen Oleynikova, Zachary Taylor, Roland Siegwart, Juan Nieto (2018)
        
        https://github.com/ethz-asl/mav_voxblox_planning/blob/master/voxblox_skeleton/src/skeleton_generator.cpp
        
        """
        
        # 0 = free cell, 1 = skeleton cell (on free cell), 2 = occupied cell
        voronoi_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        voronoi_grid[self.voro] = 1
        obstacle_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        obstacle_grid[self.occupancy_grid > 0] = 2
        grid = voronoi_grid + obstacle_grid
        
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
    
    def _connect_special_points_to_graph(self):
        """
        Connects special points to the graph vertex located on the same Voronoi cell.
        """
        for sp in self.access_points + self.io_points:
            x = sp[0]
            y = sp[1]
            if self.voro[int(y), int(x)]:
                # find the vertex_id of the graph vertex located on the same Voronoi cell
                for vertex_id, (vy, vx) in self.graph["vertices"].items():
                    if int(vy) == int(y) and int(vx) == int(x):
                        # create vertex for the special point in the graph
                        sp_vertex_id = len(self.graph["vertices"])
                        self.graph["vertices"][sp_vertex_id] = (y, x)
                        # connect the special point vertex to the graph vertex
                        self.graph["edges"].add(tuple((vertex_id, sp_vertex_id)))
                        break
            else:
                raise ValueError(f"Special point at ({y}, {x}) is not on a Voronoi cell.")
        
        
    def draw_grid(self, iteration=0):
        fig, ax = plt.subplots(figsize=(25, 20))

        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)

        if iteration > 0:
            # cells affected by updates (red)
            affected_mask = np.zeros((self.height, self.width), dtype=bool)
            for cell in self.affected_cells:
                affected_mask[cell[0], cell[1]] = True
            vis_map[affected_mask] = [1, 0, 0, 0.2]

        # buffer zone (light grey)
        buffer_zone = load_obstacle_collision_zone(iteration)
        scale_factor = 3
        buffer_zone = shapely.affinity.scale(buffer_zone, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        buffer_zone = shapely.affinity.translate(buffer_zone, xoff=1, yoff=1)
        vertices = []
        codes = []
        for poly in buffer_zone.geoms:
            exterior = np.asarray(poly.exterior.coords)
            vertices.extend(exterior)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY])
            for interior in poly.interiors:
                hole = np.asarray(interior.coords)
                vertices.extend(hole)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
        path = Path(vertices, codes)
        ax.add_patch(PathPatch(path, facecolor=BUFFER_COLOR[:3], alpha=BUFFER_COLOR[3], edgecolor="none", zorder=0))

        # simple obstacles (dark grey)
        obstacle_mask = self.occupancy_grid == 1
        vis_map[obstacle_mask] = SIMPLE_OBSTACLE_COLOR
        # unit loads (grey)
        unit_load_mask = self.occupancy_grid == 2
        vis_map[unit_load_mask] = UNIT_LOAD_COLOR

    
        # voronoi cells (blue)
        voronoi_mask = (self.voro > 0)
        vis_map[voronoi_mask] = [0.3, 0.3, 0.5, 0.7]
        
        # convert squared distances to actual distances
        distance_overlay = np.sqrt(np.copy(self.D))
        
        # overlay the distance map on top of the occupancy grid visualization
        # distance_im = ax.imshow(distance_overlay, origin='lower', extent=(0, self.width, 0, self.height), cmap='viridis', alpha=0.5)
        # fig.colorbar(distance_im, ax=ax, fraction=0.046, pad=0.04, label='Distance')

        ax.imshow(vis_map, origin='lower', extent=(0, self.width, 0, self.height))

        scale_factor = 3
        ax.set_xticks(np.arange(1, self.width, scale_factor))
        ax.set_yticks(np.arange(1, self.height, scale_factor))
        ax.grid(True, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.margins(0)
        ax.invert_yaxis()
        
        # access points
        access_x = [p[0] for p in self.access_points]
        access_y = [p[1] for p in self.access_points]
        ax.scatter(access_x, access_y, s=30, color="red", zorder=3, label="Access points")

        # IO points
        io_x = [p[0] for p in self.io_points]
        io_y = [p[1] for p in self.io_points]
        ax.scatter(io_x, io_y, s=50, color="green", zorder=3, label="IO points")
        ax.legend(loc='upper right')

        return fig, ax

    def visualize_gvd(self, title=None, iteration=0):
        
        print(f"Visualizing GVD: {title if title else 'unnamed'}...")

        fig, ax = self.draw_grid(iteration=iteration)

        plt.title(title if title else "Incremental GVD")
        plt.show()

    def visualize_graph(self, title=None, iteration=0):
        print(f"Visualizing graph: {title if title else 'unnamed'}...")

        fig, ax = self.draw_grid(iteration=iteration)

        # edges
        edge_segments = []
        for vertex1, vertex2 in self.graph["edges"]:
            y1, x1 = self.graph["vertices"][vertex1]
            y2, x2 = self.graph["vertices"][vertex2]
            edge_segments.append([(x1, y1), (x2, y2)])
        edge_collection = LineCollection(edge_segments, colors="blue", linewidths=2, zorder=2)
        ax.add_collection(edge_collection)

        # vertices
        vertices = np.array([(x, y) for y, x in self.graph["vertices"].values()])
        ax.scatter(vertices[:, 0], vertices[:, 1], color="orange", s=30, zorder=3)

        plt.title("Graph from Voronoi Diagram")
        plt.show()


# TODO: thinning ([21] in Oleynikova paper)
# TODO: edge splitting (obstacle avoidance) -- necessary?
#         - probably YES (edges can cross buffer zone, at least for CORNER_SEGMENTS=0)


if __name__ == "__main__":
    from load_scenario import load_scenario_as_numpy
    
    rng = np.random.default_rng(42)
    
    # load initial scenario and add outer walls
    scenario_0 = load_scenario_as_numpy(0)
    scenario_1 = load_scenario_as_numpy(1)
    scenario_2 = load_scenario_as_numpy(2)
    
    # ---- iteration 0 ----
    
    # initialize spatial representation and compute distance map
    print("\nInitialize and compute GVD...")
    incr_gvd = IncrementalGVD(
        rng=rng,
        occupancy_grid=scenario_0["occupancy"],
        access_points=scenario_0["access_points"],
        io_points=scenario_0["io_points"],
        agv_radius=scenario_0["agv_radius"],
        scale_factor=3,
    )
    incr_gvd.visualize_gvd("Incremental GVD Results", iteration=0)
    
    # connect special points to the voronoi diagram
    incr_gvd.connect_special_points_to_skeleton()
    incr_gvd.visualize_gvd("Incremental GVD Results After Connecting Special Points", iteration=0)
    
    # build voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph("Incremental GVD Graph", iteration=0)
    
    # ---- iteration 1 ----
    
    # apply updates to the occupancy grid
    print("\nApplying updates to the occupancy grid...")
    incr_gvd.apply_updates(scenario_1["updates"], scenario_1["access_points"], scenario_1["io_points"])
    
    # recompute distance map and voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    incr_gvd.visualize_gvd("Incremental GVD Results After Changes", iteration=1)
    
    # connect special points to the voronoi diagram
    incr_gvd.connect_special_points_to_skeleton()
    incr_gvd.visualize_gvd("Incremental GVD Results After Connecting Special Points", iteration=1)
        
    # build voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph("Incremental GVD Graph After Changes", iteration=1)
    
    # ---- iteration 2 ----
    
    # apply updates to the occupancy grid
    print("\nApplying updates to the occupancy grid...")
    incr_gvd.apply_updates(scenario_2["updates"], scenario_2["access_points"], scenario_2["io_points"])
    
    # recompute distance map and voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    incr_gvd.visualize_gvd("Incremental GVD Results After Changes", iteration=2)
    
    # connect special points to the voronoi diagram
    incr_gvd.connect_special_points_to_skeleton()
    incr_gvd.visualize_gvd("Incremental GVD Results After Connecting Special Points", iteration=2)

    # build voronoi graph
    incr_gvd.build_graph()
    incr_gvd.visualize_graph("Incremental GVD Graph After Changes", iteration=2)