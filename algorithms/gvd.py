"""
Lau, Sprunk, Burgard
"Efficient grid-based spatial representations for robot navigation in dynamic environments"
(Robotics and Autonomous Systems, 2013)
"""

import numpy as np
from collections import defaultdict, deque

from scenario.geometry import centered_io_points, offset_access_points


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


class GVD:
    
    def __init__(self, rng, occupancy_grid, scale_factor, access_points, io_points, agv_radius, recorder):
        self.rng = rng

        self.occupancy_grid = occupancy_grid.copy()
        self.height, self.width = occupancy_grid.shape
        
        self.scale_factor = scale_factor
        
        self.access_points = access_points
        self.io_points = io_points
        
        self.agv_radius = agv_radius

        self.recorder = recorder
        
        # increase resolution and add outer walls
        self.increase_resolution()
        self.add_outer_walls()
        
        # distance map: stores distances to closest obstacle for each cell
        self.D = np.full((self.height, self.width), np.inf, dtype=np.float32)
        
        # obstacle reference map: stores (y, x) of closest occupied cell for each cell
        self.obst = np.empty((self.height, self.width), dtype=object)
        self.obst.fill((-1, -1))
        
        # voronoi map: True if cell is on Voronoi diagram
        self.voro = np.zeros((self.height, self.width), dtype=bool)
        
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

    def _plot_context(self):
        """Return the complete grid state in the GVD coordinate system.

        The GVD changes resolution and adds its own outer wall, so plotting
        must use this state rather than reconstructing it from the source
        scenario.  Keeping the context here also makes every saved snapshot
        self-contained and reproducible.
        """
        return {
            "occupancy": self.occupancy_grid,
            "access_points": self.access_points,
            "io_points": self.io_points,
            "agv_radius": self.agv_radius,
            "scale_factor": self.scale_factor,
            "iteration": self.iteration,
            "affected_distance_cells": self.affected_distance_cells,
            "affected_voro_cells": self.affected_voro_cells,
        }

    def _record(self, data_type, **data):
        """Save an algorithm snapshot together with its authoritative context."""
        if self.recorder is None:
            return None
        return self.recorder.save(
            data_type=data_type,
            data={**self._plot_context(), **data},
            plot=True,
        )
    
    
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
        
        # scale and translate special points to match the new resolution and offset access points to account for obstacle buffer
        self.access_points = [
            (y * self.scale_factor, x * self.scale_factor, direction)
            for y, x, direction in offset_access_points(self.access_points, self.agv_radius)
        ]
        self.io_points = [(y * self.scale_factor, x * self.scale_factor) for y, x in centered_io_points(self.io_points)]
        
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
        for y in range(self.height):
            for x in range(self.width):
                s = (y, x)
                if self.occupancy_grid[y, x] > 0:  # if an object is in the cell
                    self.D[y, x] = 0 # distance to itself is zero
                    self.obst[y, x] = s # is its own closest obstacle
                    self.open.insert(s, 0) # enqueue for wavefront propagation: distance=0, cell=s
                        
        # propagate wavefronts
        while not self.open.is_empty():
            s, _ = self.open.pop() # pop cell with smallest distance
            self.lower(s) # propagate to neighbors and update their distances if closer to obstacle
        print("Distance map complete.")
        
        # self.prune_voronoi(connectivity=8) # prune
        
        self._record("distance_map", D=self.D, obst=self.obst)
        self._record("voronoi_skeleton", voro=self.voro)
        
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
        
        self._record("distance_map", D=self.D, obst=self.obst)
        self._record("voronoi_skeleton", voro=self.voro)

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
                
        # the same transformations of special points like during initialization
        source_agv_radius = self.agv_radius / self.scale_factor
        self.access_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1, direction)
            for y, x, direction in offset_access_points(new_access_points, source_agv_radius)
        ]
        self.io_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1) for y, x in centered_io_points(new_io_points)]

        self.special_point_bridges.fill(False)  # clear special bridges, as they need to be recalculated now
        
        self.iteration += 1
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)


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
            while 0 <= int(y) < self.height and 0 <= int(x) < self.width:
                # get current cell
                on_voro = self.voro[int(y), int(x)]

                # if on a Voronoi cell, stop the ray
                if on_voro:
                    connected += 1
                    break
                
                # else: add the cell to the bridge
                self.special_point_bridges[int(y), int(x)] = True
                
                # move in the direction of the dir label
                y += dir_vec[0]
                x += dir_vec[1]
            else:
                print(f"Warning: Access point {ap} does not reach the Voronoi skeleton.")

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
                while 0 <= int(curr_y) < self.height and 0 <= int(curr_x) < self.width:
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

        self._record("voronoi_skeleton_with_bridges", voro=self.voro,
                     bridges=self.special_point_bridges)


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
            # The graph is still recorded below for inspection.
        
        self._record("graph", graph=self.graph, voro=self.voro,
                     bridges=self.special_point_bridges)
    
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
                print(f"Warning: Special point at ({y}, {x}) is not connected to the graph.")
        print(f"Connected {connected}/{len(self.access_points) + len(self.io_points)} special points to graph vertices.")
    
    


# TODO: edge splitting (obstacle avoidance) -- necessary?
#         - probably YES (edges can cross buffer zone, at least for CORNER_SEGMENTS=0)
