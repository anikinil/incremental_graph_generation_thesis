"""
Lau, Sprunk, Burgard
"Efficient grid-based spatial representations for robot navigation in dynamic environments"
(Robotics and Autonomous Systems, 2013)
"""

import numpy as np
from collections import defaultdict, deque

from algorithms.geometry import offset_access_points
from logs.progress import ProgressReportingAlgorithm


class BucketPriorityQueue:
    """
    "Fast Euclidean Distance Transformation by Propagation Using Multiple Neighborhoods",
    O. Cuisenaire and B. Macq (1999)
    """
    
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


class GVD(ProgressReportingAlgorithm):

    progress_name = "GVD"
    
    def __init__(self, rng, occupancy_grid, scale_factor, access_points, interaction_points,
                 robot_radius, recorder, print_progress=True):
        self.rng = rng

        self.occupancy_grid = occupancy_grid.copy()
        self.height, self.width = occupancy_grid.shape
        
        
        self.access_points = access_points
        self.interaction_points = interaction_points
        
        self.robot_radius = robot_radius
        
        self.scale_factor = 1

        self.recorder = recorder
        self.print_progress = bool(print_progress)
        
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
        
        self.graph = None # { "nodes": [vid_1: (y, x), vid_2: (y, x), ...], "edges": [(vid_1, vid_2), ...] }
        
        # retained for compatibility; special points connect directly to Voronoi cells
        self.special_point_connections = np.zeros((self.height, self.width), dtype=bool)

        # maps special point indices to their Voronoi connection cells
        self.access_point_connection_cells = []
        self.interaction_point_connection_cells = []
    
        self.iteration = 0 # increased after every layout update
        self._generation_seconds_total = 0

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
            "interaction_points": self.interaction_points,
            "robot_radius": self.robot_radius,
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
            for y, x, direction in offset_access_points(self.access_points, self.robot_radius)
        ]
        self.interaction_points = [(y * self.scale_factor, x * self.scale_factor) for y, x in self.interaction_points]
        
        # scale robot_radius
        self.robot_radius *= self.scale_factor
        
    
    def add_outer_walls(self):
        
        # add outer walls to the occupancy grid
        self.height += 2
        self.width += 2
        new_occupancy = np.ones((self.height, self.width), dtype=self.occupancy_grid.dtype)
        new_occupancy[1:-1, 1:-1] = self.occupancy_grid
        self.occupancy_grid = new_occupancy
        
        # adjust access points and interaction points
        self.access_points = [(y + 1, x + 1, dir) for (y, x, dir) in self.access_points]
        self.interaction_points = [(y + 1, x + 1) for (y, x) in self.interaction_points]
        
    
    def _addImaginaryObstacles():
        """
        TODO maybe experiment with that:
        
        "Redundant Voronoi roadmap graph using imaginary obstacles for multi-robot path planning", Aryadi et. al (2023)
        """
        pass
    
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
        progress = self._create_progress("Generating distance map")
        if progress is not None:
            progress.start(f"grid: {self.height}x{self.width}")

        start_time = self._start_generation_time()
        
        # initialize and enqueue occupied cells
        for y in range(self.height):
            for x in range(self.width):
                s = (y, x)
                if self.occupancy_grid[y, x] > 0:  # if an object is in the cell
                    self.D[y, x] = 0 # distance to itself is zero
                    self.obst[y, x] = s # is its own closest obstacle
                    self.open.insert(s, 0) # enqueue for wavefront propagation: distance=0, cell=s
                        
        # propagate wavefronts
        if progress is not None:
            processed_cells = 0
        while not self.open.is_empty():
            s, _ = self.open.pop() # pop cell with smallest distance
            self.lower(s) # propagate to neighbors and update their distances if closer to obstacle
            if progress is not None:
                processed_cells += 1
                if processed_cells % 1000 == 0:
                    progress.update(processed_cells, details="wavefront cells processed")

        # self.prune_voronoi(connectivity=8) # prune
        
        # only capture the distance map generation time, not saving time
        self._generation_seconds_total += self._generation_seconds(start_time)
        
        self._record("distance_map", D=self.D, obst=self.obst)
        self._record("voronoi_skeleton", voro=self.voro)
       
        if progress is not None:
            progress.finish(True, f"processed {processed_cells} wavefront cells")
       
        self.build_graph()
        
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
                    # threshold = max(self.robot_radius, 1)
                    threshold = self.robot_radius
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
        start_time = self._start_generation_time()
        progress = self._create_progress("Updating distance map")
        if progress is not None:
            progress.start()
            processed_cells = 0
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
            if progress is not None:
                processed_cells += 1
                if processed_cells % 1000 == 0:
                    progress.update(processed_cells, details="wavefront cells processed")
        
        # # determine which cells were affected
        # changed_mask = (previous_D != self.D)
        # # store as tuples
        # self.affected_distance_cells = {(int(y), int(x)) for y, x in np.argwhere(changed_mask)}
        
        self._generation_seconds_total += self._generation_seconds(start_time)
        
        if progress is not None:
            progress.finish(True, f"processed {processed_cells} wavefront cells")
        self._record("distance_map", D=self.D, obst=self.obst)
        self._record("voronoi_skeleton", voro=self.voro)
        
        self.build_graph()

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
        
        # threshold = 1

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

    def apply_layout_updates(self, obstacle_updates, new_access_points, new_interaction_points):
        start_time = self._start_generation_time()

        # clear affected cells
        self.affected_voro_cells.clear()
        self.affected_distance_cells.clear()

        scaled_updates = []
        for update in obstacle_updates:
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
        source_robot_radius = self.robot_radius / self.scale_factor
        self.access_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1, direction)
            for y, x, direction in offset_access_points(new_access_points, source_robot_radius)
        ]
        self.interaction_points = [(y * self.scale_factor + 1, x * self.scale_factor + 1) for y, x in new_interaction_points]

        self.special_point_connections.fill(False)
        self.access_point_connection_cells = []
        self.interaction_point_connection_cells = []
        self.iteration += 1
        
        self._generation_seconds_total += self._generation_seconds(start_time)
        
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)


    def _find_special_point_connections(self):
        """
        Finds the first Voronoi cell reached by a ray from each special point.
        
        NOTE theory, why that seems to alway work: Access point rays always meet a Voronoi cell before an obstacle.
        There is always a Voronoi cell between two obstacles, so there is always a Voronoi cell between the access point's obstacle
        and any other, the ray is headed towards.
        """
        
        dirs = {'n': (-1, 0), 's': (1, 0), 'e': (0, 1), 'w': (0, -1)} # dir: (dy, dx)
        
        progress = self._create_progress("Connecting special points to Voronoi skeleton")
        if progress is not None:
            total_special_points = len(self.access_points) + len(self.interaction_points)
            progress.start(f"target: {total_special_points} special points")

        self.special_point_connections.fill(False) # reset special point connection map
        self.access_point_connection_cells = [] # reset access point connection cells
        self.interaction_point_connection_cells = [] # reset interaction point connection cells
        if progress is not None:
            connected_access_points = 0
            processed_special_points = 0
        for ap in self.access_points:
            
            y, x, dir = ap
            # determine the direction vector based on the dir label
            dir_vec = dirs[dir]
            # while 0 <= int(y) < self.height and 0 <= int(x) < self.width:
            while True:
                # get current cell
                current_cell = (int(y), int(x))

                # if on a Voronoi cell, save it as the access point's connection cell
                if self.voro[current_cell]:
                    self.access_point_connection_cells.append(current_cell)
                    if progress is not None:
                        connected_access_points += 1
                    break
                
                # move in the direction of the dir label
                y += dir_vec[0]
                x += dir_vec[1]
            if progress is not None:
                processed_special_points += 1
                if processed_special_points % 10 == 0 or processed_special_points == total_special_points:
                    progress.update(
                        processed_special_points,
                        total_special_points,
                        f"access points connected: {connected_access_points}/{len(self.access_points)}",
                    )

        if progress is not None:
            connected_interaction_points = 0
        for iop in self.interaction_points:
            y, x = iop
            best_connection_cell = None

            # store the current position for every active ray
            rays = [(x, y, dir_vec) for dir_vec in dirs.values()]

            # advance every active ray by one cell per iteration
            while rays and best_connection_cell is None:
                next_rays = []
                for curr_x, curr_y, dir_vec in rays:
                    current_cell = (int(curr_y), int(curr_x))
                    if self.voro[current_cell]: # if on skeleton, this is one of the shortest connections
                        best_connection_cell = current_cell
                        break

                    # if hit an obstacle, stop searching in this direction
                    if self.occupancy_grid[current_cell] > 0:
                        continue

                    # move in the direction of the dir label
                    next_rays.append((curr_x + dir_vec[1], curr_y + dir_vec[0], dir_vec))

                rays = next_rays

            self.interaction_point_connection_cells.append(best_connection_cell)
            if best_connection_cell is not None:
                if progress is not None:
                    connected_interaction_points += 1
            elif progress is not None:
                progress.warning(f"Interaction point {iop} does not reach the Voronoi skeleton")

            if progress is not None:
                processed_special_points += 1
                if processed_special_points % 10 == 0 or processed_special_points == total_special_points:
                    progress.update(
                        processed_special_points,
                        total_special_points,
                        f"Interaction points connected: {connected_interaction_points}/{len(self.interaction_points)}",
                    )

        if progress is not None:
            access_connected = len(self.access_point_connection_cells)
            progress.finish(
                connected_interaction_points == len(self.interaction_points)
                and access_connected == len(self.access_points),
                f"access points: {access_connected}/{len(self.access_points)} | "
                f"Interaction points: {connected_interaction_points}/{len(self.interaction_points)}",
                completed=total_special_points,
                total=total_special_points,
            )


    def _special_point_connection_on_cell(self, cell):
        """
        Checks if a given cell is a connection cell for any special point.
        """
        return cell in self.access_point_connection_cells or cell in self.interaction_point_connection_cells
    
    # for testing and validation only
    def _get_connected_components(self):
        if not self.graph or not self.graph["nodes"]:
            return []
        nodes = set(self.graph["nodes"].keys())
        adjacency = defaultdict(set)
        for v1, v2 in self.graph["edges"]:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
        components = []
        while nodes:
            start = nodes.pop()
            component = {start}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor in nodes:
                        nodes.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    # for testing and validation only
    def _is_graph_connected(self):
        components = self._get_connected_components()
        if len(components) == 1:
            return True
        else:
            return False
    
    def _diagonal_crosses_blocked_corner(self, start_cell, end_cell, blocked):
        """
        Returns True if a one-cell diagonal crosses the corner of a blocked cell.
        """
        start_y, start_x = start_cell
        end_y, end_x = end_cell

        dy = end_y - start_y
        dx = end_x - start_x

        # This rule applies only to directly diagonal neighbouring cells.
        if abs(dy) != 1 or abs(dx) != 1:
            return False

        side_cell_1 = (start_y + dy, start_x)
        side_cell_2 = (start_y, start_x + dx)

        return blocked[side_cell_1] or blocked[side_cell_2]
    
    def build_graph(self):
        """
        "Sparse 3D Topological Graphs for Micro-Aerial Vehicle Planning",
        Helen Oleynikova, Zachary Taylor, Roland Siegwart, Juan Nieto (2018)
        
        https://github.com/ethz-asl/mav_voxblox_planning/blob/master/voxblox_skeleton/src/skeleton_generator.cpp
        
        """
        start_time = self._start_generation_time()
        
        # self.thin() # not working: deleting essential endpoints. TODO remove, or find better solution
        # perhaps modify "Building Skeleton Models via 3D Medial Surface/Axis Thinning Algorithms", T.-C. Lee, R. Kashyap, C.-N. Chu (1994)
        
        self._find_special_point_connections()
        
        # 0 = free cell, 1 = skeleton cell (on free cell), 2 = occupied cell
        voronoi_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        voronoi_grid[self.voro] = 1
        obstacle_grid = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        obstacle_grid[self.occupancy_grid > 0] = 2
        grid = voronoi_grid + obstacle_grid
        
        progress = self._create_progress("Generating graph from Voronoi skeleton")
        if progress is not None:
            skeleton_cell_count = int(np.sum(voronoi_grid))
            progress.start(f"skeleton cells: {skeleton_cell_count}")

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
        # # neighbor cell offsets for 4-connectivity
        # directions = [(-1, 0), ( 0, -1), ( 0, 1), ( 1, 0)]

        # calculate the degree of each skeleton cell (number of neighbors that are also skeleton cells)
        node_cells = [tuple(cell) for cell in np.argwhere(grid == 1)]
        # degree = {}
        # for point_number, (r, c) in enumerate(skeleton_points, start=1):
        #     degree[(r, c)] = len(neighbors((r, c)))
        #     if point_number % 1000 == 0 or point_number == len(skeleton_points):
        #         progress.update(point_number, len(skeleton_points), "classifying skeleton cells")

        # identify node cells: skeleton cells that are not degree 2 or connect a special point
        # NOTE different from original paper: we also include special point connection cells as nodes, even if they have degree 2, to ensure that special points can be connected to the graph
        # node_cells = [(r, c) for (r, c), deg in degree.items() if deg != 2 or self._special_point_connection_on_cell((r, c))]
        #  = [(r, c) for (r, c), deg in degree.items()]
        
        # create mappings between node_id and node cell coordinates and vice versa
        node_id_to_cell_coords = {} # maps from node_id to cell coordinates of node cells
        cell_coords_to_node_id = {} # maps from cell coordinates of node cells to node_id
        for node_id, cell in enumerate(node_cells):
            node_id_to_cell_coords[node_id] = cell
            cell_coords_to_node_id[cell] = node_id

        # maps from skeleton cell coordinates to the node_id of the graph node that reached it
        cell_coords_to_origin_id = {}

        # queue for flood-fill propagation from each node to trace skeleton paths
        floodfill_queue = deque()

        # initialize the flood-fill queue with the neighbors of each node cell
        for id, coords in node_id_to_cell_coords.items():
            
            # cells that are already graph nodes are assigned to themselves as their origin
            cell_coords_to_origin_id[coords] = id

            for neighbor in neighbors(coords):
                floodfill_queue.append((neighbor, id))

        # graph edges: touple of node_ids (sorted to avoid duplicate edges)
        edges = set()

        # while there are cells to process in the flood-fill queue
        if progress is not None:
            processed_floodfill_cells = 0
        while floodfill_queue:
            # pop the next cell and its source node_id from the queue
            curr_cell_coords, source_node_id = floodfill_queue.popleft()
            
            # stores the node_id of the node that reached this cell for later edge creation
            connected_node_id = -1
            
            # get the node_id of the node that reached this cell, or -1 if it hasn't been reached yet
            origin = cell_coords_to_origin_id.get(curr_cell_coords, -1)

            # if the current cell is not a node cell
            if curr_cell_coords not in cell_coords_to_node_id:

                # if already visited by this flood fill, move on to the next cell in queue
                if origin == source_node_id:
                    continue

                # if already visited by a different flood fill
                if origin != -1:
                    # two different nodes have become connected
                    connected_node_id = origin
                else: # cell has not been visited yet
                    # mark the cell as visited by this flood fill
                    cell_coords_to_origin_id[curr_cell_coords] = source_node_id

                    # add the neighbors of the current cell to the flood-fill queue
                    for neighbor in neighbors(curr_cell_coords):
                        floodfill_queue.append((neighbor, source_node_id))
            else: # cell IS a node cell
                
                # if the current cell is a node cell, get its node_id
                connected_node_id = cell_coords_to_node_id[curr_cell_coords]
                
                # if the connected node is the same as the source node, skip to the next cell in queue
                if connected_node_id == source_node_id:
                    continue
        
            # if a connection between two different nodes was found
            if connected_node_id >= 0:
                # add an edge between the source node and the connected node, if it does not cross a blocked corner
                if not self._diagonal_crosses_blocked_corner(node_id_to_cell_coords[source_node_id], node_id_to_cell_coords[connected_node_id], self.occupancy_grid > 0):
                    edges.add(tuple(sorted((source_node_id, connected_node_id))))
            if progress is not None:
                processed_floodfill_cells += 1
                if processed_floodfill_cells % 1000 == 0:
                    progress.update(processed_floodfill_cells, details="tracing skeleton paths")

        # replace the cell coordinates of node cells with their cell centers
        nodes = {
            node_id: (cell[0] + 0.5, cell[1] + 0.5)
            for node_id, cell in node_id_to_cell_coords.items()
        }

        # create individual graph nodes for special points, so points located on the same cell are not merged
        special_point_nodes = {"access": {}, "interaction": {}}
        next_node_id = len(nodes)
        for point_type, special_points in (("access", self.access_points), ("interaction", self.interaction_points)):
            for point_number, special_point in enumerate(special_points):
                if point_type == "access":
                    connection_cell = self.access_point_connection_cells[point_number]
                else:
                    connection_cell = self.interaction_point_connection_cells[point_number]
                hub_node_id = cell_coords_to_node_id.get(connection_cell)

                # skip special points that could not be connected to the Voronoi skeleton
                if hub_node_id is None:
                    continue

                # add the special point as a node and connect it directly to the Voronoi cell
                special_node_id = next_node_id
                next_node_id += 1
                nodes[special_node_id] = (special_point[0], special_point[1])
                edges.add(tuple(sorted((special_node_id, hub_node_id))))
                special_point_nodes[point_type][point_number] = special_node_id

        self.graph = {
            "nodes": nodes,
            "edges": edges,
            "special_point_nodes": special_point_nodes,
        
        }
        # add the graph building time to the total, do not capture graph saving time
        self._generation_seconds_total += self._generation_seconds(start_time)
        phase = "generation" if self.iteration == 0 else "update"
        self._record_generation_time(self._generation_seconds_total, phase)
        self._generation_seconds_total = 0
        
        # check that the graph is connected (only for testing)
        if progress is not None:
            graph_connected = self._is_graph_connected()
            connected_special_points = sum(len(special_nodes) for special_nodes in special_point_nodes.values())
            progress.finish(
                graph_connected,
                f"nodes: {len(nodes)} | edges: {len(edges)} | "
                f"special points connected: {connected_special_points}/"
                f"{len(self.access_points) + len(self.interaction_points)}",
            )

        if self.recorder is not None:
            save_progress = self._create_progress("Saving graph to disk")
            if save_progress is not None:
                save_progress.start()
            try:
                saved_artifacts = self._record("graph", graph=self.graph, voro=self.voro)
            except Exception as error:
                if save_progress is not None:
                    save_progress.finish(False, f"could not save graph: {error}")
                raise
            if saved_artifacts is None:
                if save_progress is not None:
                    save_progress.finish(True, "skipped; recorder returned no artifact")
            else:
                if save_progress is not None:
                    save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        # return graph_connected
    
# TODO: edge splitting (obstacle avoidance) -- necessary?
#         - probably YES (edges can cross buffer zone, at least for CORNER_SEGMENTS=0)
