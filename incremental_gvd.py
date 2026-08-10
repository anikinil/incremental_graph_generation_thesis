"""
Lau, Sprunk, Burgard
"Efficient grid-based spatial representations for robot navigation in dynamic environments"
(Robotics and Autonomous Systems, 2013)
"""

from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

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
    
    def __init__(self, occupancy_grid, access_points, io_points, agv_radius):
        self.occupancy_grid = occupancy_grid
        
        self.height, self.width = occupancy_grid.shape
        
        self.access_points = access_points
        self.io_points = io_points
        
        self.agv_radius = agv_radius ** 2  # squared, since all distances are squared (efficiency)
        
        # distance map: stores distances to closest obstacle for each cell
        self.D = np.full((self.height, self.width), np.inf, dtype=np.float32)
        
        # obstacle reference map: stores (y, x) of closest occupied cell for each cell
        self.obst = np.empty((self.height, self.width), dtype=object)
        print()
        for i in range(self.height):
            for j in range(self.width):
                print(f"Initializing obstacle reference for cell ({i}, {j})", end="\r")
                self.obst[i, j] = (-1, -1)
        
        # voronoi map: True if cell is on Voronoi diagram
        self.voro = np.zeros((self.height, self.width), dtype=bool)
        
        # queue for wavefront propagation
        self.open = BucketPriorityQueue(max_distance=self.height ** 2 + self.width ** 2)  # max possible squared distance in grid
        
        # flag for raise/lower wavefronts
        self.toRaise = np.zeros((self.height, self.width), dtype=bool)
        
        self.affected_cells = set()  # track affected cells after dynamic updates for visualization
        
        # initialize with static brushfire algorithm
        self.computeDistanceMap()
        
    
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
                print(f"Initializing cell ({y}, {x}) with occupancy {self.occupancy_grid[y, x]}", end="\r")
                if self.occupancy_grid[y, x] > 0:  # if cell occupied
                    self.D[y, x] = 0 # distance to itself is zero
                    self.obst[y, x] = s # is its own closest obstacle
                    self.open.insert(s, 0) # enqueue for wavefront propagation: distance=0, cell=s
                # else leave distance as inf and reference as (-1, -1)
                # TODO why do they set it to inf here?
                        
        # propagate wavefronts
        print()
        while not self.open.is_empty():
            print(f"Processing queue with current min distance {self.open.current_min_distance}", end="\r")
            # if self.open.is_empty(): # if no more cells to process: done
                # break
            s, _ = self.open.pop() # pop cell with smallest distance
            self.lower(s) # propagate to neighbors and update their distances if closer to obstacle
    
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
                    self.checkVoro(s, n, threshold=self.agv_radius)  # check Voronoi condition between current cell and neighbor
    
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
        
        self.affected_cells.clear()  # clear affected cells before processing
        
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

    def apply_changes(self, changes):
        
        for change in changes:
            action = change["action"]
            if action == 'add':
                s, value = change["cell"], change["obstacle_type"]
                self.setObstacle(s, value)
            elif action == 'remove':
                s = change["cell"]
                self.removeObstacle(s)
            elif action == 'move':
                s_from, s_to = change["from_cell"], change["to_cell"]
                value = self.removeObstacle(s_from)
                self.setObstacle(s_to, value)

    def update_access_points(self, access_points):
        
        self.access_points = access_points

    def visualize_gvd(self, title=None):
        """Visualize the occupancy grid, distance map, and Voronoi diagram."""
        
        print(f"Visualizing GVD: {title if title else 'unnamed'}...")
        
        fig, ax = plt.subplots(figsize=(25, 20))

        # convert squared distances to actual distances
        distance_overlay = np.sqrt(np.copy(self.D))

        # create RGBA visualization
        vis_map = np.zeros((self.height, self.width, 4), dtype=np.float32)
        
        # highlight cells affected during dynamic updates
        affected_mask = np.zeros((self.height, self.width), dtype=bool)
        for cell in self.affected_cells:
            affected_mask[cell[0], cell[1]] = True
        vis_map[affected_mask] = [1, 0, 0, 0.2]

        # obstacles (black)
        obstacle_mask = self.occupancy_grid == 1
        vis_map[obstacle_mask] = [0, 0, 0, 1]

        # unit loads (dark grey)
        unit_load_mask = self.occupancy_grid == 2
        vis_map[unit_load_mask] = [0.3, 0.3, 0.3, 1]

        # Voronoi cells (dark grey)
        voronoi_mask = (self.voro > 0) & (~obstacle_mask)
        vis_map[voronoi_mask] = [0.3, 0.3, 0.5, 1]

        # plot maps
        # distance_im = ax.imshow(distance_overlay, origin='lower', extent=(0, self.width, 0, self.height), cmap='viridis', alpha=0.5)
        ax.imshow(vis_map, origin='lower', extent=(0, self.width, 0, self.height))

        # fig.colorbar(distance_im, ax=ax, fraction=0.046, pad=0.04, label='Distance')

        # access points in the center of the cell (x and y are swapped due to row/column indexing)
        access_x = [p[0] for p in self.access_points]
        access_y = [p[1] for p in self.access_points]
        ax.scatter(access_x, access_y, s=30, color="red", zorder=3, label="Access points")

        # IO points in the center of the cell
        io_x = [p[0] for p in self.io_points]
        io_y = [p[1] for p in self.io_points]
        ax.scatter(io_x, io_y, s=50, color="green", zorder=3, label="IO points")

        # axis setup
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.margins(0)

        ax.legend(loc='upper right')
        
        ax.invert_yaxis()

        plt.title(title if title else "Incremental GVD")
        plt.show()


def add_outer_walls_to_scenario(scenario):
    
    occupancy = scenario["occupancy"]
    height, width = occupancy.shape
    new_occupancy = np.ones((height + 2, width + 2), dtype=occupancy.dtype)
    new_occupancy[1:-1, 1:-1] = occupancy
    scenario["occupancy"] = new_occupancy
    scenario["width"] += 2
    scenario["height"] += 2
    
    # adjust access points and IO points
    scenario["access_points"] = {(y + 1, x + 1) for (y, x) in scenario["access_points"]}
    scenario["io_points"] = {(y + 1, x + 1) for (y, x) in scenario["io_points"]}
    
    # adjust changes to the grid
    adjusted_changes = []
    for change in scenario["changes"]:
        action = change["action"]
        if action =='add':
            (y, x), value = change["cell"], change["obstacle_type"]
            adjusted_changes.append({"action": action, "cell": (y + 1, x + 1), "obstacle_type": value})
        if action == 'remove':
            y, x = change["cell"]
            adjusted_changes.append({"action": action, "cell": (y + 1, x + 1)})
        elif action == 'move':
            (y_from, x_from), (y_to, x_to) = change["from_cell"], change["to_cell"]
            adjusted_changes.append({"action": action, "from_cell": (y_from + 1, x_from + 1), "to_cell": (y_to + 1, x_to + 1)})
    scenario["changes"] = adjusted_changes
    
    return scenario

def increase_resolution(scenario, scale_factor=3):
    
    occupancy = scenario["occupancy"]
    height, width = occupancy.shape
    
    # create a new high-resolution occupancy grid
    new_occupancy = np.zeros((height * scale_factor, width * scale_factor), dtype=occupancy.dtype)
    
    # fill in the high-resolution grid based on the original occupancy values
    for y in range(height):
        for x in range(width):
            value = occupancy[y, x]
            new_occupancy[y*scale_factor:(y+1)*scale_factor, x*scale_factor:(x+1)*scale_factor] = value
    
    scenario["occupancy"] = new_occupancy
    
    # scale positions of access points and IO points
    scenario["access_points"] = {(y * scale_factor, x * scale_factor) for (y, x) in scenario["access_points"]}
    scenario["io_points"] = {(y * scale_factor, x * scale_factor) for (y, x) in scenario["io_points"]}
    
    # scale agv_radius
    scenario["agv_radius"] *= scale_factor

    # scale changes to the grid
    adjusted_changes = []
    for change in scenario["changes"]:
        action = change["action"]
        if action =='add':
            (y, x), value = change["cell"], change["obstacle_type"]
            for x_ in range(scale_factor):
                for y_ in range(scale_factor):
                    adjusted_changes.append({"action": action, "cell": (y * scale_factor + y_, x * scale_factor + x_), "obstacle_type": value})
        if action == 'remove':
            y, x = change["cell"]
            for x_ in range(scale_factor):
                for y_ in range(scale_factor):
                    adjusted_changes.append({"action": action, "cell": (y * scale_factor + y_, x * scale_factor + x_)})
        elif action == 'move':
            (y_from, x_from), (y_to, x_to) = change["from_cell"], change["to_cell"]
            for x_ in range(scale_factor):
                for y_ in range(scale_factor):
                    adjusted_changes.append({"action": action, "from_cell": (y_from * scale_factor + y_, x_from * scale_factor + x_), "to_cell": (y_to * scale_factor + y_, x_to * scale_factor + x_)})
    scenario["changes"] = adjusted_changes
    
    return scenario

    
if __name__ == "__main__":
    from load_scenario import load_scenario_as_numpy
    
    # load initial scenario and add outer walls
    scenario = load_scenario_as_numpy(0)    
    scenario = increase_resolution(scenario)
    scenario = add_outer_walls_to_scenario(scenario)
    
    # initialize spatial representation and compute distance map
    print("\nInitialize and compute GVD...")
    incr_gvd = IncrementalGVD(
        occupancy_grid=scenario["occupancy"],
        access_points=scenario["access_points"],
        io_points=scenario["io_points"],
        agv_radius=scenario["agv_radius"]
    )
    
    # visualize
    incr_gvd.visualize_gvd("Incremental GVD Results")
    
    # load new scenario and add outer walls
    print("\nApplying changes to the occupancy grid...")
    new_scenario = load_scenario_as_numpy(1)
    new_scenario = increase_resolution(new_scenario)
    new_scenario = add_outer_walls_to_scenario(new_scenario)
    
    # apply changes to the occupancy grid
    scenario_updates = new_scenario["changes"]
    incr_gvd.apply_changes(scenario_updates)
    
    # recompute distance map and voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    # incr_gvd.compute_voronoi_diagram()
    
    # update access points of IncrementalGVD after changes
    new_access_points = new_scenario["access_points"]
    incr_gvd.update_access_points(new_access_points)
    
    # visualize
    incr_gvd.visualize_gvd("Incremental GVD Results After Changes")
    
    
    # load new scenario and add outer walls
    print("\nApplying changes to the occupancy grid...")
    new_scenario = load_scenario_as_numpy(2)
    new_scenario = increase_resolution(new_scenario)
    new_scenario = add_outer_walls_to_scenario(new_scenario)
    
    # apply changes to the occupancy grid
    scenario_updates = new_scenario["changes"]
    incr_gvd.apply_changes(scenario_updates)
    
    # recompute distance map and voronoi diagram after changes
    print("\nUpdating distance map and recomputing Voronoi diagram...")
    incr_gvd.updateDistanceMap()
    # incr_gvd.compute_voronoi_diagram()
    
    # update access points of IncrementalGVD after changes
    new_access_points = new_scenario["access_points"]
    incr_gvd.update_access_points(new_access_points)
    
    # visualize
    incr_gvd.visualize_gvd("Incremental GVD Results After Changes 2")
    