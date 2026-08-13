import sys
import time

from matplotlib.colors import ListedColormap
import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, label

import matplotlib.pyplot as plt


class ScenarioGenerator:
    def __init__(self, seed, height, width, occupancy_rate, sigma_coarse, sigma_fine, number_of_corridors, corridor_sigma, io_points, IO_points_radius, simple_obstacle_rate, agv_radius):
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # random number generator with a fixed seed for reproducibility
        
        self.height = height
        self.width = width
        self.occupancy_rate = occupancy_rate # percentage of the area that should be occupied by obstacles
        
        self.sigma_coarse = sigma_coarse
        self.sigma_fine = sigma_fine
        
        self.number_of_corridors = number_of_corridors
        self.corridor_sigma = corridor_sigma # determines the width of the corridors, smaller values create narrower corridors
        
        self.io_points = set(io_points) # input/output points that should remain free of obstacles (y, x)
        self.access_points = set() # (y, x, direction)
        self.IO_points_radius = IO_points_radius
        self.simple_obstacle_rate = simple_obstacle_rate # how many of the obstacles are regular (non-unit-load) obstacles
        
        self.agv_radius = agv_radius

        self.occupancy = np.zeros((height, width), dtype=np.uint8)

        self.obstacle_count = 0
        
        self.updates = [] # updates made for dynamic scenarios
        
    
    def generate_noise_field(self):
        
        # generate initial noise field
        noise = self.rng.normal(size=(self.height, self.width))
        
        coarse_noise = gaussian_filter(noise, sigma=self.sigma_coarse)
        fine_noise = gaussian_filter(noise, sigma=self.sigma_fine)
        
        field = coarse_noise + fine_noise
        
        # normalize
        field = (field - field.min()) / (field.max() - field.min())

        rr, cc = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")

        # create a penalty field around IO points
        io_penalty = np.zeros_like(field)

        for r, c in self.io_points:
            distance = (rr - r)**2 + (cc - c)**2
            io_penalty += np.exp(-distance / (2 * self.IO_points_radius**2))
            
        field -= io_penalty
        
        # create corridors in the field
        corridor_field = np.zeros_like(field)

        for _ in range(self.number_of_corridors):

            # random corridor orientation
            horizontal_corridor = self.rng.random() > 0.5

            if horizontal_corridor:
                row = self.rng.integers(0, self.height)
                distance = np.abs(rr - row)
            else:
                col = self.rng.integers(0, self.width)
                distance = np.abs(cc - col)

            # gaussian corridor profile
            corridor_field += np.exp(-(distance**2) / (2 * self.corridor_sigma**2))
            
            # print(f"Corridor between {'row' if horizontal_corridor else 'col'} {row if horizontal_corridor else col} with length {distance.max()} created.")

        # normalize corridor field
        if corridor_field.max() > 0:
            corridor_field /= corridor_field.max()

        # make obstacles inside corridors unlikely
        field -= corridor_field

        field = (field - field.min()) / (field.max() - field.min())

        # guarantee IO points are free
        for r, c in self.io_points:
            field[r, c] = 0.0

        return field

    def generate_map(self):

        # generate the noise field
        field = self.generate_noise_field()

        # determine the number of obstacles to place
        unit_loads_num = int(self.height * self.width * self.occupancy_rate * (1-self.simple_obstacle_rate))
        simple_obstacles_num = int(self.height * self.width * self.occupancy_rate) - unit_loads_num
        self.obstacle_count = unit_loads_num + simple_obstacles_num

        obstacles = np.concatenate([
            np.zeros(simple_obstacles_num, dtype=bool), # False: simple obstacles
            np.ones(unit_loads_num, dtype=bool)         # True: unit-load obstacles
        ])
        
        # shuffle the obstacles to randomize their placement
        self.rng.shuffle(obstacles)
        
        unit_loads_placed = 0
        simple_obstacles_placed = 0
        placed = 0
        while obstacles.size > 0:

            # highest remaining obstacle probability
            idx = np.argmax(field)
            r, c = np.unravel_index(idx, field.shape)

            # mark as processed so the same cell is not selected again
            field[r, c] = -np.inf

            # skip if it is an IO point and should remain free
            if (r, c) in self.io_points:
                continue
            
            if obstacles[0]: # if it is a unit-load obstacle: 2
                self.occupancy[r, c] = 2
            else: # if it is a simple obstacle: 1
                self.occupancy[r, c] = 1

            # if the placement does not disconnect the free space, keep it
            if self.connected():
                placed += 1
                # track how many of each type of obstacle has been placed
                if obstacles[0]:
                    unit_loads_placed += 1
                else:
                    simple_obstacles_placed += 1
                # print progress
                if placed % 100 == 0:
                    print(f"Unit lodas placed {unit_loads_placed}/{unit_loads_num} ({unit_loads_placed/unit_loads_num*100:.1f}%)\n"
                          f"Simple obstacles placed {simple_obstacles_placed}/{simple_obstacles_num} ({simple_obstacles_placed/simple_obstacles_num*100:.1f}%)", end="\n\n")
                obstacles = obstacles[1:] # remove the placed obstacle from the list
            else:
                # else: undo placement
                self.occupancy[r, c] = 0

        # determine access points for placed obstacles
        self.update_access_points()
        print(f"\nFinished placing {placed}/{self.obstacle_count} obstacles.")
        

    # check if the free sapce is connected, by checking that the occupancy contains not more than one free component
    # time complexity: O(width*height)
    def connected(self):
        free = self.occupancy == 0

        structure = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ])

        # James R. Weaver, "Centrosymmetric (cross-symmetric) matrices, their basic properties, eigenvalues, and eigenvectors." The American Mathematical Monthly 92.10 (1985): 711-717.
        _, components = label(free, structure=structure)
        return components <= 1 
    
    def update_access_points(self, locations=None):
        directions = [
            (-1, 0, "n"),
            (1,  0, "s"),
            (0, -1, "w"),
            (0,  1, "e")
        ]

        if locations is None:
            # if no locations specified, check all cells
            locations = np.ndindex(self.occupancy.shape)
        else:
            cells = set(locations)

            # include neighbors of specified locaitons
            for r, c in list(cells):
                for dr, dc, _ in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        cells.add((nr, nc))

            locations = cells

        for r, c in locations: # for each cell
            for dr, dc, direction in directions: #  for each direction
                nr, nc = r + dr, c + dc # get neighbor coordinates

                # skip if neighbor is out of bounds
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                
                point = (nr + r + 1) / 2, (nc + c + 1) / 2, direction

                if self.occupancy[r, c] == 2 and self.occupancy[nr, nc] == 0:
                    self.access_points.add(point)
                else:
                    self.access_points.discard(point)

    def make_updates(self, number_of_updates=100):
        changed_cells = set()  # track changed cells to update access points after updates
                
        for _ in range(number_of_updates):
            
            print(f"Making change {_+1}/{number_of_updates}...", end="\r")
            
            action = self.rng.choice(["add", "move", "remove"])
            
            success = False
            while not success:
                
                if action == "add":
                    # randomly select a free cell to add an obstacle
                    free_cells = np.argwhere(self.occupancy == 0)
                    if len(free_cells) == 0:
                        print("No free cells available to add an obstacle.")
                        continue
                    cell = tuple(free_cells[self.rng.integers(len(free_cells))])
                    obst_type = 2 if self.rng.random() > self.simple_obstacle_rate else 1 # try to preserve the ratio of unit-load to simple obstacles
                    success = self._add_obstacle(cell, type=obst_type)  # 1 for simple obstacle
                    if success:
                        changed_cells.add(cell)
                    
                if action == "move":
                    # randomly select an obstacle to move
                    obstacle_cells = np.argwhere(self.occupancy > 0)
                    if len(obstacle_cells) == 0:
                        print("No obstacles available to move.")
                        continue
                    from_cell = tuple(obstacle_cells[self.rng.integers(len(obstacle_cells))])
                    
                    # randomly select a free cell to move the obstacle to
                    free_cells = np.argwhere(self.occupancy == 0)
                    if len(free_cells) == 0:
                        print("No free cells available to move the obstacle.")
                        continue
                    to_cell = tuple(free_cells[self.rng.integers(len(free_cells))])
                    
                    success = self._move_obstacle(from_cell, to_cell)
                    if success:
                        changed_cells.add(from_cell)
                        changed_cells.add(to_cell)
                    
                if action == "remove":
                    # randomly select an obstacle to remove
                    obstacle_cells = np.argwhere(self.occupancy > 0)
                    if len(obstacle_cells) == 0:
                        print("No obstacles available to remove.")
                        continue
                    cell = tuple(obstacle_cells[self.rng.integers(len(obstacle_cells))])
                    success = self._remove_obstacle(cell)
                    if success:
                        changed_cells.add(cell)
        self.update_access_points(changed_cells)
        
    def _add_obstacle(self, cell, type):
        self.occupancy[cell] = type  # 1 for simple obstacle, 2 for unit-load obstacle
        
        if (cell in self.io_points):
            # undo addition if it involves IO points
            self.occupancy[cell] = 0
            return False

        # ensure connectivity after addition
        if not self.connected():
            # undo addition if it disconnects the map
            self.occupancy[cell] = 0
            return False
        
        self.obstacle_count += 1
        
        self.updates.append({"action": "add", "cell": cell, "obstacle_type": type})
        return True
    
    def _move_obstacle(self, from_cell, to_cell):
        
        # do not move if an IO point is at target
        if (to_cell in self.io_points):
            return False
        
        # move the obstacle
        from_val = self.occupancy[from_cell]
        self.occupancy[from_cell] = 0
        self.occupancy[to_cell] = from_val

        # ensure connectivity after move
        if not self.connected():
            # undo move if it disconnects the map
            self.occupancy[from_cell] = from_val
            self.occupancy[to_cell] = 0
            return False
        
        self.updates.append({"action": "move", "from_cell": from_cell, "to_cell": to_cell})
        return True

    def _remove_obstacle(self, cell):
        
        value = self.occupancy[cell]
        self.occupancy[cell] = 0

        # ensure connectivity after removal
        if not self.connected():
            # undo removal if it disconnects the map
            self.occupancy[cell] = value
            return False
        
        self.obstacle_count -= 1
        
        self.updates.append({"action": "remove", "cell": cell})
        return True
    
    def remove_region(self, top_left, bottom_right):

        r1, c1 = top_left
        r2, c2 = bottom_right
        
        # ensure the region is within bounds
        if not (0 <= r1 < r2 <= self.height and 0 <= c1 < c2 <= self.width):
            raise ValueError("Region is out of bounds.")
        
        old_occupancy = self.occupancy.copy()
        
        # remove obstacles in the region
        for r in range(r1, r2):
            for c in range(c1, c2):
                if old_occupancy[r, c] > 0:
                    self.occupancy[r, c] = 0
                    self.updates.append({"action": "remove", "cell": (r, c)})

        if not self.connected():
            # undo removal if it disconnects the map
            self.occupancy = old_occupancy
            raise ValueError("Removing this region disconnects the map.")

        self.update_access_points(locations=[(r, c) for r in range(r1, r2) for c in range(c1, c2)])
        
        
    def visualize(self, iteration=""):

            plt.figure(figsize=(25, 20))

            # cells
            cmap = ListedColormap(["white", "black", "grey"])
            plt.pcolormesh(self.occupancy, cmap=cmap, edgecolors="black", linewidth=0.5, shading="flat")
            
            # access points
            xs = [x for _, x, _ in self.access_points]
            ys = [y for y, _, _ in self.access_points]
            plt.scatter(xs, ys, s=20, color="red", zorder=5)
            plt.gca().invert_yaxis()

            # IO points
            for r, c in self.io_points:
                plt.plot(c + 0.5, r + 0.5, marker="o", markersize=10, markerfacecolor="green", markeredgecolor="black")

            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect("equal")

            plt.savefig(f"scenario_{iteration}.png", dpi=600, bbox_inches="tight")
            plt.show()
            plt.close()

    def save_scenario(self, filename):
        
        np.savez(
            filename,
            seed=self.seed,
            width=self.width,
            height=self.height,
            occupancy=np.array(self.occupancy, dtype=np.uint8),
            occupancy_rate=self.occupancy_rate,
            io_points=np.array(list(self.io_points)),
            io_points_radius=self.IO_points_radius,
            access_points=np.array(list(self.access_points)),
            agv_radius=self.agv_radius,
            updates=np.array(self.updates, dtype=object) if self.updates else np.array([], dtype=object),
        )

def main():
    
    height = int(sys.argv[1]) if len(sys.argv)>1 else 15
    width = int(sys.argv[2]) if len(sys.argv)>2 else 20

    io_points = { # (y, x)
        (1,0),
        (height-1,int(width*0.9)),
        (height-1,int(width*0.8)),
        (int(height*0.5),width-1),
        (int(height*0.6),int(width*0.7)),
    }

    occupancy_rate = float(sys.argv[3]) if len(sys.argv)>3 else 0.3 # area occupied by obstacles
    
    coarse_sigma = float(sys.argv[4]) if len(sys.argv)>4 else 2  # standard deviation for coarse noise
    fine_sigma = float(sys.argv[5]) if len(sys.argv)>5 else 1  # standard deviation for fine noise
    
    seed = int(sys.argv[6]) if len(sys.argv)>6 else 42  # random seed for reproducibility
        
    number_of_corridors = 30  # number of corridors to generate
    corridor_sigma = 1    # determines the width of corridors
        
    IO_points_radius = 3  # how much free space around IO points is created

    simple_obstacle_rate = 0.1 # how common non-unit-load (regular) obstacles should be

    agv_radius = 0.4  # radius of the AGV in block units


    # static_generator = ScenarioGenerator(
    #     seed,
    #     height, width,
    #     occupancy_rate,
    #     coarse_sigma, fine_sigma,
    #     number_of_corridors, corridor_sigma,
    #     io_points, IO_points_radius,
    #     simple_obstacle_rate,
    #     agv_radius,
    # )

    # start = time.perf_counter()
    # static_generator.generate_map()
    # static_generator.save_scenario("scenario")
    # static_generator.visualize(iteration="static")
    
    # print("Elapsed for static scenario:", time.perf_counter()-start)


    dynamic_generator = ScenarioGenerator(
        seed,
        height, width,
        occupancy_rate,
        coarse_sigma, fine_sigma,
        number_of_corridors, corridor_sigma,
        io_points, IO_points_radius,
        simple_obstacle_rate,
        agv_radius,
    )
    
    start = time.perf_counter()
    dynamic_generator.generate_map()
    dynamic_generator.visualize(iteration="dynamic initial")
    
    dynamic_generator.save_scenario("scenario/0")
    
    dynamic_generator.remove_region((int(height * 0.1), int(width * 0.1)), (int(height * 0.4), int(width * 0.4)))
    dynamic_generator.save_scenario("scenario/1")
    
    # dynamic_generator.visualize(iteration="dynamic after region removal")
    
    dynamic_generator.make_updates(10)
    dynamic_generator.save_scenario("scenario/2")

    # dynamic_generator.visualize(iteration="dynamic")
    
    print("Elapsed for dynamic scenario:", time.perf_counter()-start)
if __name__ == "__main__":
    main()