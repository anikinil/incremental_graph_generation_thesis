import sys
import time

from matplotlib.colors import ListedColormap
import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, label

import matplotlib.pyplot as plt


class ScenarioGenerator:
    def __init__(self, seed, height, width, occupancy_rate, sigma_coarse, sigma_fine, noisy_aisles_num, noisy_aisle_sigma, io_points, IO_points_radius, simple_obstacle_rate, agv_radius, ordered_aisle_spacing = (0, 0), ordered_aisle_offset = (0, 0), ordered_aisle_widths = (1, 1), wall_clearance = (0, 0, 0, 0)):
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # random number generator with a fixed seed for reproducibility
        
        self.height = height
        self.width = width
        self.occupancy_rate = occupancy_rate # percentage of the area that should be occupied by obstacles
        
        self.sigma_coarse = sigma_coarse
        self.sigma_fine = sigma_fine
        
        self.noisy_aisles_num = noisy_aisles_num
        self.noisy_aisle_sigma = noisy_aisle_sigma # determines the width of the aisles, smaller values create narrower aisles

        self.io_points = set(io_points) # input/output points that should remain free of obstacles (y, x)
        self.access_points = set() # (y, x, direction)
        self.IO_points_radius = IO_points_radius
        self.simple_obstacle_rate = simple_obstacle_rate # how many of the obstacles are regular (non-unit-load) obstacles
        
        self.agv_radius = agv_radius

        self.ordered_aisle_spacing = ordered_aisle_spacing # (y, x): y rows between aisles, x columns between aisles
        self.ordered_aisle_offset = ordered_aisle_offset # (y, x): offset of the first aisle from the top-left corner
        self.ordered_aisle_widths = ordered_aisle_widths # (y, x): width of the ordered aisles in rows and columns

        self.wall_clearance = wall_clearance  # (north, south, east, west): offset of the wall from the edges

        self.occupancy = np.zeros((height, width), dtype=np.uint8)

        self.obstacle_count = 0
        
        self.updates = [] # updates made for dynamic scenarios
        
        self.connectivity_structure = np.array([[0, 1, 0],
                                                [1, 1, 1],
                                                [0, 1, 0]], dtype=np.uint8) # structure for connectivity checks
        
        self.probability_field = None  # normalized probability field for obstacle placement
    
    def generate_noise_field(self):
        
        # generate initial gaussian noise field
        noise = self.rng.normal(size=(self.height, self.width))
        coarse_noise = gaussian_filter(noise, sigma=self.sigma_coarse)
        fine_noise = gaussian_filter(noise, sigma=self.sigma_fine)
        field = coarse_noise + fine_noise
        
        # normalize field
        field = (field - field.min()) / (field.max() - field.min())
        
        # create a grid containing the row coordinate of every cell
        rr, cc = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")

        # create an obstacle penalty around all I/O points
        io_penalty = np.zeros_like(field)
        for r, c in self.io_points:
            distance = (rr - r) ** 2 + (cc - c) ** 2
            io_penalty += np.exp(-distance / (2 * self.IO_points_radius ** 2))
            
        # reduce obstacle probability around I/O points
        field -= io_penalty

        # create binary mask where cells that must reliably remain free of obstacles are marked 0
        guaranteed_free = np.ones_like(field)

        # add wall clearance to mask
        wall_n, wall_s, wall_e, wall_w = self.wall_clearance
        guaranteed_free[:wall_n, :] = 0.0
        guaranteed_free[-wall_s:, :] = 0.0
        guaranteed_free[:, -wall_e:] = 0.0
        guaranteed_free[:, :wall_w] = 0.0

        # add ordered aisles to mask
        y_spacing, x_spacing = self.ordered_aisle_spacing
        y_offset, x_offset = self.ordered_aisle_offset
        y_width, x_width = self.ordered_aisle_widths
        for r in range(y_offset, self.height, y_spacing + y_width):
            guaranteed_free[r:r+y_width, x_offset:] = 0.0
        for c in range(x_offset, self.width, x_spacing + x_width):
            guaranteed_free[y_offset:, c:c+x_width] = 0.0


        if self.noisy_aisles_num > 0:
            # create a field describing the noisy aisles
            noisy_aisle_field = np.zeros_like(field)

            # generate noisy aisles
            for _ in range(self.noisy_aisles_num):
                horizontal_aisle = self.rng.random() > 0.5
                if horizontal_aisle:
                    row = self.rng.integers(0, self.height)
                    distance = np.abs(rr - row)
                else:
                    col = self.rng.integers(0, self.width)
                    distance = np.abs(cc - col)
                # create a Gaussian-shaped influence around the noisy aisle
                noisy_aisle_field += np.exp(-(distance ** 2) / (2 * self.noisy_aisle_sigma ** 2))
                
            # scale the noisy aisle influence to values between zero and one
            noisy_aisle_field /= noisy_aisle_field.max()
            
            # reduce the obstacle probability near noisy aisles
            field -= noisy_aisle_field

        # force every guaranteed-free cell to have zero obstacle probability
        field[guaranteed_free == 0.0] = 0.0

        # force every I/O point cell to be guaranteed free
        for r, c in self.io_points:
            field[r, c] = 0.0

        # # Clamp the final field so obstacle probabilities cannot be negative
        field = np.clip(field, 0.0, 1.0)
        
        self.probability_field = field
        
    # returns the best free cell for potential obstacle location based on the probability field
    def get_best_obstacle_location(self):
        # find the index of the maximum value in the probability field
        idx = np.unravel_index(np.argmax(self.probability_field), self.probability_field.shape)
        return idx
        
    # returns a randomly selected free cell
    def get_random_cell(self, free):
        # free=True: select a free cell, free=False: select an occupied cell
        free_cells = np.argwhere(self.occupancy != free)
        if len(free_cells) == 0:
            print("No free cells available to add an obstacle.")
            return None
        cell = tuple(free_cells[self.rng.integers(len(free_cells))])
        return cell
        
    def generate_map(self):

        # generate the noise field
        self.generate_noise_field()
        selection_field = self.probability_field.copy()
        
        # determine the number of obstacles to place
        unit_loads_num = int(self.height * self.width * self.occupancy_rate * (1-self.simple_obstacle_rate))
        simple_obstacles_num = int(self.height * self.width * self.occupancy_rate) - unit_loads_num
        self.obstacle_count = unit_loads_num + simple_obstacles_num

        obstacles = np.concatenate([
            np.zeros(simple_obstacles_num, dtype=bool), # False: simple obstacles
            np.ones(unit_loads_num, dtype=bool)         # True: unit-load obstacles
        ])
        
        # shuffle the obstacles to randomize their type when placing
        self.rng.shuffle(obstacles)
        
        unit_loads_placed = 0
        simple_obstacles_placed = 0
        placed_obstacles = 0
        processed_cells = 0
        max_free_cells = np.sum(selection_field > 0) # maximum number of free cells not forbidden by probability field
        while obstacles.size > 0:

            # highest remaining obstacle probability
            idx = np.argmax(selection_field)
            r, c = np.unravel_index(idx, selection_field.shape)

            # mark as processed so the same cell is not selected again
            selection_field[r, c] = -np.inf

            # skip if it is an IO point and should remain free
            if (r, c) in self.io_points:
                continue
            
            if obstacles[0]: # if it is a unit-load obstacle: 2
                self.occupancy[r, c] = 2
            else: # if it is a simple obstacle: 1
                self.occupancy[r, c] = 1

            # if the placement does not disconnect the free space, keep it
            if self.connected():
                placed_obstacles += 1
                # track how many of each type of obstacle has been placed
                if obstacles[0]:
                    unit_loads_placed += 1
                else:
                    simple_obstacles_placed += 1
                # print progress
                if placed_obstacles % 100 == 0:
                    print(f"\r{' ' * 50}\rPlaced {placed_obstacles}/{self.obstacle_count} obstacles - {placed_obstacles/self.obstacle_count*100:.1f}% (unit-load: {unit_loads_placed}, simple: {simple_obstacles_placed})...", end="", flush=True)
                obstacles = obstacles[1:] # remove the placed obstacle from the list
            else:
                # else: undo placement
                self.occupancy[r, c] = 0
                
            processed_cells += 1
            if processed_cells >= max_free_cells:
                print(f"\nWarning: Not enough free cells to place all obstacles. Stopping generation prematurely. Placed {placed_obstacles}/{self.obstacle_count} obstacles.")
                break

        # determine access points for placed obstacles
        self.update_access_points()
        print(f"\nFinished placing {placed_obstacles}/{self.obstacle_count} obstacles.")
        

    # check if the free sapce is connected, by checking that the occupancy contains not more than one free component
    # time complexity: O(width*height)
    def connected(self):
        free = self.occupancy == 0

        # James R. Weaver, "Centrosymmetric (cross-symmetric) matrices, their basic properties, eigenvalues, and eigenvectors." The American Mathematical Monthly 92.10 (1985): 711-717.
        _, components = label(free, structure=self.connectivity_structure)
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

    def make_updates(self, respect_probab_field=True, number_of_updates=100):
        changed_cells = set()  # track changed cells to update access points after map updates
        
        for _ in range(number_of_updates):
            
            print(f"Making change {_+1}/{number_of_updates}...", end="\r")
            
            action = self.rng.choice(["add", "move", "remove"])
            
            success = False
            while not success:
                
                # only respect the probability field when adding or moving obstacles
                # removal cannot violate forbidden areas
                if action == "add" or action == "move":
                    if respect_probab_field:
                        to_cell = self.get_best_obstacle_location()
                    else: # select random free cell
                        to_cell = self.get_random_cell(free=True)
                
                if action == "add":
                    obst_type = 2 if self.rng.random() > self.simple_obstacle_rate else 1 # try to preserve the ratio of unit-load to simple obstacles
                    success = self._add_obstacle(to_cell, type=obst_type)
                    if success:
                        changed_cells.add(to_cell)

                if action == "move":
                    # randomly select an occupied cell to move the obstacle from
                    from_cell = self.get_random_cell(free=False)
                    
                    success = self._move_obstacle(from_cell, to_cell)
                    if success:
                        changed_cells.add(from_cell)
                        changed_cells.add(to_cell)
                    
                if action == "remove":
                    # randomly select an obstacle to remove
                    cell = self.get_random_cell(free=False)
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
            
            # probability / noise field
            plt.imshow(self.probability_field, cmap="viridis", alpha=0.5, extent=(0, self.width, self.height, 0), zorder=0)
            plt.colorbar(label="Probability")

            # obstacles
            SIMPLE_OBST_COLOR = (0.3, 0.3, 0.3, 1)
            UNIT_LOAD_OBST_COLOR = (0.5, 0.5, 0.5, 1)

            CELL_MARGIN = 0.1
            CELL_SIZE = 1 - 2 * CELL_MARGIN

            for r in range(self.height):
                for c in range(self.width):
                    obstacle_type = self.occupancy[r, c]
                    if obstacle_type == 1:
                        color = SIMPLE_OBST_COLOR
                    elif obstacle_type == 2:
                        color = UNIT_LOAD_OBST_COLOR
                    else:
                        continue

                    rect = plt.Rectangle((c + CELL_MARGIN, r + CELL_MARGIN), CELL_SIZE, CELL_SIZE, facecolor=color, edgecolor="black", linewidth=2, zorder=2)
                    plt.gca().add_patch(rect)

            # IO points
            for r, c in self.io_points:
                circle = plt.Circle((c + 0.5, r + 0.5), radius=CELL_SIZE * 0.2, facecolor="green", edgecolor="black", linewidth=2, zorder=3)
                plt.gca().add_patch(circle)
            
            # access points as arrows
            for r, c, dir in self.access_points:
                if dir == "n":
                    dx, dy = 0, -0.5
                    r += 0.3
                elif dir == "s":
                    dx, dy = 0, 0.5
                    r -= 0.3
                elif dir == "e":
                    dx, dy = 0.5, 0
                    c -= 0.3
                elif dir == "w":
                    dx, dy = -0.5, 0
                    c += 0.3
                else:
                    continue
                arrow = plt.Arrow(c, r, dx, dy, width=0.3, facecolor="red", zorder=3)
                plt.gca().add_patch(arrow)

            plt.xticks(range(self.width + 1))
            plt.yticks(range(self.height + 1))
            plt.grid(which="major", color="white", linewidth=1, alpha=0.5)
            
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
        # (1,0),
        (height-1,int(width*0.9)),
        # (height-1,int(width*0.8)),
        (int(height*0.5),width-1),
        (int(height*0.6),int(width*0.7)),
    }

    occupancy_rate = float(sys.argv[3]) if len(sys.argv)>3 else 0.3 # area occupied by obstacles
    
    print(f"Obstacles required: {int(height*width*occupancy_rate)} ({occupancy_rate*100:.1f}%)")
    
    coarse_sigma = float(sys.argv[4]) if len(sys.argv)>4 else 2  # standard deviation for coarse noise
    fine_sigma = float(sys.argv[5]) if len(sys.argv)>5 else 1  # standard deviation for fine noise
    
    seed = int(sys.argv[6]) if len(sys.argv)>6 else 42  # random seed for reproducibility
        
    noisy_aisles_num = 0  # number of aisles to generate
    noisy_aisle_sigma = 3    # determines the width of aisles
        
    IO_points_radius = 3  # how much free space around IO points is created

    simple_obstacle_rate = 0.1 # how common non-unit-load (regular) obstacles should be

    agv_radius = 0.4  # radius of the AGV in block units


    # static_generator = ScenarioGenerator(
    #     seed,
    #     height, width,
    #     occupancy_rate,
    #     coarse_sigma, fine_sigma,
    #     noisy_aisles_num, noisy_aisle_sigma,
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
        noisy_aisles_num, noisy_aisle_sigma,
        io_points, IO_points_radius,
        simple_obstacle_rate,
        agv_radius,
        ordered_aisle_spacing=(5, 3),
        ordered_aisle_offset=(int(height * 0.2), int(width * 0.3)),
        ordered_aisle_widths=(3, 2),
        wall_clearance=(4, 4, 2, 2) # n, s, e, w
    )
    
    start = time.perf_counter()
    dynamic_generator.generate_map()
    # dynamic_generator.visualize(iteration="dynamic initial")
    
    dynamic_generator.save_scenario("scenario/0")
    
    dynamic_generator.remove_region((int(height * 0.1), int(width * 0.1)), (int(height * 0.4), int(width * 0.4)))
    dynamic_generator.save_scenario("scenario/1")
    
    # dynamic_generator.visualize(iteration="dynamic after region removal")
    
    dynamic_generator.make_updates(respect_probab_field=True, number_of_updates=10)
    dynamic_generator.save_scenario("scenario/2")

    dynamic_generator.visualize(iteration="dynamic")
    
    print("Elapsed for dynamic scenario:", time.perf_counter()-start)
if __name__ == "__main__":
    main()