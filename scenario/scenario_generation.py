import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, label
from shapely.geometry import Point, box
from artifacts.progress import ProgressReportingAlgorithm


class Scenario(ProgressReportingAlgorithm):

    progress_name = "Scenario"

    def __init__(self, rng, height, width, occupancy_rate, sigma_coarse, sigma_fine,
                 noisy_aisles_num, noisy_aisle_sigma, interaction_points,
                 interaction_points_radius, simple_obstacle_rate, robot_radius,
                 ordered_aisle_widths=(0, 0), ordered_aisle_spacing=(0, 0),
                 ordered_aisle_offset=(0, 0), wall_clearance=(0, 0, 0, 0),
                 recorder=None, print_progress=True):
        
        self.rng = rng
        
        self.height = height
        self.width = width
        self.occupancy_rate = occupancy_rate # percentage of the area that should be occupied by obstacles
        
        self.sigma_coarse = sigma_coarse
        self.sigma_fine = sigma_fine
        
        self.noisy_aisles_num = noisy_aisles_num
        self.noisy_aisle_sigma = noisy_aisle_sigma # determines the width of the aisles, smaller values create narrower aisles

        self.interaction_points = set(interaction_points) # input/output points that should remain free of obstacles (y, x)
        self.access_points = set() # (y, x, direction)
        self.interaction_points_radius = interaction_points_radius
        self.simple_obstacle_rate = simple_obstacle_rate # how many of the obstacles are simple (non-unit-load) obstacles
        
        self.robot_radius = robot_radius

        self.ordered_aisle_widths = ordered_aisle_widths # (y, x): width of the ordered aisles in rows and columns (if both zero, no ordered aisles)
        self.ordered_aisle_spacing = ordered_aisle_spacing # (y, x): y rows between aisles, x columns between aisles
        self.ordered_aisle_offset = ordered_aisle_offset # (y, x): offset of the first aisle from the top-left corner

        self.wall_clearance = wall_clearance  # (north, south, east, west): offset of the wall from the edges

        self.occupancy = np.zeros((height, width), dtype=np.uint8)

        self.obstacle_count = 0
        
        self.updates = [] # updates made for dynamic scenarios
        
        self.connectivity_structure = np.array([[0, 1, 0],
                                                [1, 1, 1],
                                                [0, 1, 0]], dtype=np.uint8) # structure for connectivity checks
        
        self.priority_field = None  # normalized priority field for obstacle placement
    
        self.iteration = 0
        
        self.recorder = recorder
        self.print_progress = bool(print_progress)
    
    def _generate_priority_field(self):
        
        # generate random noise sampled from a normal distribution
        noise = self.rng.normal(size=(self.height, self.width))
        
        # smooth the noise with gaussian filters
        coarse_noise = gaussian_filter(noise, sigma=self.sigma_coarse)
        fine_noise = gaussian_filter(noise, sigma=self.sigma_fine)
        
        field = coarse_noise + fine_noise
        
        # normalize field
        field = (field - field.min()) / (field.max() - field.min())
        
        # create a grid containing the row and column indices of each cell
        rr, cc = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")

        # create an obstacle penalty around all interaction points
        interaction_point_penalty = np.zeros_like(field)
        for r, c in self.interaction_points:
            distance = (rr - r) ** 2 + (cc - c) ** 2
            # calculate penalty, using a radial basis function kernel with sigma equal to the interaction points radius
            # https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            interaction_point_penalty += np.exp(-distance / (2 * self.interaction_points_radius ** 2))
            
        # apply the obstacle penalty around interaction points
        field -= interaction_point_penalty
        
        # normalize again, since the penalty may have changed the range of values
        field = (field - field.min()) / (field.max() - field.min())
        
        # create binary mask where cells that must reliably remain free of obstacles are marked 0
        guaranteed_free = np.ones_like(field)

        # add wall clearance to mask
        wall_n, wall_s, wall_e, wall_w = self.wall_clearance
        if wall_n > 0:
            guaranteed_free[:wall_n, :] = 0.0
        if wall_s > 0:
            guaranteed_free[-wall_s:, :] = 0.0
        if wall_e > 0:
            guaranteed_free[:, -wall_e:] = 0.0
        if wall_w > 0:
            guaranteed_free[:, :wall_w] = 0.0

        # add ordered aisles to mask
        y_width, x_width = self.ordered_aisle_widths
        if y_width > 0 and x_width > 0:
            y_spacing, x_spacing = self.ordered_aisle_spacing
            y_offset, x_offset = self.ordered_aisle_offset
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
            
            # apply the noisy aisle influence to the obstacle priority field
            field -= noisy_aisle_field
            
            # normalize again
            field = (field - field.min()) / (field.max() - field.min())

        # force every guaranteed-free cell to have zero obstacle priority
        field[guaranteed_free == 0.0] = 0.0

        # force every interaction point cell to be guaranteed free
        for r, c in self.interaction_points:
            field[int(r), int(c)] = 0.0

        self.priority_field = field
        
    # returns the best free cell for potential obstacle location based on the priority field
    def _get_best_obstacle_location(self):
        # find the index of the maximum value in the priority field
        idx = np.unravel_index(np.argmax(self.priority_field), self.priority_field.shape)
        return idx
        
    # returns a randomly selected free cell
    def _get_random_cell(self, free):
        # free=True: select a free cell, free=False: select an occupied cell
        if free:
            candidate_cells = np.argwhere(self.occupancy == 0)
        else:
            candidate_cells = np.argwhere(self.occupancy != 0)
        if len(candidate_cells) == 0:
            return None
        cell = tuple(candidate_cells[self.rng.integers(len(candidate_cells))])
        return cell
        
    def generate(self):
        # determine the number of obstacles to place
        unit_loads_num = int(self.height * self.width * self.occupancy_rate * (1-self.simple_obstacle_rate))
        simple_obstacles_num = int(self.height * self.width * self.occupancy_rate) - unit_loads_num
        target_obstacle_count = unit_loads_num + simple_obstacles_num
        progress = self._create_progress("Generating obstacle layout")
        if progress is not None:
            progress.start(f"target: {target_obstacle_count} obstacles")

        # generate the object placement priority field
        self._generate_priority_field()
        selection_field = self.priority_field.copy()

        # create a shuffled list of obstacle types to place
        # 1: simple obstacles, 2: unit-load obstacles        
        obstacles_to_place = [1] * simple_obstacles_num + [2] * unit_loads_num        
        self.rng.shuffle(obstacles_to_place)
        
        if progress is not None:
            unit_loads_placed = 0
            simple_obstacles_placed = 0
        placed_obstacles = 0
        attempted_placements = 0
        unforbidden_cells = np.sum(selection_field > 0) # number of cells not forbidden by the priority field 
        # attempt to place obstacles until all are placed or no more valid placements are possible
        while len(obstacles_to_place) > 0 and attempted_placements < unforbidden_cells:

            # highest remaining obstacle priority
            idx = np.argmax(selection_field)
            r, c = np.unravel_index(idx, selection_field.shape)

            # mark as attempted so the same cell is not selected again
            selection_field[r, c] = -np.inf
            attempted_placements += 1

            self.occupancy[r, c] = obstacles_to_place[0] # place the next obstacle in the list

            # if the placement does not disconnect the free space, keep it
            if self.connected():
                placed_obstacles += 1
                # track how many of each type of obstacle has been placed
                if progress is not None:
                    if obstacles_to_place[0] == 2:
                        unit_loads_placed += 1
                    else:
                        simple_obstacles_placed += 1
                    if placed_obstacles % 100 == 0:
                        progress.update(
                            placed_obstacles,
                            target_obstacle_count,
                            f"unit-load: {unit_loads_placed} | simple: {simple_obstacles_placed}",
                        )
                obstacles_to_place = obstacles_to_place[1:] # remove the placed obstacle from the list
            else:
                # else: undo placement
                self.occupancy[r, c] = 0 

        # determine access points for placed obstacles
        self.update_access_points()
        self.obstacle_count = placed_obstacles
        
        # if not all obstacles could be placed, the scenario generation failed
        success = len(obstacles_to_place) == 0
        if progress is not None:
            if success:
                details = (
                    f"placed {placed_obstacles}/{target_obstacle_count} obstacles | "
                    f"unit-load: {unit_loads_placed} | simple: {simple_obstacles_placed}"
                )
            else:
                details = (
                    "not enough occupiable cells; stopped prematurely after "
                    f"{placed_obstacles}/{target_obstacle_count} obstacles | "
                    f"unit-load: {unit_loads_placed} | simple: {simple_obstacles_placed}"
                )
            progress.finish(success, details, completed=placed_obstacles, total=target_obstacle_count)
        
        if self.recorder is not None:
            self._record_scenario()

        return success

    def _record_scenario(self):
        self.recorder.set_scenario({
            "occupancy": self.occupancy,
            "priority_field": self.priority_field,
            "access_points": list(self.access_points),
            "interaction_points": list(self.interaction_points),
            "height": self.height,
            "width": self.width,
            "robot_radius": self.robot_radius,
        # }, plot=True, 
        }, plot=False, 
        iteration=self.iteration)


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
            for dr, dc, direction in directions: # for each direction
                nr, nc = r + dr, c + dc # get neighbor coordinates

                # if neighbor is not out of bounds
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    
                    # the access point is the midpoint between the obstacle cell and the free neighbor cell, with direction towards the free neighbor
                    point = (nr + r + 1) / 2, (nc + c + 1) / 2, direction

                    if self.occupancy[r, c] == 2 and self.occupancy[nr, nc] == 0:
                        self.access_points.add(point)
                    else:
                        self.access_points.discard(point)

    def make_updates(self, respect_probab_field=True, number_of_updates=100):
        """Make random updates to the scenario by adding, moving, or removing obstacles."""
        self.iteration += 1
        changed_cells = set()  # track changed cells to update access points after map updates
        progress = self._create_progress("Generating dynamic updates")
        if progress is not None:
            progress.start(f"target: {number_of_updates} updates")
        
        for _ in range(number_of_updates):
            action = self.rng.choice(["add", "move", "remove"])

            if action in {"move", "remove"} and not np.any(self.occupancy):
                raise ValueError(f"Cannot {action} an obstacle: the scenario contains no obstacles.")
            
            success = False
            while not success:
                
                # only respect the priority field when adding or moving obstacles
                # removal cannot violate forbidden areas
                if action == "add" or action == "move":
                    if respect_probab_field:
                        to_cell = self._get_best_obstacle_location()
                    else: # select random free cell
                        to_cell = self._get_random_cell(free=True)
                
                if action == "add":
                    obst_type = 2 if self.rng.random() > self.simple_obstacle_rate else 1 # try to preserve the ratio of unit-load to simple obstacles
                    success = self._add_obstacle(to_cell, type=obst_type)
                    if success:
                        changed_cells.add(to_cell)

                if action == "move":
                    # randomly select an occupied cell to move the obstacle from
                    from_cell = self._get_random_cell(free=False)
                    
                    success = self._move_obstacle(from_cell, to_cell)
                    if success:
                        changed_cells.add(from_cell)
                        changed_cells.add(to_cell)
                    
                if action == "remove":
                    # randomly select an obstacle to remove
                    cell = self._get_random_cell(free=False)
                    success = self._remove_obstacle(cell)
                    if success:
                        changed_cells.add(cell)
            if progress is not None:
                progress.update(_ + 1, number_of_updates, f"latest action: {action}")
        self.update_access_points(changed_cells)
        
        if progress is not None:
            progress.finish(
                True,
                f"applied {number_of_updates}/{number_of_updates} updates",
                completed=number_of_updates,
                total=number_of_updates,
            )
        if self.recorder is not None:
            self._record_scenario()

    def consume_updates(self):
        """Return and clear the changes since the last consumer update."""
        updates = self.updates
        self.updates = []
        return updates
    

    def add_obstacles_randomly(self, count, type=None, max_attempts=None):
        """Add a specified number of specified types of obstacles at random locations to the scenario while preserving connectivity.
        If type is not specified, a random type will be chosen based on the simple_obstacle_rate.
        """
        if max_attempts is None:
            max_attempts = count * 1000

        added_cells = []
        attempts = 0
        while len(added_cells) < count:
            if attempts >= max_attempts:
                raise RuntimeError( f"Could only add {len(added_cells)}/{count} obstacles after {max_attempts} attempts.")
            
            cell = self._get_random_cell(free=True)
            if cell is None:
                raise RuntimeError("No more free cells available.")

            if type is None:
                obst_type = 2 if self.rng.random() > self.simple_obstacle_rate else 1
            else:
                obst_type = type

            success = self._add_obstacle(cell, type=obst_type)
            if success:
                added_cells.append(cell)
            attempts += 1

    def _obstacle_blocks_interaction_point(self, cell):
        row, column = cell
        obstacle_space = box(column, row, column + 1, row + 1).buffer(self.robot_radius, quad_segs=16, join_style="round")
        for interaction_row, interaction_column in self.interaction_points:
            if obstacle_space.covers(Point(interaction_column, interaction_row)):
                return True
        return False
            
    def remove_obstacles_randomly(self, count, max_attempts=None):
        """Remove a specified number of obstacles at random locations from the scenario while preserving connectivity."""
        if max_attempts is None:
            max_attempts = count * 1000

        removed_cells = []
        attempts = 0
        while len(removed_cells) < count:
            if attempts >= max_attempts:
                raise RuntimeError(f"Could only remove {len(removed_cells)}/{count} obstacles after {max_attempts} attempts.")

            cell = self._get_random_cell(free=False)
            if cell is None:
                raise RuntimeError("No obstacles remain to remove.")

            if self._remove_obstacle(cell):
                removed_cells.append(cell)
            attempts += 1
            
            
    def _add_obstacle(self, cell, type):
        """Add an obstacle of the specified type at the given cell if it does not disconnect the free space."""
        if self._obstacle_blocks_interaction_point(cell):
            return False

        self.occupancy[cell] = type  # 1 for simple obstacle, 2 for unit-load obstacle

        # ensure connectivity after addition
        if not self.connected():
            # undo addition if it disconnects the map
            self.occupancy[cell] = 0
            return False
        
        self.obstacle_count += 1
        
        self.updates.append({"action": "add", "cell": cell, "obstacle_type": type})
        return True

    def _move_obstacle(self, from_cell, to_cell):
        """Move an obstacle from one cell to another if it does not disconnect the free space."""
        
        if self._obstacle_blocks_interaction_point(to_cell):
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
        """Remove an obstacle from the given cell if it does not disconnect the free space."""
        
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
        """Remove all obstacles in a rectangular region."""

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
        
        self.iteration += 1
        if self.recorder is not None:
            self._record_scenario()
        
    def add_region(self, top_left, bottom_right, obstacle_type=2):
        """Add obstacles of the specified type in a rectangular region."""
        r1, c1 = top_left
        r2, c2 = bottom_right
        
        # ensure the region is within bounds
        if not (0 <= r1 < r2 <= self.height and 0 <= c1 < c2 <= self.width):
            raise ValueError("Region is out of bounds.")
        
        old_occupancy = self.occupancy.copy()
        
        # add obstacles in the region
        for r in range(r1, r2):
            for c in range(c1, c2):
                if (r, c) not in self.interaction_points:
                    self.occupancy[r, c] = obstacle_type
                    self.updates.append({"action": "add", "cell": (r, c), "obstacle_type": obstacle_type})

        if not self.connected():
            # undo addition if it disconnects the map
            self.occupancy = old_occupancy
            raise ValueError("Adding this region disconnects the map.")

        self.update_access_points(locations=[(r, c) for r in range(r1, r2) for c in range(c1, c2)])
        
        self.iteration += 1
        if self.recorder is not None:
            self._record_scenario()
