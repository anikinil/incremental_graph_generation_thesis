import networkx as nx

from artifacts.progress import ProgressReportingAlgorithm
from algorithms.geometry import offset_access_points


class IncrementalKingsGraph(ProgressReportingAlgorithm):
    """
    "Continuous-Space Roadmap Generation for Mobile Robot Fleets with Distance Constraints and Geometry-Aware Discretization"
    Rüdt, Enke and Furmans (2026)
    
    maybe:
    "Complete Decentralized Method for On-Line Multi-Robot Trajectory Planning in Well-Formed Infrastructures"
    Cáp, Vokrínek and Kleiner (2015)
    """

    progress_name = "Incremental King's grid"

    def __init__(self, bounds, robot_radius, occupancy, access_points, interaction_points,
                 recorder=None, print_progress=True):
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.occupancy = occupancy.copy() # make a copy to avoid mutating the original occupancy grid

        self.access_points = list(offset_access_points(access_points, robot_radius))
        self.interaction_points = list(interaction_points)

        self.iteration = 0
        self.recorder = recorder
        self.print_progress = bool(print_progress)

        self.G = None

    def apply_layout_updates_and_update_graph(self, obstacle_updates, new_access_points, new_interaction_points):
        """Apply whole layout update and update the graph at changed locations."""
        changed_cells = set()

        for update in obstacle_updates:
            action = update["action"]
            if action == 'add':
                cell, value = update["cell"], update["obstacle_type"]
                self.occupancy[cell] = value
                changed_cells.add(cell)
            elif action == 'remove':
                cell = update["cell"]
                self.occupancy[cell] = 0
                changed_cells.add(cell)
            elif action == 'move':
                cell_from, cell_to = update["from_cell"], update["to_cell"]
                value = self.occupancy[cell_from]
                self.occupancy[cell_from] = 0
                self.occupancy[cell_to] = value
                changed_cells.update((cell_from, cell_to))

        new_access_points = offset_access_points(new_access_points, self.robot_radius)

        self._refresh_local_grid_topology(changed_cells)
        self._update_special_points(new_access_points, list(new_interaction_points))

        self.iteration += 1

        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)


    def _refresh_local_grid_topology(self, changed_cells):
        """Synchronize grid topology in the union of changed-cell neighborhoods."""
        if not changed_cells:
            return

        grid_cells = set() # union of neighborhoods of changed cells
        for row, col in changed_cells:

            # get the 3x3 neighborhood of the changed cell (bounded by grid limits)
            neighbor_cells = []
            for neighbor_row in range(max(self.bounds[0], row - 1), min(self.bounds[2], row + 2)):
                for neighbor_col in range(max(self.bounds[1], col - 1), min(self.bounds[3], col + 2)):
                    neighbor_cells.append((neighbor_row, neighbor_col))

            # add the neighbor cells to the union of neighborhoods
            grid_cells.update(neighbor_cells)

        # remove or add grid vertices where the occupancy has changed
        for row, col in changed_cells:
            if self.occupancy[row, col] == 0:
                self.G.add_node((row, col), pos=(col + 0.5, row + 0.5), type="grid")
            elif self.G.has_node((row, col)) and self.G.nodes[(row, col)].get("type") == "grid":
                self.G.remove_node((row, col))

        # remove edges between grid vertices in the neighborhoods of changed cells
        grid_cells = {
            node for node in grid_cells
            if self.G.has_node(node) and self.G.nodes[node].get("type") == "grid"
        }
        for node in grid_cells:
            for neighbor in list(self.G.neighbors(node)):
                if neighbor in grid_cells and self.G.nodes[neighbor].get("type") == "grid":
                    self.G.remove_edge(node, neighbor)

        # add edges between grid vertices in the neighborhoods of changed cells
        for row, col in grid_cells:
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)): # right, bottom-left, bottom, bottom-right
                neighbor = (row + dr, col + dc)
                if neighbor in grid_cells:
                    if self._grid_edge_is_valid((row, col), neighbor):
                        self.G.add_edge((row, col), neighbor)

    def _grid_edge_is_valid(self, first, second):
        """Return whether two adjacent free cells can be joined."""
        row, col = first
        neighbor_row, neighbor_col = second
        row_offset = neighbor_row - row
        col_offset = neighbor_col - col

        if self.occupancy[row, col] != 0 or self.occupancy[neighbor_row, neighbor_col] != 0: # one of the cells is occupied
            return False
        if abs(row_offset) == 1 and abs(col_offset) == 1: # diagonal neighbor
            #if not touching any occupied corners: True
            return (self.occupancy[row, neighbor_col] == 0 and self.occupancy[neighbor_row, col] == 0)
        return True

    def _update_special_points(self, new_access_points, new_interaction_points):
        """Recompute graph special points after a layout update."""

        # remove old special points from the graph
        for row, col, direction in self.access_points:
            if self.G.has_node((row, col)):
                self.G.remove_node((row, col))
        for row, col in self.interaction_points:
            if self.G.has_node((row, col)):
                self.G.remove_node((row, col))

        self._connect_special_points_to_graph(self.G, new_interaction_points, new_access_points)

        # update the special points in the class
        self.access_points = new_access_points
        self.interaction_points = new_interaction_points

    def _connect_special_points_to_graph(self, G, new_interaction_points, new_access_points):
        """Add a vertex for each special point and connect it to the closest grid vertex in the graph."""
        special_progress = self._create_progress("Connecting special points")
        if special_progress is not None:
            total_special_points = len(new_interaction_points) + len(new_access_points)
            special_progress.start(f"target: {total_special_points} special points")
            processed_special_points = 0

        for row, col in new_interaction_points:
            G.add_node((row, col), pos=(col, row))
            closest_grid_vertex = (int(row), int(col))
            if G.has_node(closest_grid_vertex):
                G.add_edge((row, col), closest_grid_vertex)
            else:
                message = f"Special point {row, col} is not adjacent to free grid vertex {closest_grid_vertex}."
                raise ValueError(message)
            if special_progress is not None:
                processed_special_points += 1
                if (processed_special_points % 10 == 0
                        or processed_special_points == total_special_points):
                    special_progress.update(
                        processed_special_points,
                        total_special_points,
                        f"connected: {processed_special_points}/{total_special_points}",
                    )

        for row, col, direction in new_access_points:
            G.add_node((row, col), pos=(col, row))
            
            # if the special point is an access point, connect it to the grid vertex that is directed towards the access point's direction
            closest_grid_vertex = (int(row), int(col)) # for east and south always the case (for north and west, need to check if the special point is on the grid line (robot_radius == 0))
            if direction == "n" and row == int(row): # for robot_radius == 0: needs to move one cell north
                closest_grid_vertex = (closest_grid_vertex[0] - 1, closest_grid_vertex[1])
            elif direction == "w" and col == int(col): # for robot_radius == 0: needs to move one cell west
                closest_grid_vertex = (closest_grid_vertex[0], closest_grid_vertex[1] - 1)

            if G.has_node(closest_grid_vertex):
                G.add_edge((row, col), closest_grid_vertex)
            else:
                message = f"Special point {row, col} is not adjacent to free grid vertex {closest_grid_vertex}."
                raise ValueError(message)
            if special_progress is not None:
                processed_special_points += 1
                if (processed_special_points % 10 == 0
                        or processed_special_points == total_special_points):
                    special_progress.update(
                        processed_special_points,
                        total_special_points,
                        f"connected: {processed_special_points}/{total_special_points}",
                    )

        if special_progress is not None:
            special_progress.finish(
                True,
                f"connected: {total_special_points}/{total_special_points} | "
                f"nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}",
                completed=total_special_points,
                total=total_special_points,
            )

    def build_graph(self):
        G = nx.Graph()
        graph_progress = self._create_progress("Building King's-grid graph")
        if graph_progress is not None:
            grid_height = self.bounds[2] - self.bounds[0]
            grid_width = self.bounds[3] - self.bounds[1]
            total_cells = grid_height * grid_width
            processed_cells = 0
            grid_vertices = 0
            graph_progress.start(f"grid: {grid_height}x{grid_width}")

        # add grid vertices and edges
        for row in range(self.bounds[0], self.bounds[2]):
            for col in range(self.bounds[1], self.bounds[3]):
                if self.occupancy[row, col] == 0:
                    if graph_progress is not None:
                        grid_vertices += 1
                    G.add_node((row, col), pos=(col + 0.5, row + 0.5), type="grid")
                    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)): # right, bottom-left, bottom, bottom-right
                        neighbor_row = row + dr
                        neighbor_col = col + dc
                        # neighbor in bound
                        if (self.bounds[0] <= neighbor_row < self.bounds[2]
                            and self.bounds[1] <= neighbor_col < self.bounds[3]
                            and self.occupancy[neighbor_row, neighbor_col] == 0):
                            if abs(dr) + abs(dc) == 2: # diagonal neighbor
                                # check if diagonal neighbor is blocked by obstacles
                                if (self.occupancy[row, neighbor_col] == 0 and self.occupancy[neighbor_row, col] == 0):
                                    G.add_edge((row, col), (neighbor_row, neighbor_col))
                            else: # orthogonal neighbor
                                G.add_edge((row, col), (neighbor_row, neighbor_col))

                if graph_progress is not None:
                    processed_cells += 1
                    if processed_cells % 1000 == 0 or processed_cells == total_cells:
                        graph_progress.update(processed_cells, total_cells, f"vertices: {grid_vertices} | edges: {G.number_of_edges()}")

        if graph_progress is not None:
            graph_progress.finish(
                True,
                f"vertices: {grid_vertices} | edges: {G.number_of_edges()}",
                completed=total_cells,
                total=total_cells,
            )

        # add special points as nodes (with labels) and connect them to the closest grid vertex
        self._connect_special_points_to_graph(G, self.interaction_points, self.access_points)

        self.G = G

    def record_graph(self):
        """Records the current roadmap graph."""
        if self.recorder is None:
            return None
        if self.G is None:
            self.build_graph()

        save_progress = self._create_progress("Saving graph to disk")
        if save_progress is not None:
            save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="incremental_kings_grid",
                data={
                    "occupancy": self.occupancy,
                    "robot_radius": self.robot_radius,
                    "access_points": self.access_points,
                    "interaction_points": self.interaction_points,
                    "bounds": self.bounds,
                    "vertices": {
                        node: {
                            "pos": tuple(attributes.get("pos", (node[1] + 0.5, node[0] + 0.5))),
                            "type": attributes.get("type", "grid"),
                        }
                        for node, attributes in self.G.nodes(data=True)
                    },
                    "edges": sorted(tuple(sorted(edge)) for edge in self.G.edges),
                },
                plot=True,
            )
        except Exception as error:
            if save_progress is not None:
                save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            if save_progress is not None:
                save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        if save_progress is not None:
            save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts
