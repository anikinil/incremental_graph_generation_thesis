import networkx as nx

from artifacts.progress import GenerationProgress
from scenario.geometry import centered_io_points, offset_access_points


class IncrementalKingsGraph:
    """
    "Continuous-Space Roadmap Generation for Mobile Robot Fleets with Distance Constraints and Geometry-Aware Discretization"
    Rüdt, Enke and Furmans (2026)
    
    maybe:
    "Complete Decentralized Method for On-Line Multi-Robot Trajectory Planning in Well-Formed Infrastructures"
    Cáp, Vokrínek and Kleiner (2015)
    """

    def __init__(self, bounds, agv_radius, occupancy, access_points, io_points, recorder):
        self.bounds = bounds
        self.agv_radius = agv_radius
        self.occupancy = occupancy.copy() # make a copy to avoid mutating the original occupancy grid

        self.access_points = offset_access_points(access_points, agv_radius)
        self.access_points = [(row, col) for row, col, dir in self.access_points] # direction information irrelevant for king's graph
        self.io_points = centered_io_points(io_points)

        self.iteration = 0
        self.recorder = recorder

        self.G = None

    def apply_layout_updates_and_update_graph(self, obstacle_updates, new_access_points, new_io_points):
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

        new_access_points = offset_access_points(new_access_points, self.agv_radius)
        new_access_points = [(row, col) for row, col, dir in new_access_points] # direction information irrelevant for king's graph
        new_io_points = centered_io_points(new_io_points)

        self._refresh_local_grid_topology(changed_cells)
        self._update_special_points(new_access_points, new_io_points)

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
            neighbor_cells = [(neighbor_row, neighbor_col)
                              for neighbor_row in range(max(self.bounds[0], row - 1), min(self.bounds[2], row + 2))
                              for neighbor_col in range(max(self.bounds[1], col - 1), min(self.bounds[3], col + 2))]
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
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor = (row + row_offset, col + col_offset)
                    if neighbor not in grid_cells:
                        continue
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

    def _update_special_points(self, new_access_points, new_io_points):
        """Recompute graph special points after a layout update."""

        # remove old special points from the graph
        for row, col in self.access_points + self.io_points:
            if self.G.has_node((row, col)):
                self.G.remove_node((row, col))

        self._connect_special_points_to_graph(self.G, new_access_points + new_io_points)

        # update the special points in the class
        self.access_points = new_access_points
        self.io_points = new_io_points

    def _connect_special_points_to_graph(self, G, special_points):
        """Add a vertex for each special point and connect it to the closest grid vertex in the graph."""
        special_progress = GenerationProgress("Incremental King's grid", "Connecting special points", self.iteration)
        special_progress.start(f"target: {len(special_points)} special points")
        for point_number, (row, col) in enumerate(special_points, start=1):
            G.add_node((row, col), pos=(col, row))
            closest_grid_vertex = (int(row), int(col))
            if G.has_node(closest_grid_vertex):
                G.add_edge((row, col), closest_grid_vertex)
            else:
                print(f"closest_grid_vertex: {closest_grid_vertex}")
                print(f"occupancy: {self.occupancy[closest_grid_vertex]}")
                message = (f"Special point {row, col} is not adjacent to any free grid vertex.")
                raise ValueError(message)
            if point_number % 10 == 0 or point_number == len(special_points):
                special_progress.update(point_number, len(special_points), f"connected: {point_number}/{len(special_points)}")

        special_progress.finish(True, f"connected: {len(special_points)}/{len(special_points)} | " f"nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

    def build_graph(self):
        G = nx.Graph()
        grid_height = self.bounds[2] - self.bounds[0]
        grid_width = self.bounds[3] - self.bounds[1]
        total_cells = grid_height * grid_width
        processed_cells = 0
        grid_vertices = 0

        graph_progress = GenerationProgress("Incremental King's grid", "Building King's-grid graph", self.iteration)
        graph_progress.start(f"grid: {grid_height}x{grid_width}")

        # add grid vertices and edges
        for row in range(self.bounds[0], self.bounds[2]):
            for col in range(self.bounds[1], self.bounds[3]):
                if self.occupancy[row, col] == 0:
                    grid_vertices += 1
                    G.add_node((row, col), pos=(col + 0.5, row + 0.5), type="grid")
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr != 0 or dc != 0:
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

                processed_cells += 1
                if processed_cells % 1000 == 0 or processed_cells == total_cells:
                    graph_progress.update(processed_cells, total_cells, f"vertices: {grid_vertices} | edges: {G.number_of_edges()}")

        graph_progress.finish(True, f"vertices: {grid_vertices} | edges: {G.number_of_edges()}")

        # add special points as nodes (with labels) and connect them to the closest grid vertex
        special_points = self.access_points + self.io_points

        self._connect_special_points_to_graph(G, special_points)

        self.G = G

    def record_graph(self):
        """Records the current roadmap graph."""
        if self.recorder is None:
            return None
        if self.G is None:
            self.build_graph()

        save_progress = GenerationProgress("Incremental King's grid", "Saving graph to disk", self.iteration)
        save_progress.start()
        try:
            saved_artifacts = self.recorder.save(
                data_type="incremental_kings_grid",
                data={
                    "occupancy": self.occupancy,
                    "agv_radius": self.agv_radius,
                    "access_points": self.access_points,
                    "io_points": self.io_points,
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
            save_progress.finish(False, f"could not save graph: {error}")
            raise

        if saved_artifacts is None:
            save_progress.finish(True, "skipped; recorder returned no artifact")
            return None

        save_progress.finish(True, f"saved to {saved_artifacts['pkl']}")
        return saved_artifacts
