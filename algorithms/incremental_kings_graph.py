import networkx as nx

from artifacts.progress import GenerationProgress
from scenario.geometry import centered_io_points, offset_access_points


class IncrementalKingsGraph:

    def __init__(self, bounds, agv_radius, occupancy, access_points, io_points, recorder):
        self.bounds = bounds
        self.agv_radius = agv_radius
        self.occupancy = occupancy.copy() # make a copy to avoid mutating the original occupancy grid
        self.access_points = offset_access_points(access_points, agv_radius)
        self.io_points = centered_io_points(io_points)

        self.iteration = 0
        self.recorder = recorder
        
        self.G = None

    def apply_layout_updates_and_update_graph(self, obstacle_updates, new_access_points, new_io_points):
        """Apply updates to the occupancy grid and updates the graph incrementally."""
        # apply updates to the occupancy grid
        for update in obstacle_updates:
            action = update["action"]
            if action == 'add':
                cell, value = update["cell"], update["obstacle_type"]
                self._add_obstacle(cell, value)
            elif action == 'remove':
                cell = update["cell"]
                self._remove_obstacle(cell)
            elif action == 'move':
                cell_from, cell_to = update["from_cell"], update["to_cell"]
                value = self.occupancy[cell_from]
                self._remove_obstacle(cell_from)
                self._add_obstacle(cell_to, value)

        self.access_points = offset_access_points(new_access_points, self.agv_radius)
        self.io_points = centered_io_points(new_io_points)

        self.iteration += 1
        self.G = None
        if self.recorder is not None:
            self.recorder.new_iteration(self.iteration)
            
    def _add_obstacle(self, cell, value):
        """Add an obstacle to the occupancy grid and update graph edges accordingly."""
        self.occupancy[cell] = value
        r, c = cell
        try:
            self.G.remove_node((r + 0.5, c + 0.5))  # remove the node corresponding to the obstacle cell
        except nx.NetworkXError: # node not in graph, ignore
            pass

    def _remove_obstacle(self, cell):
        """Remove an obstacle from the occupancy grid at the specified cell."""
        self.occupancy[cell] = 0
        r, c = cell
        try:
            self.G.remove_node((r + 0.5, c + 0.5))  # remove the node corresponding to the obstacle cell
        except nx.NetworkXError: # node not in graph, ignore
            pass

    def build_graph(self):
        G = nx.Graph()
        grid_height = self.bounds[2] - self.bounds[0]
        grid_width = self.bounds[3] - self.bounds[1]
        total_cells = grid_height * grid_width
        processed_cells = 0
        grid_vertices = 0

        graph_progress = GenerationProgress("Incremental King's grid", "Building King's-grid graph", self.iteration)
        graph_progress.start(f"grid: {grid_height}x{grid_width}")

        for row in range(self.bounds[0], self.bounds[2]):
            for col in range(self.bounds[1], self.bounds[3]):
                if self.occupancy[row, col] == 0:
                    grid_vertices += 1
                    G.add_node((row, col), pos=(col + 0.5, row + 0.5), type="graph")
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

        special_points = [(row, col, "access") for row, col, _ in self.access_points] \
                       + [(row, col, "io") for row, col in self.io_points]
        
        special_progress = GenerationProgress("Incremental King's grid", "Connecting special points", self.iteration)
        special_progress.start(f"target: {len(special_points)} special points")

        for point_number, (row, col, point_type) in enumerate(special_points, start=1):
            G.add_node((row, col), pos=(col, row), type=point_type)
            closest_grid_vertex = (int(row), int(col))
            if closest_grid_vertex in G:
                G.add_edge((row, col), closest_grid_vertex)
            else:
                message = (f"{point_type.capitalize()} point {row, col} is not adjacent to any free grid vertex.")
                special_progress.finish(False, message)
                raise ValueError(message)
            if point_number % 10 == 0 or point_number == len(special_points):
                special_progress.update(point_number, len(special_points), f"connected: {point_number}/{len(special_points)}")

        special_progress.finish(True, f"connected: {len(special_points)}/{len(special_points)} | " f"nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

        self.G = G


    def record_graph(self):
        """Records the current roadmap graph."""
        if self.recorder is None:
            return None
        if self.G is None:
            self.build_graph()

        return self.recorder.save(
            data_type="incremental_kings_grid",
            data={
                "occupancy": self.occupancy,
                "agv_radius": self.agv_radius,
                "access_points": self.access_points,
                "io_points": self.io_points,
                "bounds": self.bounds,
                "nodes": {
                    node: {
                        "pos": tuple(attributes.get("pos", (node[1] + 0.5, node[0] + 0.5))),
                        "type": attributes.get("type", "graph"),
                    }
                    for node, attributes in self.G.nodes(data=True)
                },
                "edges": sorted(tuple(sorted(edge)) for edge in self.G.edges),
            },
            plot=True,
        )
