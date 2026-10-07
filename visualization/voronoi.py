"""Numerically construct a generalized Voronoi roadmap for polygonal obstacles.

Run from the repository root with::

    python3 -m visualization.voronoi diagram

For every free-space sample, the renderer measures the Euclidean distance to
every obstacle boundary and to the workspace boundary. A generalized Voronoi
edge is retained wherever two boundary features are equally close and neither
is farther away than a third feature.
The same distance field yields a roadmap graph on which Dijkstra's algorithm
finds the highlighted start-to-goal path.
"""

from heapq import heappop, heappush
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np


WORLD_WIDTH = 16.0
WORLD_HEIGHT = 9.0
GRID_COLUMNS = 1600
GRID_ROWS = 900
OUTPUT_DIRECTORY = Path("artifacts/visualization")
RNG_SEED = 20261007
START_CONFIGURATION = (1.6, 1)
GOAL_CONFIGURATION = (13.6, 5.45)

# Values mirror the visual language used by the PRM illustration while keeping
# this renderer independent of optional geometry and graph dependencies.
BACKGROUND_COLOR = "white"
ROADMAP_COLOR = (0.28, 0.28, 0.28, 0.55)
OBSTACLE_COLOR = (0.3, 0.3, 0.3, 1.0)
PATH_COLOR = (0.0, 0.0, 0.0, 1.0)
TERMINAL_EDGE_COLOR = (0.1, 0.1, 0.1, 1.0)
TERMINAL_LABEL_COLOR = "white"
TERMINAL_LABEL_SIZE = 11
FIGURE_SIZE = (9, 5.5)


def _obstacles():
    """Create a reproducible collection of non-overlapping polygonal obstacles."""
    rng = np.random.default_rng(RNG_SEED)
    centres = np.array(((3.4, 6.2), (7.2, 1.9), (10.7, 5.2), (13.4, 7.1)))
    centres += rng.uniform((-0.22, -0.18), (0.22, 0.18), centres.shape)
    configurations = (
        (4, 1.75, 0.28),
        (5, 1.18, -0.42),
        (3, 1.26, 0.08),
        (4, 1.00, 0.36),
    )

    obstacles = []
    for centre, (vertex_count, radius, rotation) in zip(centres, configurations):
        angles = np.linspace(0.0, 2.0 * np.pi, vertex_count, endpoint=False)
        angles += rotation + rng.uniform(-0.12, 0.12, vertex_count)
        radii = radius * rng.uniform(0.79, 1.04, vertex_count)
        obstacles.append(centre + np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))
    return obstacles


def _points_inside_polygon(x, y, polygon):
    """Return a mask for points inside one polygon using vectorized ray casting."""
    inside = np.zeros(x.shape, dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        crosses_ray = (y0 > y) != (y1 > y)
        intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        inside ^= crosses_ray & (x < intersection_x)
        previous = current
    return inside


def _distance_to_polygon(x, y, polygon):
    """Return Euclidean distance from each sample to a polygonal obstacle boundary."""
    distance = np.full(x.shape, np.inf)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        parameter = np.clip(
            ((x - start[0]) * direction[0] + (y - start[1]) * direction[1]) / length_squared,
            0.0,
            1.0,
        )
        nearest_x = start[0] + parameter * direction[0]
        nearest_y = start[1] + parameter * direction[1]
        distance = np.minimum(distance, np.hypot(x - nearest_x, y - nearest_y))
    return distance


def _distance_field(obstacles):
    """Measure each free-space sample's distance to obstacle and world boundaries."""
    x = np.linspace(0.0, WORLD_WIDTH, GRID_COLUMNS)
    y = np.linspace(0.0, WORLD_HEIGHT, GRID_ROWS)
    x_grid, y_grid = np.meshgrid(x, y)
    obstacle_distances = [_distance_to_polygon(x_grid, y_grid, polygon) for polygon in obstacles]
    workspace_boundary_distance = np.minimum.reduce((x_grid, y_grid, WORLD_WIDTH - x_grid, WORLD_HEIGHT - y_grid))
    component_distances = np.stack((*obstacle_distances, workspace_boundary_distance))
    occupied = np.logical_or.reduce(
        [_points_inside_polygon(x_grid, y_grid, polygon) for polygon in obstacles]
    )
    return x, y, component_distances, ~occupied


def _draw_gvd(axis, x, y, distances, free_space):
    """Draw contours that satisfy the generalized Voronoi equidistance condition."""
    obstacle_count = len(distances)
    grid_tolerance = 1.5 * np.hypot(x[1] - x[0], y[1] - y[0])
    for first in range(obstacle_count):
        for second in range(first + 1, obstacle_count):
            others = [index for index in range(obstacle_count) if index not in (first, second)]
            nearest_other = np.min(distances[others], axis=0) if others else np.inf
            pair_is_closest = np.minimum(distances[first], distances[second]) <= nearest_other + grid_tolerance
            equidistance = np.ma.masked_where(
                ~(free_space & pair_is_closest),
                distances[first] - distances[second],
            )
            axis.contour(
                x,
                y,
                equidistance,
                levels=[0.0],
                colors=[ROADMAP_COLOR],
                linewidths=0.8,
                zorder=2,
            )


def _gvd_skeleton(distances, free_space, x, y):
    """Return grid samples close enough to two nearest obstacles to form a GVD graph."""
    nearest_two = np.partition(distances, kth=1, axis=0)[:2]
    tolerance = 1.7 * np.hypot(x[1] - x[0], y[1] - y[0])
    return free_space & (nearest_two[1] - nearest_two[0] <= tolerance)


def _point_in_free_space(point, obstacles):
    """Return whether one query configuration is outside every obstacle."""
    x = np.asarray([[point[0]]])
    y = np.asarray([[point[1]]])
    return all(not _points_inside_polygon(x, y, polygon)[0, 0] for polygon in obstacles)


def _query_points(obstacles):
    """Return the configured terminal configurations after collision validation."""
    terminals = {
        "Start": np.asarray(START_CONFIGURATION, dtype=float),
        "Goal": np.asarray(GOAL_CONFIGURATION, dtype=float),
    }
    invalid_terminals = [
        label for label, point in terminals.items()
        if not _point_in_free_space(point, obstacles)
    ]
    if invalid_terminals:
        raise RuntimeError(f"Terminal configuration lies inside an obstacle: {', '.join(invalid_terminals)}")
    return terminals


def _cell_centre(row, column, x, y):
    """Convert a GVD-grid index into its workspace coordinate."""
    return np.array((x[column], y[row]))


def _line_is_free(start, end, free_space, x, y):
    """Check a terminal connection against the sampled obstacle space."""
    spacing = min(x[1] - x[0], y[1] - y[0])
    sample_count = max(2, int(np.ceil(np.linalg.norm(end - start) / spacing)) + 1)
    samples = np.linspace(start, end, sample_count)
    columns = np.rint((samples[:, 0] / WORLD_WIDTH) * (len(x) - 1)).astype(int)
    rows = np.rint((samples[:, 1] / WORLD_HEIGHT) * (len(y) - 1)).astype(int)
    return np.all(free_space[np.clip(rows, 0, len(y) - 1), np.clip(columns, 0, len(x) - 1)])


def _nearest_visible_skeleton_cell(point, skeleton, free_space, x, y):
    """Find the closest GVD cell which the terminal can reach without collision."""
    candidates = np.argwhere(skeleton)
    candidate_points = np.column_stack((x[candidates[:, 1]], y[candidates[:, 0]]))
    order = np.argsort(np.sum((candidate_points - point) ** 2, axis=1))
    for index in order:
        row, column = candidates[index]
        if _line_is_free(point, candidate_points[index], free_space, x, y):
            return int(row), int(column)
    raise RuntimeError("Could not connect a terminal to the generalized Voronoi roadmap.")


def _shortest_path(skeleton, free_space, x, y, queries):
    """Use Dijkstra's algorithm on the GVD skeleton and return a polyline."""
    start_cell = _nearest_visible_skeleton_cell(queries["Start"], skeleton, free_space, x, y)
    goal_cell = _nearest_visible_skeleton_cell(queries["Goal"], skeleton, free_space, x, y)
    directions = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    distances = {start_cell: 0.0}
    predecessors = {}
    frontier = [(0.0, start_cell)]
    while frontier:
        distance, current = heappop(frontier)
        if current == goal_cell:
            break
        if distance != distances[current]:
            continue
        for delta_row, delta_column in directions:
            neighbour = current[0] + delta_row, current[1] + delta_column
            if not (0 <= neighbour[0] < skeleton.shape[0] and 0 <= neighbour[1] < skeleton.shape[1]):
                continue
            if not skeleton[neighbour]:
                continue
            step_cost = np.hypot(delta_column * (x[1] - x[0]), delta_row * (y[1] - y[0]))
            candidate = distance + step_cost
            if candidate < distances.get(neighbour, np.inf):
                distances[neighbour] = candidate
                predecessors[neighbour] = current
                heappush(frontier, (candidate, neighbour))

    if goal_cell not in distances:
        raise RuntimeError("The start and goal are not connected by the generalized Voronoi roadmap.")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(predecessors[cells[-1]])
    cells.reverse()
    return [
        queries["Start"],
        *(_cell_centre(row, column, x, y) for row, column in cells),
        queries["Goal"],
    ]


def plot_voronoi_diagram():
    """Render the generalized Voronoi roadmap and its shortest graph path."""
    obstacles = _obstacles()
    x, y, distances, free_space = _distance_field(obstacles)
    queries = _query_points(obstacles)
    skeleton = _gvd_skeleton(distances, free_space, x, y)
    shortest_path = _shortest_path(skeleton, free_space, x, y, queries)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    _draw_gvd(axis, x, y, distances, free_space)
    for polygon in obstacles:
        axis.add_patch(Polygon(polygon, closed=True, facecolor=OBSTACLE_COLOR, edgecolor="none", zorder=4))
    axis.add_collection(
        LineCollection(
            list(zip(shortest_path, shortest_path[1:])),
            colors=[PATH_COLOR],
            linewidths=2.2,
            capstyle="round",
            joinstyle="round",
            zorder=5,
        )
    )

    for label, point in queries.items():
        axis.scatter(
            *point,
            s=220,
            color=PATH_COLOR,
            edgecolors=TERMINAL_EDGE_COLOR,
            linewidths=0.8,
            zorder=6,
        )
        axis.annotate(
            "S" if label == "Start" else "G",
            xy=point,
            xytext=(0, -1.5),
            textcoords="offset points",
            color=TERMINAL_LABEL_COLOR,
            fontsize=TERMINAL_LABEL_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
        )

    axis.add_patch(
        Rectangle((0.0, 0.0), WORLD_WIDTH, WORLD_HEIGHT, fill=False,
                  edgecolor=TERMINAL_EDGE_COLOR, linewidth=1.0, zorder=8)
    )
    axis.set_xlim(0.0, WORLD_WIDTH)
    axis.set_ylim(0.0, WORLD_HEIGHT)
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, axis


VISUALIZATIONS = {
    "diagram": (plot_voronoi_diagram, OUTPUT_DIRECTORY / "voronoi_diagram.pdf"),
}


def main(arguments=None):
    """Render the selected Voronoi visualization."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1 or arguments[0] not in VISUALIZATIONS:
        available = ", ".join(VISUALIZATIONS)
        raise SystemExit(f"Choose one Voronoi visualization: {available}")

    plot, output_path = VISUALIZATIONS[arguments[0]]
    figure, _ = plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
