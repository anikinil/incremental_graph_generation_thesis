"""Static illustration of an eight-connected King's-grid roadmap.

Run from the repository root with::

    python3 -m visualization.kings_graph diagram
"""

from pathlib import Path
import sys
from heapq import heappop, heappush
from math import hypot

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORY = Path("artifacts/visualization")
SINGLE_FIGURE_SIZE = (9, 5.5)

# Shared thesis-visualization palette.
SIMPLE_OBSTACLE_COLOR = (0.3, 0.3, 0.3, 1)
GRAPH_NODE_COLOR = (0.1, 0.4, 0.7, 1)
GRAPH_EDGE_COLOR = (0.1, 0.4, 0.7, 1)
GRID_COLOR = (0, 0, 0, 1)
GRID_ZORDER = 4
GRAPH_NODE_ZORDER = GRID_ZORDER + 1
GRAPH_NODE_SIZE_SCALE = 2
SHORTEST_PATH_COLOR = (0, 0, 0, 1)
TERMINAL_COLOR = (0, 0, 0, 1)
TERMINAL_LABEL_COLOR = "white"
START_CONFIGURATION = (0.22, 4.9)
GOAL_CONFIGURATION = (13.22, 6.3)


OCCUPANCY = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)


def _build_kings_graph(occupancy):
    """Return free-cell nodes and valid 8-connected edges without corner cutting."""
    height, width = occupancy.shape
    nodes = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if occupancy[row, column] == 0
    ]
    edges = []
    for row, column in nodes:
        for row_offset, column_offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour_row = row + row_offset
            neighbour_column = column + column_offset
            if not (0 <= neighbour_row < height and 0 <= neighbour_column < width):
                continue
            if occupancy[neighbour_row, neighbour_column] != 0:
                continue
            if abs(row_offset) == 1 and abs(column_offset) == 1:
                if occupancy[row, neighbour_column] or occupancy[neighbour_row, column]:
                    continue
            edges.append(((row, column), (neighbour_row, neighbour_column)))
    return nodes, edges


def _draw_terminals(axis, positions):
    """Draw terminals, their closest grid nodes, and return those nodes."""
    terminal_nodes = {}
    for configuration, label in ((START_CONFIGURATION, "S"), (GOAL_CONFIGURATION, "G")):
        closest_index = np.argmin(np.sum((positions - configuration) ** 2, axis=1))
        closest_node = positions[closest_index]
        terminal_nodes[label] = tuple(closest_node)
        axis.plot(
            (configuration[0], closest_node[0]),
            (configuration[1], closest_node[1]),
            color=GRAPH_NODE_COLOR,
            linewidth=1.15,
            solid_capstyle="round",
            zorder=5,
        )
        axis.scatter(
            *configuration,
            s=220,
            color=TERMINAL_COLOR,
            edgecolors="none",
            zorder=7,
        )
        axis.annotate(
            label,
            xy=configuration,
            xytext=(0, -1.5),
            textcoords="offset points",
            color=TERMINAL_LABEL_COLOR,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
    return terminal_nodes


def _shortest_grid_path(edges, start, goal):
    """Find the Euclidean-length shortest path between two King's-grid nodes."""
    adjacency = {}
    for left, right in edges:
        left = (left[1] + 0.5, left[0] + 0.5)
        right = (right[1] + 0.5, right[0] + 0.5)
        weight = hypot(right[0] - left[0], right[1] - left[1])
        adjacency.setdefault(left, []).append((right, weight))
        adjacency.setdefault(right, []).append((left, weight))

    distances = {start: 0.0}
    predecessors = {}
    queue = [(0.0, start)]
    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        if node == goal:
            break
        for neighbour, weight in adjacency[node]:
            candidate_distance = distance + weight
            if candidate_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate_distance
                predecessors[neighbour] = node
                heappush(queue, (candidate_distance, neighbour))

    if goal not in distances:
        raise ValueError("The start and goal are not connected in the King's graph.")

    path = [goal]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    return list(reversed(path))


def _draw_shortest_path(axis, edges, terminal_nodes):
    """Highlight the complete shortest start-to-goal route over the roadmap."""
    grid_path = _shortest_grid_path(edges, terminal_nodes["S"], terminal_nodes["G"])
    path = [START_CONFIGURATION, *grid_path, GOAL_CONFIGURATION]
    axis.add_collection(
        LineCollection(
            list(zip(path, path[1:])),
            colors=[SHORTEST_PATH_COLOR],
            linewidths=2.2,
            capstyle="round",
            zorder=6,
        )
    )


def plot_kings_graph():
    """Draw the obstacle grid and the corresponding King's-grid graph."""
    occupancy = OCCUPANCY
    height, width = occupancy.shape
    nodes, edges = _build_kings_graph(occupancy)
    figure, axis = plt.subplots(figsize=SINGLE_FIGURE_SIZE)

    for row, column in zip(*np.where(occupancy)):
        axis.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=SIMPLE_OBSTACLE_COLOR,
                edgecolor="none",
                zorder=1,
            )
        )

    axis.add_collection(
        LineCollection(
            [
                ((left[1] + 0.5, left[0] + 0.5), (right[1] + 0.5, right[0] + 0.5))
                for left, right in edges
            ],
            colors=[GRAPH_EDGE_COLOR],
            linewidths=0.95,
            capstyle="round",
            zorder=2,
        )
    )
    positions = np.array([(column + 0.5, row + 0.5) for row, column in nodes])
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=18 * GRAPH_NODE_SIZE_SCALE,
        color=GRAPH_NODE_COLOR,
        edgecolors="none",
        zorder=GRAPH_NODE_ZORDER,
    )

    grid_segments = (
        [[(column, 0), (column, height)] for column in range(width + 1)]
        + [[(0, row), (width, row)] for row in range(height + 1)]
    )
    axis.add_collection(
        LineCollection(grid_segments, colors=[GRID_COLOR], linewidths=0.55, zorder=GRID_ZORDER)
    )
    terminal_nodes = _draw_terminals(axis, positions)
    _draw_shortest_path(axis, edges, terminal_nodes)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    return figure, axis


VISUALIZATIONS = {
    "diagram": (plot_kings_graph, OUTPUT_DIRECTORY / "kings_graph.pdf"),
}


def main(arguments=None):
    """Render the selected King's-grid visualization."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1 or arguments[0] not in VISUALIZATIONS:
        raise SystemExit(f"Choose one King's-grid visualization: {', '.join(VISUALIZATIONS)}")

    plot, output_path = VISUALIZATIONS[arguments[0]]
    figure, _ = plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
