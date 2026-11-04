"""Illustrate a shortest-path change after adding one storage obstacle.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/codex-topology-matplotlib venv/bin/python -m experiments.topology_change_figures

The combined vector PDF uses the existing thesis
plotting style and is saved in a timestamped directory below
``artifacts/visualization/thesis_figures``. Before and after share one legend.
The long path is recomputed with
Euclidean edge weights and the project's polygonal obstacle model.

Edit ``OBSTACLE_LAYOUT`` to change the map, and ``ADDED_CELL`` to change the
obstacle inserted in the second figure. Map dimensions follow the list size.
Walls are drawn outside the grid. Paths use a hidden 0.3-unit corner offset;
robot-radius buffers are not shown.
"""

import argparse
from datetime import datetime
from itertools import combinations
from math import dist
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from algorithms.geometry import SPECIAL_POINT_CLEARANCE_MARGIN, navigation_space
from artifacts.plot import (
    AFFECTED_GRAPH_NODE_COLOR,
    GRAPH_NODE_COLOR,
    ROADMAP_EDGE_ZORDER,
    ROADMAP_NODE_ZORDER,
    SIMPLE_OBSTACLE_COLOR,
    UNIT_LOAD_COLOR,
    VisibilityGraphPlotter,
    _draw_continuous_environment,
    _free_cell_legend_handle,
    _move_legend_to_result_margin,
    _obstacle_legend_handle,
    _register_result_legend,
)
from visualization.access_points_arrows import _draw_walls, _WallLegendHandler
from visualization.meta import (
    PRM_GOAL_COLOR,
    PRM_START_COLOR,
    PRM_TERMINAL_LABEL_COLOR,
    PRM_TERMINAL_LABEL_SIZE,
)


# Edit this grid directly. Keep every row the same length.
# 0 = free cell, 1 = simple obstacle, 2 = unit-load obstacle.
# Walls are automatic outside this grid; do not add a border of wall cells.
# Rows run from top to bottom; columns run from left to right (zero-based).
OBSTACLE_LAYOUT = [
    [2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
    [2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2],
    [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2],
    [2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
]
HEIGHT, WIDTH = len(OBSTACLE_LAYOUT), len(OBSTACLE_LAYOUT[0])
ROBOT_RADIUS = 0.3  # Offset obstacle corners by 0.3 along each coordinate axis.
NUM_BUFFER_SEGMENTS = 0
POINT_A = (4.5, 2.5)
POINT_B = (14.5, 3.5)
ADDED_CELL = (2, 9)  # (row, column) of the obstacle added in the second figure
TERMINAL_SIZE = 220  # Matches visualization.prm, voronoi, and kings_graph.
ADDED_OBSTACLE_COLOR = "#F2D64B"


def make_layouts():
    """Keep all obstacles fixed except one added unit-load cell."""
    before = np.array(OBSTACLE_LAYOUT, dtype=np.uint8)
    after = before.copy()
    after[ADDED_CELL] = 2
    assert np.count_nonzero(before != after) == 1
    return before, after


def shortest_path(occupancy):
    """Find the Euclidean shortest route, with a numerical clearance margin."""
    free, _ = navigation_space(
        occupancy, ROBOT_RADIUS + SPECIAL_POINT_CLEARANCE_MARGIN,
        NUM_BUFFER_SEGMENTS,
    )
    free = free.intersection(box(
        ROBOT_RADIUS, ROBOT_RADIUS, WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS,
    ))
    # Include every free-space boundary ring: after the addition, the island
    # joins the enclosing boundary and changes the free-space boundary ring.
    # Simplification with zero tolerance only removes collinear vertices.
    boundary = free.simplify(0)
    components = [boundary] if boundary.geom_type == "Polygon" else list(boundary.geoms)
    vertices = [POINT_A, POINT_B]
    for component in components:
        for ring in (component.exterior, *component.interiors):
            vertices.extend(ring.coords[:-1])
    vertices = list(dict.fromkeys(vertices))
    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    for left, right in combinations(vertices, 2):
        if free.covers(LineString([left, right])):
            graph.add_edge(left, right, weight=dist(left, right))
    points = nx.shortest_path(graph, POINT_A, POINT_B, weight="weight")
    path = LineString(points)
    obstacles = unary_union([
        box(column, row, column + 1, row + 1)
        for row, column in zip(*occupancy.nonzero())
    ])
    assert path.distance(obstacles) > ROBOT_RADIUS
    assert box(ROBOT_RADIUS, ROBOT_RADIUS, WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS).covers(path)
    return points, path.length


def topology_signature(occupancy):
    """Count connected regions and holes in the robot's free space."""
    free, _ = navigation_space(occupancy, ROBOT_RADIUS, NUM_BUFFER_SEGMENTS)
    free = free.intersection(box(
        ROBOT_RADIUS, ROBOT_RADIUS, WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS,
    ))
    components = [free] if free.geom_type == "Polygon" else list(free.geoms)
    return len(components), sum(len(part.interiors) for part in components)


class TopologyChangePlotter(VisibilityGraphPlotter):
    """Reuse the existing environment rendering and right-hand legend."""

    def draw(self, ax, data):
        _draw_continuous_environment(
            ax, data["occupancy"], bounds=(0, 0, WIDTH, HEIGHT),
        )
        _draw_walls(ax, HEIGHT, WIDTH)
        ax.set_axis_off()
        handles, labels = [], []
        if data["former_path"] is not None:
            former = np.asarray(data["former_path"])
            ax.plot(
                former[:, 0], former[:, 1],
                color=AFFECTED_GRAPH_NODE_COLOR,
                linewidth=2.5, linestyle=(0, (5, 3)),
                zorder=ROADMAP_EDGE_ZORDER,
            )
            handles.append(Line2D(
                [], [], color=AFFECTED_GRAPH_NODE_COLOR,
                linewidth=2.5, linestyle=(0, (5, 3)),
            ))
            labels.append("Blocked path")

        current = np.asarray(data["path"])
        ax.plot(
            current[:, 0], current[:, 1],
            color=GRAPH_NODE_COLOR, linewidth=3,
            solid_capstyle="round", solid_joinstyle="round",
            zorder=ROADMAP_EDGE_ZORDER + 0.2,
        )
        handles.insert(0, Line2D([], [], color=GRAPH_NODE_COLOR, linewidth=3))
        labels.insert(0, "Shortest path")

        handles.insert(0, Line2D(
            [], [], marker="o", linestyle="", color=PRM_START_COLOR,
            markeredgecolor="none", markersize=np.sqrt(TERMINAL_SIZE),
        ))
        labels.insert(0, "Robot positions A and B")
        for point, label, color in (
            (POINT_A, "A", PRM_START_COLOR), (POINT_B, "B", PRM_GOAL_COLOR),
        ):
            ax.scatter(
                *point, s=TERMINAL_SIZE, color=color,
                edgecolors="none", zorder=ROADMAP_NODE_ZORDER,
            )
            ax.annotate(
                label, point, xytext=(0, -1.5), textcoords="offset points",
                ha="center", va="center", fontsize=PRM_TERMINAL_LABEL_SIZE,
                fontweight="bold", color=PRM_TERMINAL_LABEL_COLOR,
                zorder=ROADMAP_NODE_ZORDER + 1,
            )

        cell_entries = [
            (_free_cell_legend_handle(), "Free cell"),
            (_obstacle_legend_handle(UNIT_LOAD_COLOR, "Obstacle"), "Unit-load obstacle"),
        ]
        if np.any(data["occupancy"] == 1):
            cell_entries.insert(1, (
                _obstacle_legend_handle(SIMPLE_OBSTACLE_COLOR, "Simple obstacle"),
                "Simple obstacle",
            ))
        if data["former_path"] is not None:
            row, column = ADDED_CELL
            ax.add_patch(Rectangle(
                (column, row), 1, 1,
                facecolor=ADDED_OBSTACLE_COLOR, edgecolor="none", zorder=2,
            ))
            cell_entries.append((
                _obstacle_legend_handle(ADDED_OBSTACLE_COLOR, "Added obstacle"),
                "Added obstacle",
            ))
        # cell_entries.append((Rectangle(
        #     (0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=6,
        # ), "Walls"))
        _register_result_legend(
            ax, handles, labels, cell_entries=cell_entries,
            handler_map={Rectangle: _WallLegendHandler()},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        Path("artifacts/visualization/thesis_figures")
        / datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    before, after = make_layouts()
    short_path, short_length = shortest_path(before)
    long_path, long_length = shortest_path(after)
    before_signature, after_signature = topology_signature(before), topology_signature(after)
    assert long_length > short_length
    _, new_obstacles = navigation_space(after, ROBOT_RADIUS, NUM_BUFFER_SEGMENTS)
    assert LineString(short_path).intersects(new_obstacles)
    assert box(ADDED_CELL[1], ADDED_CELL[0], ADDED_CELL[1] + 1, ADDED_CELL[0] + 1).intersects(LineString(short_path))

    plotter = TopologyChangePlotter()
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.subplots_adjust(left=0.015, right=0.76, bottom=0.06, top=0.98, wspace=0.08)
    for ax, occupancy, path, former in (
        (axes[0], before, short_path, None),
        (axes[1], after, long_path, short_path),
    ):
        plotter.draw(ax, {
            "occupancy": occupancy,
            "robot_radius": ROBOT_RADIUS,
            "num_buffer_segments": NUM_BUFFER_SEGMENTS,
            "height": HEIGHT, "width": WIDTH,
            "path": path, "former_path": former,
        })
    # The after-panel legend contains every key needed for both panels.
    axes[0].get_legend().remove()
    _move_legend_to_result_margin(fig)
    fig.savefig(output_dir / "topology_change.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Shortest path before: {short_length:.3f} cell units")
    print(f"Shortest path after:  {long_length:.3f} cell units")
    print(f"One cell added; regions/holes: {before_signature} -> {after_signature}")
    if before_signature == after_signature:
        print("Region and hole counts are unchanged; this layout illustrates a path change.")


if __name__ == "__main__":
    main()
