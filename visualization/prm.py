"""Handcrafted probabilistic-roadmap visualizations.

Run from the repository root with::

    python -m visualization.prm allow-cycles
    python -m visualization.prm avoid-cycles
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MatplotlibPath
import networkx as nx
import numpy as np
from networkx.utils import UnionFind
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from visualization.meta import (
    GRAPH_NODE_SIZE_SCALE,
    PRM_EDGE_COLOR,
    PRM_GOAL_COLOR,
    PRM_PATH_COLOR,
    PRM_SAMPLE_COLOR,
    PRM_START_COLOR,
    PRM_TERMINAL_LABEL_COLOR,
    PRM_TERMINAL_LABEL_SIZE,
    SIMPLE_OBSTACLE_COLOR,
    SINGLE_FIGURE_SIZE,
    SPECIAL_POINT_EDGE_COLOR,
)


WORLD_WIDTH = 10
WORLD_HEIGHT = 7
SAMPLE_COUNT = 400
NEIGHBOURS_PER_SAMPLE = 7
QUERY_CONNECTIONS = 6
MAX_CONNECTION_DISTANCE = 1
START_CONFIGURATION = (0.55, 6.25)
GOAL_CONFIGURATION = (9.25, 0.65)
OUTPUT_DIRECTORY = Path("artifacts/visualization")


def _obstacles():
    """Return a varied, deterministic set of general continuous obstacles."""
    return unary_union(
        [
            Point(1.9, 1.9).buffer(0.75, quad_segs=12),
            box(3.35, 3.65, 5.35, 5.55).buffer(0.25, quad_segs=12),
            Polygon(((5.8, 1.0), (7.45, 1.35), (7.2, 2.75), (5.55, 2.45))),
            Point(8.05, 4.95).buffer(0.78, quad_segs=12),
        ]
    )


def _distance(left, right):
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def _sample_free_configurations(free_space):
    """Generate deterministic, collision-free configurations in the free space."""
    rng = np.random.default_rng(42)
    samples = []
    while len(samples) < SAMPLE_COUNT:
        candidate = (rng.uniform(0, WORLD_WIDTH), rng.uniform(0, WORLD_HEIGHT))
        if free_space.contains(Point(candidate)):
            samples.append(candidate)
    return samples


def _connect_visible_neighbours(
    graph,
    positions,
    free_space,
    node,
    candidates,
    limit,
    components=None,
):
    """Connect one node to its nearest visible neighbours with a straight local planner.

    ``components`` tracks connected components when cycle-creating edges should
    be omitted. Passing ``None`` permits all visible neighbour connections.
    """
    nearest = sorted(
        (other for other in candidates if other != node),
        key=lambda other: _distance(positions[node], positions[other]),
    )
    connections = 0
    for other in nearest:
        if _distance(positions[node], positions[other]) > MAX_CONNECTION_DISTANCE:
            break

        if components is not None and components[node] == components[other]:
            continue

        edge = LineString((positions[node], positions[other]))
        if free_space.covers(edge):
            graph.add_edge(node, other, weight=edge.length)
            if components is not None:
                components.union(node, other)
            connections += 1
            if connections == limit:
                return


def build_prm_demo(*, avoid_cycles=False):
    """Build a roadmap and start-to-goal query for the selected construction mode."""
    world = box(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
    obstacles = _obstacles()
    free_space = world.difference(obstacles)
    samples = _sample_free_configurations(free_space)
    positions = dict(enumerate(samples))

    roadmap = nx.Graph()
    roadmap.add_nodes_from(positions)
    roadmap_components = UnionFind(positions) if avoid_cycles else None
    for node in positions:
        _connect_visible_neighbours(
            roadmap,
            positions,
            free_space,
            node,
            positions,
            NEIGHBOURS_PER_SAMPLE,
            roadmap_components,
        )

    query = roadmap.copy()
    positions.update(start=START_CONFIGURATION, goal=GOAL_CONFIGURATION)
    query.add_nodes_from(("start", "goal"))
    query_components = UnionFind(positions) if avoid_cycles else None
    if query_components is not None:
        for left, right in roadmap.edges:
            query_components.union(left, right)

    for node in ("start", "goal"):
        _connect_visible_neighbours(
            query,
            positions,
            free_space,
            node,
            range(len(samples)),
            QUERY_CONNECTIONS,
            query_components,
        )

    path = nx.shortest_path(query, "start", "goal", weight="weight")
    return {
        "obstacles": obstacles,
        "samples": samples,
        "positions": positions,
        "roadmap_edges": list(roadmap.edges),
        "query_edges": list(query.edges),
        "path": path,
    }


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return geometry.geoms


def _add_geometry(ax, geometry, color, zorder):
    """Draw a polygonal Shapely geometry as vector patches, including holes."""
    patches = []
    for polygon in _polygons(geometry):
        vertices = []
        codes = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords)
            vertices.extend(coordinates)
            codes.extend(
                [MatplotlibPath.MOVETO]
                + [MatplotlibPath.LINETO] * (len(coordinates) - 2)
                + [MatplotlibPath.CLOSEPOLY]
            )
        patches.append(PathPatch(MatplotlibPath(vertices, codes)))
    ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor="none", zorder=zorder))


def _segments(edges, positions):
    return [(positions[left], positions[right]) for left, right in edges]


def _draw_environment(ax, demo):
    _add_geometry(ax, demo["obstacles"], SIMPLE_OBSTACLE_COLOR, zorder=1)
    ax.add_patch(
        Rectangle(
            (0, 0), WORLD_WIDTH, WORLD_HEIGHT,
            fill=False,
            edgecolor=SPECIAL_POINT_EDGE_COLOR,
            linewidth=1.0,
            zorder=30,
        )
    )
    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(WORLD_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_samples(ax, samples):
    points = np.asarray(samples)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=26 * GRAPH_NODE_SIZE_SCALE,
        color=PRM_SAMPLE_COLOR,
        edgecolors="none",
        zorder=25,
    )


def _draw_edges(ax, edges, positions, *, color=PRM_EDGE_COLOR, linewidth=0.7, zorder=10):
    ax.add_collection(
        LineCollection(
            _segments(edges, positions),
            colors=[color],
            linewidths=linewidth,
            capstyle="round",
            zorder=zorder,
        )
    )


def _draw_query(ax, demo):
    positions = demo["positions"]
    _draw_edges(ax, demo["query_edges"], positions)
    _draw_edges(
        ax,
        list(zip(demo["path"], demo["path"][1:])),
        positions,
        color=PRM_PATH_COLOR,
        linewidth=2.2,
        zorder=26,
    )
    for name, label, color in (("start", "S", PRM_START_COLOR), ("goal", "G", PRM_GOAL_COLOR)):
        ax.scatter(
            *positions[name],
            s=220,
            color=color,
            edgecolors="none",
            zorder=27,
        )
        ax.annotate(
            label,
            xy=positions[name],
            xytext=(0, -1.5),
            textcoords="offset points",
            color=PRM_TERMINAL_LABEL_COLOR,
            fontsize=PRM_TERMINAL_LABEL_SIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=28,
        )


def plot_prm_roadmap(*, avoid_cycles=False):
    """Plot a start-to-goal query through the selected PRM construction mode."""
    demo = build_prm_demo(avoid_cycles=avoid_cycles)
    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)

    _draw_environment(ax, demo)
    _draw_samples(ax, demo["samples"])
    _draw_query(ax, demo)

    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.98)
    return fig, ax


CONSTRUCTION_MODES = {
    "allow-cycles": False,
    "avoid-cycles": True,
}


def main(arguments=None):
    """Render the selected PRM visualization."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1 or arguments[0] not in CONSTRUCTION_MODES:
        modes = ", ".join(CONSTRUCTION_MODES)
        raise SystemExit(f"Choose a PRM construction mode: {modes}")

    mode = arguments[0]
    figure, _ = plot_prm_roadmap(avoid_cycles=CONSTRUCTION_MODES[mode])
    output_path = OUTPUT_DIRECTORY / f"prm_demo_{mode}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
