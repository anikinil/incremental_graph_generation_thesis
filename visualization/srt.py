"""Illustration of a sampling-based roadmap of trees (SRT).

Run from the repository root with::

    python -m visualization.srt roadmap
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MatplotlibPath
import numpy as np
from networkx.utils import UnionFind
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from visualization.meta import (
    SIMPLE_OBSTACLE_COLOR,
    SINGLE_FIGURE_SIZE,
    SPECIAL_POINT_EDGE_COLOR,
    SRT_CONNECTION_COLOR,
    SRT_TREE_COLORS,
)


WORLD_WIDTH = 10
WORLD_HEIGHT = 7
TREE_NODE_COUNT = 30
TREE_RADIUS = 1.3
TREE_STEERING_STEP = 0.28
CLOSE_TREE_CANDIDATES = 2
RANDOM_TREE_CANDIDATES = 1
NODE_PAIR_CANDIDATES = 5
OUTPUT_DIRECTORY = Path("artifacts/visualization")

# These collision-free roots are the SRT milestones.  The local RRT-like trees
# grown from them represent roadmap vertices; the black lines are successful
# connections between separate trees.
TREE_ROOTS = (
    (0.75, 1.10),
    (1.25, 5.90),
    (0.75, 4.55),
    (3.15, 2.80),
    (3.40, 0.75),
    # (5.35, 0.60),
    (7.05, 0.75),
    (5.77, 3.22),
    (8.70, 1.45),
    (9.20, 3.15),
    (8.75, 5.90),
    (6.20, 6.20),
    # (5.10, 6.20),
    (3.70, 6.20),
)


def _obstacles():
    """Return a varied, deterministic set of continuous obstacles."""
    return unary_union(
        [
            Polygon(((0.95, 1.10), (2.55, 1.20), (2.90, 2.15), (1.35, 2.60))),
            Point(5.10, 1.70).buffer(0.66, quad_segs=12),
            Polygon(((3.20, 3.90), (5.25, 4.20), (4.60, 5.60), (3.05, 5.25))),
            box(6.5, 3.65, 8.10, 6.30).buffer(0.30, quad_segs=12),
        ]
    )


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return geometry.geoms


def _add_geometry(ax, geometry, color, zorder):
    """Draw a Shapely polygonal geometry as a vector patch collection."""
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


def _sample_tree_target(root, free_space, rng):
    """Sample a collision-free RRT target within the local tree's neighbourhood."""
    for _ in range(100):
        direction = rng.uniform(0, 2 * np.pi)
        radius = TREE_RADIUS * np.sqrt(rng.uniform())
        target = root + radius * np.array((np.cos(direction), np.sin(direction)))
        if free_space.covers(Point(target)):
            return target
    raise RuntimeError("Could not sample a free steering target.")


def _grow_tree(root, free_space, rng):
    """Grow one bounded local RRT by steering its nearest node toward random targets."""
    root = np.asarray(root, dtype=float)
    nodes = [root]
    edges = []
    attempts = 0

    while len(nodes) < TREE_NODE_COUNT and attempts < 2_000:
        attempts += 1
        target = _sample_tree_target(root, free_space, rng)
        parent_index = np.argmin([np.linalg.norm(target - node) for node in nodes])
        direction = target - nodes[parent_index]
        distance = np.linalg.norm(direction)
        if distance == 0:
            continue
        child = nodes[parent_index] + min(TREE_STEERING_STEP, distance) * direction / distance

        if np.linalg.norm(child - root) > TREE_RADIUS:
            continue
        if not free_space.covers(LineString((nodes[parent_index], child))):
            continue

        nodes.append(child)
        edges.append((nodes[parent_index], child))

    if len(nodes) != TREE_NODE_COUNT:
        raise RuntimeError("Could not grow the requested local tree.")
    nodes = np.asarray(nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "representative": np.mean(nodes, axis=0),
    }


def _candidate_tree_pairs(trees):
    """Select nearest and random tree representatives, as in SRT's connection phase."""
    rng = np.random.default_rng(2)
    representatives = np.asarray([tree["representative"] for tree in trees])
    pairs = set()

    for index, representative in enumerate(representatives):
        distances = np.linalg.norm(representatives - representative, axis=1)
        neighbours = np.argsort(distances)
        closest = [int(other) for other in neighbours if other != index][:CLOSE_TREE_CANDIDATES]
        remaining = [other for other in range(len(trees)) if other not in {index, *closest}]
        random_count = min(RANDOM_TREE_CANDIDATES, len(remaining))
        random_trees = rng.choice(remaining, size=random_count, replace=False)
        for other in (*closest, *random_trees):
            pairs.add(tuple(sorted((index, int(other)))))
    return sorted(pairs)


def _candidate_node_pairs(left_nodes, right_nodes):
    """Return the closest node pairs considered by the SRT connection planner."""
    pairs = sorted(
        (
            (float(np.linalg.norm(left - right)), left, right)
            for left in left_nodes
            for right in right_nodes
        ),
        key=lambda candidate: candidate[0],
    )
    return pairs[:NODE_PAIR_CANDIDATES]


def _connect_trees(trees, free_space):
    """Connect selected tree pairs through collision-free node-pair bridges."""
    components = UnionFind(range(len(trees)))
    connections = []

    for left_index, right_index in _candidate_tree_pairs(trees):
        if components[left_index] == components[right_index]:
            continue
        for _, left, right in _candidate_node_pairs(
            trees[left_index]["nodes"],
            trees[right_index]["nodes"],
        ):
            if free_space.covers(LineString((left, right))):
                components.union(left_index, right_index)
                connections.append((left, right))
                break
    return connections


def build_srt_demo():
    """Build local trees and their sparse roadmap connections."""
    world = box(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
    obstacles = _obstacles()
    free_space = world.difference(obstacles)
    rng = np.random.default_rng(23)
    trees = [_grow_tree(root, free_space, rng) for root in TREE_ROOTS]
    return {
        "obstacles": obstacles,
        "trees": trees,
        "connections": _connect_trees(trees, free_space),
    }


def _draw_environment(ax, demo):
    _add_geometry(ax, demo["obstacles"], SIMPLE_OBSTACLE_COLOR, zorder=1)
    ax.add_patch(
        Rectangle(
            (0, 0),
            WORLD_WIDTH,
            WORLD_HEIGHT,
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


def _draw_trees(ax, trees):
    for index, tree in enumerate(trees):
        color = SRT_TREE_COLORS[index % len(SRT_TREE_COLORS)]
        ax.add_collection(
            LineCollection(tree["edges"], colors=[color], linewidths=0.9, capstyle="round", zorder=10)
        )
        ax.scatter(
            tree["nodes"][:, 0],
            tree["nodes"][:, 1],
            s=18,
            color=color,
            edgecolors=SPECIAL_POINT_EDGE_COLOR,
            linewidths=0.35,
            zorder=15,
        )
        ax.scatter(
            *tree["representative"],
            s=70,
            color=color,
            edgecolors=SPECIAL_POINT_EDGE_COLOR,
            linewidths=0.9,
            zorder=16,
        )


def plot_srt_roadmap():
    """Plot local trees as milestones joined into an SRT roadmap."""
    demo = build_srt_demo()
    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)

    _draw_environment(ax, demo)
    _draw_trees(ax, demo["trees"])
    ax.add_collection(
        LineCollection(
            demo["connections"],
            colors=[SRT_CONNECTION_COLOR],
            linewidths=1.7,
            linestyles="dashed",
            capstyle="round",
            zorder=20,
        )
    )

    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.03, top=0.98)
    return fig, ax


VISUALIZATIONS = {
    "roadmap": (plot_srt_roadmap, OUTPUT_DIRECTORY / "srt_demo.pdf"),
}


def main(arguments=None):
    """Render the selected SRT visualization."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1 or arguments[0] not in VISUALIZATIONS:
        available = ", ".join(VISUALIZATIONS)
        raise SystemExit(f"Choose one SRT visualization: {available}")

    plot, output_path = VISUALIZATIONS[arguments[0]]
    figure, _ = plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
