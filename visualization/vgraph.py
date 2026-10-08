"""Handcrafted static visibility-graph visualizations.

Run a named visualization from the repository root, for example::

    python -m visualization.vgraph buffer-space

Add future VGraph illustrations to ``VISUALIZATIONS`` so that they receive the
same command-line entry point and output handling.
"""

from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from algorithms.geometry import buffered_obstacle_space, navigation_space
from artifacts.plot import (
    _area_legend_handler_map,
    _buffer_zone_legend_handle,
    _draw_continuous_environment,
    _free_cell_legend_handle,
    _obstacle_legend_handle,
)
from visualization.meta import (
    FONT_SIZE,
    GRAPH_NODE_SIZE_SCALE,
    GRAPH_VERTEX_COLOR,
    GRID_ZORDER,
    LEGEND_POSITION,
    LEGEND_RIGHT,
    OFFSET_VEC_COLOR,
    RADIUS_OFFSET_COLOR,
    SIMPLE_OBSTACLE_COLOR,
    SPECIAL_POINT_EDGE_COLOR,
    SINGLE_FIGURE_SIZE,
    VISIBILITY_EDGE_COLOR,
)


BUFFER_SPACE_OCCUPANCY = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ],
    dtype=np.int8,
)
BUFFER_SPACE_AGV_RADIUS = 0.4
# Static VGraph uses square buffer corners when ``num_buffer_segments`` is 0.
STATIC_VGRAPH_BUFFER_SEGMENTS = 0
AGV_RADIUS_ARROW_COLOR = RADIUS_OFFSET_COLOR
ORIGINAL_VERTEX_COLOR = to_rgba(OFFSET_VEC_COLOR, alpha=1)
ORIGINAL_VERTEX_ZORDER = GRID_ZORDER + 2
BUFFERED_VERTEX_ZORDER = GRID_ZORDER + 3

OUTPUT_DIRECTORY = Path("artifacts/visualization")


def _vertex_offset_pairs(occupancy):
    """Pair each static-VGraph vertex with its nearest unbuffered obstacle corner."""
    height, width = occupancy.shape
    rows, columns = np.nonzero(occupancy)
    obstacle_space = unary_union(
        [box(column, row, column + 1, row + 1) for row, column in zip(rows, columns)]
    )
    obstacle_polygons = (
        [obstacle_space] if obstacle_space.geom_type == "Polygon" else obstacle_space.geoms
    )
    original_vertices = {
        tuple(corner)
        for polygon in obstacle_polygons
        for corner in list(polygon.exterior.coords)[:-1]
    }
    buffer_space = buffered_obstacle_space(
        occupancy,
        BUFFER_SPACE_AGV_RADIUS,
        STATIC_VGRAPH_BUFFER_SEGMENTS,
    )
    buffer_polygons = (
        [buffer_space] if buffer_space.geom_type == "Polygon" else buffer_space.geoms
    )

    pairs = []
    for polygon in buffer_polygons:
        corners = list(polygon.exterior.coords)[:-1]
        for previous, buffered, following in zip(
            corners[-1:] + corners[:-1], corners, corners[1:] + corners[:1]
        ):
            turn = (
                (buffered[0] - previous[0]) * (following[1] - buffered[1])
                - (buffered[1] - previous[1]) * (following[0] - buffered[0])
            )
            touches_boundary = (
                np.isclose(buffered[0], 0)
                or np.isclose(buffered[0], width)
                or np.isclose(buffered[1], 0)
                or np.isclose(buffered[1], height)
            )
            if turn < 0 and not touches_boundary:
                original = min(
                    original_vertices,
                    key=lambda vertex: (vertex[0] - buffered[0]) ** 2
                    + (vertex[1] - buffered[1]) ** 2,
                )
                pairs.append((original, tuple(buffered)))
    return pairs


def _visibility_edges(occupancy, vertices):
    """Return the static VGraph edges between mutually visible vertices."""
    free_space, _ = navigation_space(
        occupancy,
        BUFFER_SPACE_AGV_RADIUS,
        STATIC_VGRAPH_BUFFER_SEGMENTS,
    )
    return [
        (left, right)
        for left, right in combinations(vertices, 2)
        if free_space.covers(LineString([left, right]))
    ]


def _draw_dashed_arrow(ax, start, end, *, color, zorder):
    """Draw a dashed shaft with an undashed, filled arrowhead."""
    color = to_rgba(color, alpha=1)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return

    arrowhead_length = min(0.14, length / 2)
    arrowhead_start = end - direction / length * arrowhead_length
    ax.add_patch(
        FancyArrowPatch(
            start,
            arrowhead_start,
            arrowstyle="-",
            linewidth=1.5,
            linestyle="dashed",
            color=color,
            zorder=zorder,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            arrowhead_start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.5,
            linestyle="solid",
            facecolor=color,
            edgecolor=color,
            zorder=zorder,
        )
    )


def _random_buffer_offsets(occupancy):
    """Return upward and leftward radius offsets for one reproducibly chosen obstacle."""
    height, width = occupancy.shape
    candidates = []
    for row, column in zip(*np.nonzero(occupancy)):
        if row == 0 or column == 0 or occupancy[row - 1, column] or occupancy[row, column - 1]:
            continue

        centre = np.array((column + 0.5, row + 0.5))
        offsets = []
        for direction in (np.array((0, -1)), np.array((-1, 0))):
            start = centre + direction / 2
            end = start + direction * BUFFER_SPACE_AGV_RADIUS
            offsets.append((start, end))
        candidates.append(offsets)

    return candidates[np.random.default_rng(0).integers(len(candidates))]


class _DashedArrowLegendHandler(HandlerBase):
    """Render a legend offset vector as a dashed line ending in a solid arrow."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, transform):
        centre_y = ydescent + height / 2
        arrowhead_start = xdescent + width * 0.72
        color = to_rgba(orig_handle.get_facecolor(), alpha=1)
        shaft = Line2D(
            [xdescent, arrowhead_start],
            [centre_y, centre_y],
            color=color,
            linestyle="dashed",
            linewidth=1.5,
            transform=transform,
        )
        arrowhead = FancyArrowPatch(
            (arrowhead_start, centre_y),
            (xdescent + width, centre_y),
            arrowstyle="-|>",
            mutation_scale=fontsize,
            linewidth=1.5,
            linestyle="solid",
            facecolor=color,
            edgecolor=color,
            transform=transform,
        )
        return [shaft, arrowhead]


def _arrow_legend_handle(label, color):
    """Return a proxy artist for the dashed-arrow legend renderer."""
    color = to_rgba(color, alpha=1)
    return FancyArrowPatch(
        (0, 0), (1, 0), arrowstyle="-|>", facecolor=color, edgecolor=color, label=label
    )


def _vertex_legend_handle(label, facecolor, marker_area):
    """Return the shared legend marker used for VGraph vertices."""
    return Line2D(
        [], [], marker="o", linestyle="", markersize=np.sqrt(marker_area),
        markerfacecolor=facecolor, markeredgecolor="none",
        label=label,
    )


def plot_buffer_space():
    """Plot obstacle space expanded by the static VGraph robot-radius buffer."""
    occupancy = BUFFER_SPACE_OCCUPANCY
    height, width = occupancy.shape
    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)

    _draw_continuous_environment(
        ax,
        occupancy,
        bounds=(0, 0, width, height),
        agv_radius=BUFFER_SPACE_AGV_RADIUS,
        num_buffer_segments=STATIC_VGRAPH_BUFFER_SEGMENTS,
    )
    ax.tick_params(labelsize=FONT_SIZE)

    vertex_offset_pairs = _vertex_offset_pairs(occupancy)
    original_vertices, buffered_vertices = zip(*vertex_offset_pairs)
    visibility_edges = _visibility_edges(occupancy, buffered_vertices)
    ax.add_collection(
        LineCollection(
            visibility_edges,
            colors=[VISIBILITY_EDGE_COLOR],
            linewidths=1,
            capstyle="round",
            zorder=10,
        )
    )
    for original, buffered in vertex_offset_pairs:
        _draw_dashed_arrow(
            ax,
            original,
            buffered,
            color=OFFSET_VEC_COLOR,
            zorder=20,
        )

    for radius_start, radius_end in _random_buffer_offsets(occupancy):
        _draw_dashed_arrow(
            ax,
            radius_start,
            radius_end,
            color=AGV_RADIUS_ARROW_COLOR,
            zorder=GRID_ZORDER + 1,
        )
    ax.scatter(
        *zip(*original_vertices),
        s=30 * GRAPH_NODE_SIZE_SCALE,
        color=ORIGINAL_VERTEX_COLOR,
        edgecolors="none",
        zorder=ORIGINAL_VERTEX_ZORDER,
    )
    ax.scatter(
        *zip(*buffered_vertices),
        s=32 * GRAPH_NODE_SIZE_SCALE,
        color=GRAPH_VERTEX_COLOR,
        edgecolors="none",
        zorder=BUFFERED_VERTEX_ZORDER,
    )

    legend_handles = [
        _free_cell_legend_handle(),
        _obstacle_legend_handle(SIMPLE_OBSTACLE_COLOR, "Obstacle"),
        _buffer_zone_legend_handle(),
        _arrow_legend_handle("Obstacle buffer offset = robot radius", AGV_RADIUS_ARROW_COLOR),
        _vertex_legend_handle(
            "Original obstacle node",
            ORIGINAL_VERTEX_COLOR,
            30 * GRAPH_NODE_SIZE_SCALE,
        ),
        _arrow_legend_handle("Node offset", OFFSET_VEC_COLOR),
        _vertex_legend_handle(
            "Roadmap nodes",
            GRAPH_VERTEX_COLOR,
            32 * GRAPH_NODE_SIZE_SCALE,
        ),
        Line2D(
            [],
            [],
            color=VISIBILITY_EDGE_COLOR,
            linewidth=0.8,
            label="Roadmap edges",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=LEGEND_POSITION,
        frameon=False,
        fontsize=FONT_SIZE,
        handler_map={
            **_area_legend_handler_map(),
            FancyArrowPatch: _DashedArrowLegendHandler(),
        },
    )
    fig.subplots_adjust(left=0.03, right=LEGEND_RIGHT, bottom=0.03, top=0.97)
    return fig, ax


VISUALIZATIONS = {
    "buffer-space": (
        plot_buffer_space,
        OUTPUT_DIRECTORY / "vgraph_buffer_space.pdf",
    ),
}


def main(arguments=None):
    """Render the VGraph visualization selected by its command name."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1 or arguments[0] not in VISUALIZATIONS:
        available = ", ".join(VISUALIZATIONS)
        raise SystemExit(f"Choose one VGraph visualization: {available}")

    plot, output_path = VISUALIZATIONS[arguments[0]]
    figure, _ = plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
