"""Handcrafted visualization of unit-load access points.

Run from the repository root with ``python -m experiments.visualization``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np

from artifacts.plot import (
    _LARGE_ZONE_LEGEND_HANDLER,
    _draw_continuous_environment,
    _draw_scenario_access_points,
    _free_cell_legend_handle,
    _obstacle_legend_handle,
    RESULT_INTERACTION_POINT_SIZE,
    RESULT_SPECIAL_POINT_SIZE,
)
from visualization.meta import (
    ACCESS_POINT_ARROW_LENGTH,
    ACCESS_POINT_COLOR,
    FONT_SIZE,
    GRID_ZORDER,
    IO_POINT_COLOR,
    LEGEND_POSITION,
    LEGEND_RIGHT,
    SIMPLE_OBSTACLE_COLOR,
    SINGLE_FIGURE_SIZE,
    UNIT_LOAD_COLOR,
    WALL_COLOR,
    WALL_THICKNESS,
)
from scenario.scenario_generation import Scenario


occupancy = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 2, 2],
    [0, 1, 0, 2, 2, 1, 0, 0],
    [0, 2, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0],
    [2, 0, 0, 0, 0, 0, 0, 0],
]

DEMO_IO_POINTS = ((1.4, 0.3), (2.8, 0.3))  # continuous (x, y) coordinates
OUTPUT_PATH = Path("artifacts/visualization/access_points_demo.pdf")

ACCESS_POINT_DIRECTION_VECTORS = {
    "n": (0.0, -1.0),
    "s": (0.0, 1.0),
    "w": (-1.0, 0.0),
    "e": (1.0, 0.0),
}


class _WallLegendHandler(HandlerBase):
    """Render the Walls key slightly smaller than the shared zone swatches."""

    _scale = 0.85

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        del legend, fontsize
        wall_width = self._scale * width
        wall_height = self._scale * 2 * height
        return [
            Rectangle(
                (
                    xdescent + (width - wall_width) / 2,
                    ydescent + (height - wall_height) / 2,
                ),
                wall_width,
                wall_height,
                facecolor=orig_handle.get_facecolor(),
                edgecolor=orig_handle.get_edgecolor(),
                linewidth=orig_handle.get_linewidth(),
                transform=trans,
                clip_on=False,
            )
        ]


def _draw_walls(ax, height, width):
    """Surround the grid with solid walls without treating them as obstacles."""
    thickness = WALL_THICKNESS
    walls = (
        Rectangle((-thickness, -thickness), width + 2 * thickness, thickness),
        Rectangle((-thickness, height), width + 2 * thickness, thickness),
        Rectangle((-thickness, 0), thickness, height),
        Rectangle((width, 0), thickness, height),
    )
    for wall in walls:
        wall.set(facecolor=WALL_COLOR, edgecolor="none", zorder=GRID_ZORDER + 1)
        ax.add_patch(wall)

    ax.set_xlim(-thickness, width + thickness)
    ax.set_ylim(height + thickness, -thickness)


def _draw_interaction_points(ax, interaction_points):
    """Draw freely positioned interaction points in continuous plot coordinates."""
    points = np.asarray(interaction_points, dtype=float)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=RESULT_INTERACTION_POINT_SIZE,
        color=IO_POINT_COLOR,
        edgecolors="none",
        zorder=GRID_ZORDER + 2,
    )


def _draw_access_point_directions(ax, access_points):
    """Draw each access point as an arrow pointing from the unit load to free space."""
    for row, column, direction in access_points:
        dx, dy = ACCESS_POINT_DIRECTION_VECTORS[direction]
        start = (column, row)
        end = (
            column + dx * ACCESS_POINT_ARROW_LENGTH,
            row + dy * ACCESS_POINT_ARROW_LENGTH,
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.9,
                facecolor=ACCESS_POINT_COLOR,
                edgecolor=ACCESS_POINT_COLOR,
                zorder=GRID_ZORDER + 3,
            )
        )


def _legend_access_point_arrow(legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize):
    """Create an arrow-shaped legend key for access-point direction."""
    centre_y = ydescent + height / 2
    return FancyArrowPatch(
        (xdescent, centre_y),
        (xdescent + width, centre_y),
        arrowstyle="-|>",
        mutation_scale=fontsize,
        linewidth=0.9,
        facecolor=ACCESS_POINT_COLOR,
        edgecolor=ACCESS_POINT_COLOR,
    )


def _access_point_direction_legend_handle():
    """Return a proxy artist whose custom handler renders an access-point arrow."""
    return FancyArrowPatch(
        (0, 0),
        (1, 0),
        arrowstyle="-|>",
        facecolor=ACCESS_POINT_COLOR,
        edgecolor=ACCESS_POINT_COLOR,
        label="Access point direction",
    )

def build_access_point_demo():
    """Return a small scenario with unit-load and simple obstacles."""
    height, width = len(occupancy), len(occupancy[0])
    scenario = Scenario(
        rng=np.random.default_rng(0),
        height=height,
        width=width,
        occupancy_rate=0,
        sigma_coarse=1,
        sigma_fine=1,
        noisy_aisles_num=0,
        noisy_aisle_sigma=1,
        interaction_points=set(),
        interaction_points_radius=1,
        simple_obstacle_rate=0,
        agv_radius=0,
    )

    scenario.occupancy = np.array(occupancy, dtype=np.int8)

    scenario.update_access_points()
    return scenario


def plot_access_point_demo():
    """Plot access-point markers and their outward directions."""
    scenario = build_access_point_demo()
    fig, ax = plt.subplots(figsize=SINGLE_FIGURE_SIZE)
    legend_handles = [
        _free_cell_legend_handle(),
        _obstacle_legend_handle(SIMPLE_OBSTACLE_COLOR, "Simple obstacle"),
        _obstacle_legend_handle(UNIT_LOAD_COLOR, "Unit-load obstacle"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=6, label="Walls"),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=np.sqrt(RESULT_SPECIAL_POINT_SIZE),
            markerfacecolor=ACCESS_POINT_COLOR,
            markeredgecolor="none",
            label="Access point",
        ),
        _access_point_direction_legend_handle(),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=np.sqrt(RESULT_INTERACTION_POINT_SIZE),
            markerfacecolor=IO_POINT_COLOR, markeredgecolor="none",
            label="Interaction point",
        ),
    ]

    _draw_continuous_environment(
        ax,
        scenario.occupancy,
        bounds=(0, 0, scenario.width, scenario.height),
    )
    access_points = sorted(scenario.access_points)
    _draw_scenario_access_points(ax, access_points)
    _draw_access_point_directions(ax, access_points)
    _draw_interaction_points(ax, DEMO_IO_POINTS)
    _draw_walls(ax, scenario.height, scenario.width)

    ax.set_axis_off()
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles],
               loc="center left", bbox_to_anchor=LEGEND_POSITION, frameon=False,
               fontsize=FONT_SIZE,
               handler_map={
                   Patch: _LARGE_ZONE_LEGEND_HANDLER,
                   Rectangle: _WallLegendHandler(),
                   FancyArrowPatch: HandlerPatch(patch_func=_legend_access_point_arrow),
               })
    fig.subplots_adjust(left=0.03, right=LEGEND_RIGHT, bottom=0.03, top=0.97)
    return fig, ax


if __name__ == "__main__":
    figure, _ = plot_access_point_demo()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.show()
