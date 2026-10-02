"""Handcrafted visualization of unit-load access points.

Run from the repository root with ``python -m experiments.visualization``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np

from artifacts.plot import (
    ACCESS_POINT_COLOR,
    GRID_ZORDER,
    IO_POINT_COLOR,
    SIMPLE_OBSTACLE_COLOR,
    SPECIAL_POINT_EDGE_COLOR,
    UNIT_LOAD_COLOR,
    _draw_continuous_environment,
    _draw_scenario_access_points,
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

WALL_COLOR = (0.3, 0.6, 0.9, 1)
WALL_THICKNESS = 0.25
ACCESS_POINT_ARROW_LENGTH = 0.42
DEMO_IO_POINTS = ((3.4, 5.7), (4.8, 5.7))  # continuous (x, y) coordinates
OUTPUT_PATH = Path("artifacts/visualization/access_points_demo.pdf")

ACCESS_POINT_DIRECTION_VECTORS = {
    "n": (0.0, -1.0),
    "s": (0.0, 1.0),
    "w": (-1.0, 0.0),
    "e": (1.0, 0.0),
}


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


def _draw_io_points(ax, io_points):
    """Draw freely positioned interaction points in continuous plot coordinates."""
    points = np.asarray(io_points, dtype=float)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=52,
        color=IO_POINT_COLOR,
        edgecolors=SPECIAL_POINT_EDGE_COLOR,
        linewidths=0.8,
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
                edgecolor=SPECIAL_POINT_EDGE_COLOR,
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
        edgecolor=SPECIAL_POINT_EDGE_COLOR,
    )


def _access_point_direction_legend_handle():
    """Return a proxy artist whose custom handler renders an access-point arrow."""
    return FancyArrowPatch(
        (0, 0),
        (1, 0),
        arrowstyle="-|>",
        facecolor=ACCESS_POINT_COLOR,
        edgecolor=SPECIAL_POINT_EDGE_COLOR,
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
        io_points=set(),
        io_points_radius=1,
        simple_obstacle_rate=0,
        agv_radius=0,
    )

    scenario.occupancy = np.array(occupancy, dtype=np.int8)

    scenario.update_access_points()
    return scenario


def plot_access_point_demo():
    """Plot access-point markers and their outward directions."""
    scenario = build_access_point_demo()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    legend_handles = [
        Patch(facecolor=WALL_COLOR, edgecolor="none", label="Walls"),
        Patch(facecolor=SIMPLE_OBSTACLE_COLOR, edgecolor="none", label="Simple obstacle"),
        Patch(facecolor=UNIT_LOAD_COLOR, edgecolor="none", label="Unit-load obstacle"),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=7,
            markerfacecolor=ACCESS_POINT_COLOR,
            markeredgecolor=SPECIAL_POINT_EDGE_COLOR,
            label="Access point",
        ),
        _access_point_direction_legend_handle(),
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=7,
            markerfacecolor=IO_POINT_COLOR, markeredgecolor=SPECIAL_POINT_EDGE_COLOR,
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
    _draw_io_points(ax, DEMO_IO_POINTS)
    _draw_walls(ax, scenario.height, scenario.width)

    ax.set_axis_off()
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles],
               loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=False,
               handler_map={FancyArrowPatch: HandlerPatch(patch_func=_legend_access_point_arrow)})
    fig.subplots_adjust(left=0.03, right=0.76, bottom=0.03, top=0.97)
    return fig, ax


if __name__ == "__main__":
    figure, _ = plot_access_point_demo()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.show()
