"""Generate two initial warehouse layouts for the evaluation-scenarios section.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/codex-layout-matplotlib venv/bin/python -m experiments.evaluation_layout_figures

Shared parameters and interaction-point positions come from the evaluation
configuration. The examples use the evaluation matrix's 40 and 60 percent
occupancy conditions.
PDFs and the parameter record are written directly to
``artifacts/visualization``.
"""

import argparse
import json
from pathlib import Path
import platform
import tomllib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import scipy
import shapely
from shapely.geometry import Point, box
from shapely.ops import unary_union

from algorithms.geometry import continuous_special_points
from artifacts.plot import (
    GRAPH_NODE_COLOR,
    OBSTACLE_LEGEND_MARKER_SIZE,
    POINT_ZORDER,
    VisibilityGraphPlotter,
    _register_result_legend,
)
from experiments.generate_scenarios import _interaction_points_from_warehouse_dimensions
from scenario.scenario_generation import Scenario


EXAMPLES = (
    {"height": 10, "width": 20, "occupancy_rate": 0.40, "robot_radius": 0.45},
    {"height": 20, "width": 40, "occupancy_rate": 0.60, "robot_radius": 0.30},
)
DEFAULT_SEED = 7522


class EvaluationLayoutPlotter(VisibilityGraphPlotter):
    """Decorate the existing roadmap plotter with a robot at its true scale."""

    def draw(self, ax, data):
        # Reuse the environment, buffers, point styles, and standard legend keys.
        super().draw(ax, data)
        ax.add_patch(Circle(
            data["example_robot"]["centre_xy"], radius=data["robot_radius"],
            facecolor=GRAPH_NODE_COLOR, edgecolor="#123d61", linewidth=1.1,
            zorder=POINT_ZORDER + 2,
        ))
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D(
            [], [], linestyle="", marker="o",
            markersize=0.9 * OBSTACLE_LEGEND_MARKER_SIZE,
            color=GRAPH_NODE_COLOR,
        ))
        labels.append("Example robot (to scale)")
        _register_result_legend(ax, handles, labels)


def _gap_example(scenario):
    """Find an existing width-one gap with unit loads on opposite sides."""
    occupancy = scenario.occupancy
    height, width = occupancy.shape
    candidates = []
    for row, column in np.argwhere(occupancy == 0):
        row, column = int(row), int(column)
        orientations = []
        if (0 < column < width - 1
                and occupancy[row, column - 1] == 2
                and occupancy[row, column + 1] == 2):
            orientations.append(("vertical", ((row - 1, column), (row + 1, column))))
        if (0 < row < height - 1
                and occupancy[row - 1, column] == 2
                and occupancy[row + 1, column] == 2):
            orientations.append(("horizontal", ((row, column - 1), (row, column + 1))))
        for orientation, ends in orientations:
            open_ends = sum(
                0 <= r < height and 0 <= c < width and occupancy[r, c] == 0
                for r, c in ends
            )
            if open_ends:
                boundary_distance = min(row, column, height - row - 1, width - column - 1)
                candidates.append((open_ends, boundary_distance, row, column, orientation))

    if not candidates:
        raise RuntimeError("The generated layout contains no suitable unit-load gap.")

    _, _, row, column, orientation = max(candidates)
    centre = Point(column + 0.5, row + 0.5)
    obstacles = unary_union([
        box(c, r, c + 1, r + 1)
        for r, c in np.argwhere(occupancy != 0)
    ])
    clearance = centre.distance(obstacles) - scenario.robot_radius
    expected_clearance = 0.5 - scenario.robot_radius
    if not np.isclose(clearance, expected_clearance):
        raise RuntimeError("The example robot does not have the expected gap clearance.")
    return {
        "row": row,
        "column": column,
        "orientation": orientation,
        "centre_xy": [column + 0.5, row + 0.5],
        "passage_width": 1.0,
        "clearance_on_each_side": float(clearance),
    }


def _plot_layout(scenario, gap, output_stem, seed):
    access_points, interaction_points = continuous_special_points(
        sorted(scenario.access_points), sorted(scenario.interaction_points),
        scenario.robot_radius,
    )
    plotter = EvaluationLayoutPlotter()
    fig, ax = plotter.setup_figure()
    plotter.draw(ax, {
        "occupancy": scenario.occupancy,
        "height": scenario.height,
        "width": scenario.width,
        "robot_radius": scenario.robot_radius,
        "num_buffer_segments": 16,
        "access_points": access_points,
        "interaction_points": interaction_points,
        "nodes": {},
        "edges": [],
        "example_robot": gap,
    })
    # The inherited finalizer moves the shared legend into the right margin.
    plotter.finalize(fig, ax)
    fig.savefig(
        Path(f"{output_stem}.pdf"),
        bbox_inches="tight",
        metadata={
            "Title": f"Evaluation layout {scenario.height}x{scenario.width}",
            "Subject": f"Initial scenario; evaluation seed {seed}; robot shown at true scale",
            "Creator": "experiments.evaluation_layout_figures",
        },
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/evaluation_matrix.toml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visualization"))
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    shared_parameters = {
        key: value for key, value in config["scenario"].items() if key != "stages"
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    metadata = {
        "config": str(args.config),
        "seed": args.seed,
        "snapshot": "initial",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "shapely": shapely.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "examples": [],
    }
    for example in EXAMPLES:
        parameters = dict(shared_parameters, **example)
        parameters["interaction_points"] = _interaction_points_from_warehouse_dimensions(
            example["height"], example["width"],
        )
        scenario = Scenario(
            rng=np.random.default_rng(args.seed), print_progress=False, **parameters,
        )
        if not scenario.generate():
            raise RuntimeError("Initial layout generation failed.")
        occupied = int(np.count_nonzero(scenario.occupancy))
        expected = int(scenario.height * scenario.width * scenario.occupancy_rate)
        if occupied != expected or not scenario.connected():
            raise RuntimeError("The generated layout failed occupancy/connectivity validation.")
        gap = _gap_example(scenario)
        stem = (
            f"layout_{scenario.height}x{scenario.width}"
            f"_occupancy_{100 * scenario.occupancy_rate:.0f}"
            f"_radius_{scenario.robot_radius:.2f}"
        )
        _plot_layout(scenario, gap, args.output_dir / stem, args.seed)
        record = {
            "filename_stem": stem,
            "parameters": dict(parameters, interaction_points=sorted(scenario.interaction_points)),
            "actual_occupancy_rate": occupied / scenario.occupancy.size,
            "occupied_cells": occupied,
            "unit_load_cells": int(np.count_nonzero(scenario.occupancy == 2)),
            "simple_obstacle_cells": int(np.count_nonzero(scenario.occupancy == 1)),
            "free_cells_connected": scenario.connected(),
            "access_points": sorted(scenario.access_points),
            "example_robot": gap,
            "occupancy": scenario.occupancy.tolist(),
        }
        metadata["examples"].append(record)
        print(json.dumps({key: record[key] for key in (
            "filename_stem", "actual_occupancy_rate", "occupied_cells",
            "unit_load_cells", "simple_obstacle_cells", "example_robot",
        )}))
    (args.output_dir / "layout_parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved figures and layout parameters to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
