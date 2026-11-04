"""Show occupancy and robot-radius effects as annotated timing heatmaps.

Run from the repository root:
    MPLCONFIGDIR=/tmp/thesis-matplotlib venv/bin/python -m experiments.plot_computation_time_parameter_effects

Condition means exclude initialization. Compute the percentage change in
these means at each fixed setting of the other parameter, then average the
two percentage changes. Export a vector PDF, summary and component CSVs.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from experiments.plot_computation_time_comparison import (
    PERCENT_RESULTS,
    SINGLE_RESULTS,
    configure_style,
)
from experiments.plot_computation_time_scaling import (
    CONDITIONS,
    OCCUPANCIES,
    RADII,
    WAREHOUSES,
    build_summaries,
)
from experiments.plot_stage_timings import ALGORITHM_FAMILIES, ALGORITHMS, REPOSITORY_ROOT


OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/visualization/computation_time_parameter_effects"
STEM = "computation_time_parameter_effects"
EFFECTS = ("occupancy", "robot_radius")
REGIMES = ("single_obstacle", "five_percent")
COLOR_LIMIT = 50
ALGORITHM_LABELS = {
    algorithm: label
    for static, dynamic, static_label, dynamic_label, _ in ALGORITHM_FAMILIES
    for algorithm, label in ((static, static_label), (dynamic, dynamic_label))
}


def parameter_comparisons(effect):
    """Compare one parameter at each fixed setting of the other parameter."""
    if effect == "occupancy":
        return [("robot_radius", radius, CONDITIONS[40, radius], CONDITIONS[60, radius])
                for radius in RADII]
    return [("occupancy_percent", occupancy, CONDITIONS[occupancy, 0.3], CONDITIONS[occupancy, 0.45])
            for occupancy in OCCUPANCIES]


def build_effects(summaries):
    """Average two changes in condition means, preserving missing results."""
    cells, components = [], []
    for regime in REGIMES:
        timings = {
            (record["warehouse"], record["condition"], record["algorithm"]): record
            for record in summaries[regime]
        }
        source_csv = summaries[regime][0]["source_csv"]
        for effect in EFFECTS:
            for algorithm in ALGORITHMS:
                for warehouse in WAREHOUSES:
                    missing = (warehouse, algorithm) == ("50x100", "dynamic_vgraph")
                    cell = {
                        "update_regime": regime,
                        "parameter_effect": effect,
                        "algorithm": algorithm,
                        "warehouse": warehouse,
                        "source_csv": source_csv,
                        "status": "not_evaluated" if missing else "evaluated",
                        "mean_change_percent": "",
                        "component_count": 0 if missing else 2,
                        "seeds_per_condition": 10,
                        "updates_per_seed": 7,
                        "color_saturated": "",
                    }
                    changes = []
                    if not missing:
                        for fixed_parameter, fixed_value, baseline, changed in parameter_comparisons(effect):
                            low = timings[warehouse, baseline, algorithm]["seconds_mean"]
                            high = timings[warehouse, changed, algorithm]["seconds_mean"]
                            change = 100 * (high / low - 1)
                            changes.append(change)
                            components.append({
                                "update_regime": regime,
                                "parameter_effect": effect,
                                "algorithm": algorithm,
                                "warehouse": warehouse,
                                "source_csv": source_csv,
                                "fixed_parameter": fixed_parameter,
                                "fixed_value": fixed_value,
                                "baseline_condition": baseline,
                                "changed_condition": changed,
                                "baseline_seconds_mean": low,
                                "changed_seconds_mean": high,
                                "change_percent": change,
                            })
                        cell["mean_change_percent"] = mean(changes)
                        cell["color_saturated"] = abs(cell["mean_change_percent"]) > COLOR_LIMIT
                    cells.append(cell)
    return cells, components


def plot_effects(cells, output_dir):
    configure_style()
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")
    norm = Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, clip=True)
    cell_lookup = {
        (cell["parameter_effect"], cell["update_regime"], cell["algorithm"], cell["warehouse"]): cell
        for cell in cells
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.18, top=0.91,
                        wspace=0.11, hspace=0.47)
    try:
        for column, (regime, title) in enumerate((
            ("single_obstacle", "Single-obstacle changes"),
            ("five_percent", "5% obstacle changes"),
        )):
            bounds = axes[0, column].get_position()
            fig.text((bounds.x0 + bounds.x1) / 2, 0.965, title,
                     ha="center", va="top", fontsize=10, fontweight="bold")
            for row, effect in enumerate(EFFECTS):
                axis = axes[row, column]
                values = np.full((len(ALGORITHMS), len(WAREHOUSES)), np.nan)
                for y, algorithm in enumerate(ALGORITHMS):
                    for x, warehouse in enumerate(WAREHOUSES):
                        cell = cell_lookup[effect, regime, algorithm, warehouse]
                        if cell["status"] == "evaluated":
                            values[y, x] = cell["mean_change_percent"]
                heatmap = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                                      aspect="auto", interpolation="nearest")
                for y in range(len(ALGORITHMS)):
                    for x in range(len(WAREHOUSES)):
                        value = values[y, x]
                        if np.isnan(value):
                            label, text_color = "N/A", "0.30"
                        else:
                            label = "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
                            red, green, blue, _ = cmap(norm(value))
                            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                            text_color = "white" if luminance < 0.53 else "0.12"
                        axis.text(x, y, label, ha="center", va="center",
                                  color=text_color, fontsize=7.3)
                axis.set_xticks(range(len(WAREHOUSES)), [
                    "$" + warehouse.replace("x", r"\times") + "$" for warehouse in WAREHOUSES
                ])
                axis.set_yticks(range(len(ALGORITHMS)), [ALGORITHM_LABELS[a] for a in ALGORITHMS])
                axis.tick_params(axis="x", labelsize=7.1, labelbottom=True, length=0, pad=5)
                axis.tick_params(axis="y", labelsize=7.5, labelleft=column == 0, length=0, pad=5)
                axis.set_xticks(np.arange(-0.5, len(WAREHOUSES)), minor=True)
                axis.set_yticks(np.arange(-0.5, len(ALGORITHMS)), minor=True)
                axis.grid(which="minor", color="white", linewidth=0.7)
                axis.tick_params(which="minor", bottom=False, left=False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.set_xlabel("Warehouse size (units)", labelpad=5, fontsize=8.3)
        for row, title in enumerate((
            r"Occupancy: 40% $\rightarrow$ 60%",
            r"Robot radius: 0.3 $\rightarrow$ 0.45",
        )):
            bounds = axes[row, 0].get_position()
            fig.text(0.035, (bounds.y0 + bounds.y1) / 2, title, rotation=90,
                     ha="center", va="center", fontsize=9.5, fontweight="bold")
        evaluated = [cell["mean_change_percent"] for cell in cells if cell["status"] == "evaluated"]
        below, above = min(evaluated) < -COLOR_LIMIT, max(evaluated) > COLOR_LIMIT
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        color_axis = fig.add_axes([0.30, 0.075, 0.655, 0.025])
        colorbar = fig.colorbar(heatmap, cax=color_axis, orientation="horizontal", extend=extend)
        # Keep the gradient and triangle vector-based so their PDF edges align.
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.set_ticks([-50, -25, 0, 25, 50], labels=["-50", "-25", "0", "+25", "+50"])
        colorbar.ax.tick_params(labelsize=8, length=3, width=0.65)
        colorbar.set_label("Change in mean computation time (%)", fontsize=9, labelpad=5)
        colorbar.outline.set_visible(False)
        pdf_path = output_dir / (STEM + ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, metadata={
            "Title": "Effects of occupancy and robot radius on computation time",
            "Subject": "Annotated percentage changes in mean update times; stages 2-8; occupancy 40% to 60%; radius 0.3 to 0.45; colors capped at +/-50%; full values annotated.",
            "Creator": "experiments.plot_computation_time_parameter_effects",
        })
        return pdf_path
    finally:
        plt.close(fig)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_csv_tables(cells, components, output_dir):
    summary_path = output_dir / (STEM + "_summary.csv")
    component_path = output_dir / (STEM + "_components.csv")
    write_csv(summary_path, cells)
    write_csv(component_path, components)
    return summary_path, component_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-results-dir", type=Path, default=SINGLE_RESULTS)
    parser.add_argument("--percent-results-dir", type=Path, default=PERCENT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summaries, _ = build_summaries(args.single_results_dir.resolve(), args.percent_results_dir.resolve())
    cells, components = build_effects(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(plot_effects(cells, args.output_dir))
    for path in write_csv_tables(cells, components, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
