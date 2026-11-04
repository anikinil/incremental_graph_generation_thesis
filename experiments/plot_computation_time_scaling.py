"""Plot computation time against warehouse size at fixed occupancy and radius.

Run from the repository root:
    MPLCONFIGDIR=/tmp/thesis-matplotlib venv/bin/python -m experiments.plot_computation_time_scaling

Export one four-panel figure for each update regime. Within every panel,
occupancy and robot radius are fixed. Stages 2-8 are averaged within each
seed, followed by the mean and sample standard deviation across ten seeds.
"""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

from experiments.plot_computation_time_comparison import (
    PERCENT_RESULTS,
    SINGLE_RESULTS,
    configure_style,
)
from experiments.plot_stage_timings import (
    ALGORITHM_FAMILIES,
    ALGORITHMS,
    REPOSITORY_ROOT,
    lighter_color,
    load_timings,
)


OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/visualization/computation_time_scaling"
WAREHOUSES = ("10x20", "20x40", "30x60", "40x80", "50x100")
OCCUPANCIES = (40, 60)
RADII = (0.3, 0.45)
CONDITIONS = {
    (40, 0.3): "or0_4__rr0_3",
    (40, 0.45): "or0_4__rr0_45",
    (60, 0.3): "or0_6__rr0_3",
    (60, 0.45): "or0_6__rr0_45",
}
UPDATE_STAGES = tuple(range(2, 9))
TIME_LIMITS = (1e-4, 1e3)
FIGURE_NAMES = {
    "single_obstacle": "computation_time_scaling_single_obstacle",
    "five_percent": "computation_time_scaling_five_percent",
}


def summarize_source(source, regime):
    """Validate the complete matrix, including the known VGRAPH omission."""
    rows = load_timings(source)
    seeds = sorted({row["seed"] for row in rows})
    if len(seeds) != 10:
        raise ValueError(f"Expected ten seeds in {source}")

    expected = {
        (warehouse, condition, algorithm, seed, stage)
        for warehouse in WAREHOUSES
        for condition in CONDITIONS.values()
        for algorithm in ALGORITHMS
        if (warehouse, algorithm) != ("50x100", "dynamic_vgraph")
        for seed in seeds
        for stage in (1, *UPDATE_STAGES)
    }
    observed = {
        (row["warehouse"], row["condition"], row["algorithm"], row["seed"], row["stage"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            f"Incomplete timing matrix in {source}: "
            f"{len(expected - observed)} missing and {len(observed - expected)} unexpected samples"
        )

    by_seed = defaultdict(list)
    for row in rows:
        if row["stage"] in UPDATE_STAGES:
            key = (row["warehouse"], row["condition"], row["algorithm"], row["seed"])
            by_seed[key].append(row["seconds"])

    records = []
    for occupancy in OCCUPANCIES:
        for radius in RADII:
            condition = CONDITIONS[occupancy, radius]
            for warehouse in WAREHOUSES:
                for algorithm in ALGORITHMS:
                    if (warehouse, algorithm) == ("50x100", "dynamic_vgraph"):
                        continue
                    values = [mean(by_seed[warehouse, condition, algorithm, seed]) for seed in seeds]
                    average, deviation = mean(values), stdev(values)
                    records.append({
                        "figure": FIGURE_NAMES[regime] + ".pdf",
                        "update_regime": regime,
                        "source_csv": str(source.relative_to(REPOSITORY_ROOT))
                        if source.is_relative_to(REPOSITORY_ROOT) else str(source),
                        "warehouse": warehouse,
                        "warehouse_area": int(warehouse.split("x")[0]) * int(warehouse.split("x")[1]),
                        "condition": condition,
                        "occupancy_percent": occupancy,
                        "robot_radius": radius,
                        "algorithm": algorithm,
                        "seed_count": len(seeds),
                        "updates_per_seed": len(UPDATE_STAGES),
                        "timing_count": len(seeds) * len(UPDATE_STAGES),
                        "seconds_mean": average,
                        "seconds_std": deviation,
                        "lower_sd_bound_clipped": average - deviation < TIME_LIMITS[0],
                    })
    return records, seeds


def build_summaries(single_results, percent_results):
    sources = (
        ("single_obstacle", single_results / "timing_results.csv"),
        ("five_percent", percent_results / "timing_results.csv"),
    )
    summaries, seed_sets = {}, []
    for regime, source in sources:
        summaries[regime], seeds = summarize_source(source, regime)
        seed_sets.append(seeds)
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("The update regimes must use the same seed set")
    return summaries, sources


def plot_summary(summary, regime, output_dir):
    """Reuse the typography, colors, and static/dynamic styles of stage plots."""
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.91,
                        wspace=0.40, hspace=0.50)
    try:
        for column, occupancy in enumerate(OCCUPANCIES):
            bounds = axes[0, column].get_position()
            fig.text((bounds.x0 + bounds.x1) / 2, 0.965,
                     f"{occupancy}% obstacle occupancy",
                     ha="center", va="top", fontsize=10, fontweight="bold")
            for row, radius in enumerate(RADII):
                axis = axes[row, column]
                panel = {
                    (record["algorithm"], record["warehouse"]): record
                    for record in summary
                    if record["occupancy_percent"] == occupancy and record["robot_radius"] == radius
                }
                for static, dynamic, static_label, dynamic_label, color in ALGORITHM_FAMILIES:
                    for algorithm, label, is_static in (
                        (static, static_label, True),
                        (dynamic, dynamic_label, False),
                    ):
                        points = [(index, panel[algorithm, warehouse])
                                  for index, warehouse in enumerate(WAREHOUSES)
                                  if (algorithm, warehouse) in panel]
                        line_color = lighter_color(color) if is_static else color
                        axis.errorbar(
                            [index for index, _ in points],
                            [record["seconds_mean"] for _, record in points],
                            yerr=[record["seconds_std"] for _, record in points],
                            color=line_color, ecolor=line_color, label=label,
                            linestyle="--" if is_static else "-", linewidth=1.15,
                            marker="o", markersize=3.7 if is_static else 2.8,
                            markerfacecolor="white" if is_static else line_color,
                            markeredgewidth=0.8, capsize=1.7, capthick=0.65,
                            elinewidth=0.65, zorder=2 if is_static else 3,
                        )
                axis.set_yscale("log", nonpositive="clip")
                axis.set_ylim(*TIME_LIMITS)
                axis.set_xlim(-0.25, len(WAREHOUSES) - 0.75)
                axis.set_xticks(range(len(WAREHOUSES)), [
                    "$" + warehouse.replace("x", r"\times") + "$" for warehouse in WAREHOUSES
                ])
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
                axis.yaxis.set_major_formatter(LogFormatterMathtext())
                axis.yaxis.set_minor_locator(NullLocator())
                axis.tick_params(which="major", direction="out", length=3, width=0.65,
                                 labelbottom=True, labelleft=True)
                axis.tick_params(axis="x", labelsize=7.3)
                axis.grid(axis="y", color="0.88", linewidth=0.5)
                axis.set_axisbelow(True)
                axis.set_ylabel("Mean computation time (s)", labelpad=5)
                axis.set_xlabel("Warehouse size (units)", labelpad=5, fontsize=8.3)
        for row, radius in enumerate(RADII):
            bounds = axes[row, 0].get_position()
            fig.text(0.065, (bounds.y0 + bounds.y1) / 2,
                     f"Robot radius: {radius:g} units", rotation=90,
                     ha="center", va="center", fontsize=10, fontweight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.54, 0.085),
                   ncol=4, frameon=False, fontsize=7.7, handlelength=2.5,
                   columnspacing=1.15, labelspacing=0.9, handletextpad=0.5)
        stem = FIGURE_NAMES[regime]
        pdf_path = output_dir / (stem + ".pdf")
        regime_title = "single-obstacle changes" if regime == "single_obstacle" else "5% obstacle changes"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, metadata={
            "Title": f"Computation time by warehouse size: {regime_title}",
            "Subject": "Occupancy as columns; robot radius as rows; stages 2-8 averaged within seed; mean and sample SD across ten seeds; initialization excluded.",
            "Creator": "experiments.plot_computation_time_scaling",
        })
        return pdf_path
    finally:
        plt.close(fig)


def write_summary_csv(summaries, output_dir):
    records = [record for summary in summaries.values() for record in summary]
    summary_path = output_dir / "computation_time_scaling_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-results-dir", type=Path, default=SINGLE_RESULTS)
    parser.add_argument("--percent-results-dir", type=Path, default=PERCENT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summaries, _ = build_summaries(args.single_results_dir.resolve(), args.percent_results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for regime, summary in summaries.items():
        print(plot_summary(summary, regime, args.output_dir))
    print(write_summary_csv(summaries, args.output_dir))


if __name__ == "__main__":
    main()
