"""Export the four-panel timing figure used in the thesis results section.

Run from the repository root:
    MPLCONFIGDIR=/tmp/thesis-matplotlib venv/bin/python -m experiments.plot_computation_time_comparison

The output directory contains a vector PDF and a CSV with the plotted
means and sample standard deviations.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

from experiments.plot_stage_timings import (
    ALGORITHM_FAMILIES,
    REPOSITORY_ROOT,
    lighter_color,
    load_timings,
    summarize_warehouse,
)


SINGLE_RESULTS = REPOSITORY_ROOT / "artifacts/results/evaluation_matrix_one_obstacle_change/2026-10-26_09-37-47"
PERCENT_RESULTS = REPOSITORY_ROOT / "artifacts/results/evaluation_matrix_5_percent_change/2026-10-21_00-59-27"
OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/visualization/computation_time_comparison"
WAREHOUSES = ("10x20", "40x80")
STAGES = tuple(range(1, 9))


def build_panels(single_results, percent_results):
    """Require all 40 matched scenarios and all eight stages in each panel."""
    sources = (
        ("single_obstacle", single_results / "timing_results.csv"),
        ("five_percent", percent_results / "timing_results.csv"),
    )
    panels = {}
    scenario_sets = {}
    for regime, source in sources:
        rows = load_timings(source)
        for warehouse in WAREHOUSES:
            selected = [row for row in rows if row["warehouse"] == warehouse]
            scenarios = {(row["seed"], row["condition"]) for row in selected}
            if len(scenarios) != 40:
                raise ValueError(f"Expected 40 scenarios for {regime}/{warehouse}")
            if {row["stage"] for row in selected} != set(STAGES):
                raise ValueError(f"Expected stages 1-8 for {regime}/{warehouse}")
            if len({seed for seed, _ in scenarios}) != 10:
                raise ValueError(f"Expected ten seeds for {regime}/{warehouse}")
            if len({condition for _, condition in scenarios}) != 4:
                raise ValueError(f"Expected four conditions for {regime}/{warehouse}")
            scenario_sets[regime, warehouse] = scenarios
            panels[regime, warehouse] = [
                {"update_regime": regime, "source_csv": str(source.relative_to(REPOSITORY_ROOT))
                 if source.is_relative_to(REPOSITORY_ROOT) else str(source), **row}
                for row in summarize_warehouse(rows, warehouse)
            ]
    for warehouse in WAREHOUSES:
        if scenario_sets["single_obstacle", warehouse] != scenario_sets["five_percent", warehouse]:
            raise ValueError(f"The two runs use different scenario cases for {warehouse}")
    return panels, sources


def configure_style():
    """Apply the typography shared by the thesis comparison figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_panels(panels, output_dir):
    """Use common log limits and one legend to support direct comparisons."""
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.91,
                        wspace=0.40, hspace=0.50)
    try:
        for column, (regime, title) in enumerate((
            ("single_obstacle", "Single-obstacle changes"),
            ("five_percent", "5% obstacle changes"),
        )):
            bounds = axes[0, column].get_position()
            fig.text((bounds.x0 + bounds.x1) / 2, 0.965, title,
                     ha="center", va="top", fontsize=10, fontweight="bold")
            for row, warehouse in enumerate(WAREHOUSES):
                axis = axes[row, column]
                summary = panels[regime, warehouse]
                by_algorithm = {
                    algorithm: sorted((record for record in summary if record["algorithm"] == algorithm),
                                      key=lambda record: record["stage"])
                    for family in ALGORITHM_FAMILIES for algorithm in family[:2]
                }
                axis.axvspan(0.72, 1.28, color="0.94", zorder=0)
                for static, incremental, static_label, incremental_label, color in ALGORITHM_FAMILIES:
                    for algorithm, label, is_static in (
                        (static, static_label, True),
                        (incremental, incremental_label, False),
                    ):
                        records = by_algorithm[algorithm]
                        line_color = lighter_color(color) if is_static else color
                        axis.errorbar(
                            STAGES,
                            [record["seconds_mean"] for record in records],
                            yerr=[record["seconds_std"] for record in records],
                            color=line_color, ecolor=line_color, label=label,
                            linestyle="--" if is_static else "-", linewidth=1.15,
                            marker="o", markersize=3.7 if is_static else 2.8,
                            markerfacecolor="white" if is_static else line_color,
                            markeredgewidth=0.8, capsize=1.7, capthick=0.65,
                            elinewidth=0.65, zorder=2 if is_static else 3,
                        )
                axis.set_yscale("log", nonpositive="clip")
                axis.set_ylim(1e-4, 1e3)
                axis.set_xlim(0.72, 8.28)
                axis.set_xticks(STAGES)
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
                axis.yaxis.set_major_formatter(LogFormatterMathtext())
                axis.yaxis.set_minor_locator(NullLocator())
                axis.tick_params(which="major", direction="out", length=3, width=0.65,
                                 labelbottom=True, labelleft=True)
                axis.grid(axis="y", color="0.88", linewidth=0.5)
                axis.set_axisbelow(True)
                axis.set_ylabel("Mean computation time (s)", labelpad=5)
                axis.set_xlabel("Scenario stage (1 = initialization)", labelpad=5, fontsize=8.3)
        for row, warehouse in enumerate(WAREHOUSES):
            bounds = axes[row, 0].get_position()
            height, width = warehouse.split("x")
            fig.text(0.065, (bounds.y0 + bounds.y1) / 2,
                     f"{height} $\\times$ {width} warehouse", rotation=90,
                     ha="center", va="center", fontsize=10, fontweight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.54, 0.085),
                   ncol=4, frameon=False, fontsize=7.7, handlelength=2.5,
                   columnspacing=1.15, labelspacing=0.9, handletextpad=0.5)
        pdf_path = output_dir / "computation_time_comparison.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, metadata={
            "Title": "Computation time by stage: single-obstacle and 5% changes",
            "Subject": "Four panels; 10x20 and 40x80 warehouses; 40 scenarios per point; mean and sample SD; stage 1 includes initialization.",
            "Creator": "experiments.plot_computation_time_comparison",
        })
        return pdf_path
    finally:
        plt.close(fig)


def write_summary_csv(panels, output_dir):
    summary_path = output_dir / "computation_time_comparison_summary.csv"
    records = [record for panel in panels.values() for record in panel]
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
    panels, _ = build_panels(args.single_results_dir.resolve(), args.percent_results_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(plot_panels(panels, args.output_dir))
    print(write_summary_csv(panels, args.output_dir))


if __name__ == "__main__":
    main()
