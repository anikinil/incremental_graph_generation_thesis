"""Export separate task-quality and density comparisons in the timing style.

Run from the repository root:
    MPLCONFIGDIR=/tmp/thesis-matplotlib venv/bin/python -m experiments.plot_roadmap_quality_overview

Each PDF shows warehouse sizes as rows and single-obstacle/5% changes as
columns. Statistics refer to the existing evaluation samples and include
stage 1. PDFs and the summary CSV share one directory.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from experiments.plot_computation_time_comparison import (
    PERCENT_RESULTS,
    SINGLE_RESULTS,
    STAGES,
    WAREHOUSES,
    configure_style,
)
from experiments.plot_stage_quality import load_quality
from experiments.plot_stage_timings import (
    ALGORITHM_FAMILIES,
    REPOSITORY_ROOT,
    lighter_color,
    summarize_warehouse,
)


OUTPUT_DIR = REPOSITORY_ROOT / "artifacts/visualization/roadmap_quality_overview"
METRIC_SETTINGS = (
    ("average_task_route_length", "Task route length", "Mean task route length (units)", "units", "task_route_length_overview"),
    ("average_non_bridge_path_percentage", "Path redundancy", "Mean non-bridge proportion (%)", "%", "path_redundancy_overview"),
    ("geometric_roadmap_density", "Geometric roadmap density", "Geometric density (units$^{-1}$)", "units/units^2", "geometric_roadmap_density_overview"),
    ("data_structure_density", "Data structure density", "Data structure density (units$^{-2}$)", "1/units^2", "data_structure_density_overview"),
)
DENSITY_METRICS = {"geometric_roadmap_density", "data_structure_density"}
REGIMES = (
    ("single_obstacle", "Single-obstacle changes"),
    ("five_percent", "5% obstacle changes"),
)


def build_summaries(single_source, percent_source):
    """Validate 40 complete trajectories per warehouse and update regime."""
    sources = (("single_obstacle", single_source), ("five_percent", percent_source))
    summaries = {}
    scenario_sets = {}
    metrics = tuple(metric for metric, *_ in METRIC_SETTINGS)
    for regime, source in sources:
        rows = load_quality(source)
        for warehouse in WAREHOUSES:
            selected = [row for row in rows if row["warehouse"] == warehouse]
            scenarios = {(row["seed"], row["condition"]) for row in selected}
            if len(scenarios) != 40:
                raise ValueError(f"Expected 40 scenarios for {regime}/{warehouse}")
            if len({seed for seed, _ in scenarios}) != 10:
                raise ValueError(f"Expected ten seeds for {regime}/{warehouse}")
            if len({condition for _, condition in scenarios}) != 4:
                raise ValueError(f"Expected four conditions for {regime}/{warehouse}")
            if {row["stage"] for row in selected} != set(STAGES):
                raise ValueError(f"Expected stages 1-8 for {regime}/{warehouse}")
            scenario_sets[regime, warehouse] = scenarios
            summaries[regime, warehouse] = summarize_warehouse(rows, warehouse, metrics=metrics)
    for warehouse in WAREHOUSES:
        if scenario_sets["single_obstacle", warehouse] != scenario_sets["five_percent", warehouse]:
            raise ValueError(f"The two runs use different scenario cases for {warehouse}")
    return summaries, sources


def plot_overview(summaries, output_dir, metric_settings):
    """Match linear scales across regimes; use one common scale for densities."""
    metric, metric_title, ylabel, _, filename = metric_settings
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.91,
                        wspace=0.40, hspace=0.50)
    try:
        row_limits = {}
        if metric in DENSITY_METRICS:
            all_records = [record for summary in summaries.values() for record in summary]
            density_upper = max(record[f"{metric}_mean"] + record[f"{metric}_std"]
                                for record in all_records)
        for warehouse in WAREHOUSES:
            if metric == "average_non_bridge_path_percentage":
                row_limits[warehouse] = (0, 105)
            elif metric in DENSITY_METRICS:
                row_limits[warehouse] = (0, 1.10 * density_upper)
            else:
                records = [record for regime, _ in REGIMES for record in summaries[regime, warehouse]]
                lower = min(record[f"{metric}_mean"] - record[f"{metric}_std"] for record in records)
                upper = max(record[f"{metric}_mean"] + record[f"{metric}_std"] for record in records)
                padding = max(0.10 * (upper - lower), 0.5)
                row_limits[warehouse] = (max(0, lower - padding), upper + padding)
        for column, (regime, title) in enumerate(REGIMES):
            bounds = axes[0, column].get_position()
            fig.text((bounds.x0 + bounds.x1) / 2, 0.965, title,
                     ha="center", va="top", fontsize=10, fontweight="bold")
            for row, warehouse in enumerate(WAREHOUSES):
                axis = axes[row, column]
                summary = summaries[regime, warehouse]
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
                            [record[f"{metric}_mean"] for record in records],
                            yerr=[record[f"{metric}_std"] for record in records],
                            color=line_color, ecolor=line_color, label=label,
                            linestyle="--" if is_static else "-", linewidth=1.15,
                            marker="o", markersize=3.7 if is_static else 2.8,
                            markerfacecolor="white" if is_static else line_color,
                            markeredgewidth=0.8, capsize=1.7, capthick=0.65,
                            elinewidth=0.65, zorder=2 if is_static else 3,
                        )
                axis.set_xlim(0.72, 8.28)
                axis.set_xticks(STAGES)
                axis.set_ylim(*row_limits[warehouse])
                if metric == "average_non_bridge_path_percentage":
                    # Headroom above 100 preserves full symmetric sample SD
                    # bars and prevents clipping of means equal to 100%.
                    axis.yaxis.set_major_locator(MultipleLocator(20))
                else:
                    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
                axis.tick_params(which="major", direction="out", length=3, width=0.65,
                                 labelbottom=True, labelleft=True)
                axis.grid(axis="y", color="0.88", linewidth=0.5)
                axis.set_axisbelow(True)
                axis.set_ylabel(ylabel, labelpad=5)
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
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, metadata={
            "Title": f"{metric_title} by stage: single-obstacle and 5% changes",
            "Subject": f"Four panels; 10x20 and 40x80 warehouses; {metric_title}; 40 scenarios per point; mean and sample SD; "
                       + ("one common linear scale starting at zero." if metric in DENSITY_METRICS
                          else "matched linear scales within rows."),
            "Creator": "experiments.plot_roadmap_quality_overview",
        })
        return pdf_path
    finally:
        plt.close(fig)


def write_summary_csv(summaries, sources, output_dir):
    records = []
    for regime, source in sources:
        source_label = str(source.relative_to(REPOSITORY_ROOT)) if source.is_relative_to(REPOSITORY_ROOT) else str(source)
        for warehouse in WAREHOUSES:
            for metric, _, _, unit, filename in METRIC_SETTINGS:
                for record in summaries[regime, warehouse]:
                    records.append({
                        "figure": f"{filename}.pdf",
                        "update_regime": regime,
                        "source_csv": source_label,
                        "warehouse": warehouse,
                        "metric": metric,
                        "unit": unit,
                        "algorithm": record["algorithm"],
                        "stage": record["stage"],
                        "sample_count": record["sample_count"],
                        "seed_count": record["seed_count"],
                        "condition_count": record["condition_count"],
                        "mean": record[f"{metric}_mean"],
                        "sample_std": record[f"{metric}_std"],
                    })
    summary_path = output_dir / "roadmap_quality_overview_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-results-dir", type=Path, default=SINGLE_RESULTS)
    parser.add_argument("--percent-results-dir", "--results-dir", dest="percent_results_dir", type=Path, default=PERCENT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--metrics", nargs="+", choices=[settings[0] for settings in METRIC_SETTINGS],
                        default=[settings[0] for settings in METRIC_SETTINGS],
                        help="Metrics to render; the summary CSV retains all four metrics")
    args = parser.parse_args()
    summaries, sources = build_summaries(
        args.single_results_dir.resolve() / "quality_results.csv",
        args.percent_results_dir.resolve() / "quality_results.csv",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric_settings in METRIC_SETTINGS:
        if metric_settings[0] not in args.metrics:
            continue
        print(plot_overview(summaries, args.output_dir, metric_settings))
    print(write_summary_csv(summaries, sources, args.output_dir))


if __name__ == "__main__":
    main()
