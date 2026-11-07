"""Plot static and incremental computation times across scenario stages.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/roadmap-matplotlib venv/bin/python -m experiments.plot_stage_timings

By default, use the latest evaluation matrix with a timing_results.csv and
write the 10x20 and 40x80 PDFs to artifacts/visualization. Each line shows the
arithmetic mean across seeds and occupancy/radius conditions at each stage.
Error bars show the mean plus or minus one sample standard deviation across
those same observations. Bounds outside the displayed y-axis range are clipped
and noted on the figure, including nonpositive lower bounds on the log axis.
Stage 1 is initial construction; stages 2-8 are rebuilds or incremental updates.
The y-axis is logarithmic by default because runtimes span several orders of
magnitude. All methods use filled circular markers; static methods use lighter,
dashed lines and incremental methods use solid lines of the same family colors.
The plotted values, sample standard deviations, and sample counts are also
exported as CSV.
"""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import LogLocator, NullLocator

from artifacts.presentation import (
    ALGORITHM_FAMILY_COLORS,
    LEGEND_FONT_SIZE,
    LEGEND_POSITION,
    LEGEND_RIGHT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_FAMILIES = (
    ("vgraph", "dynamic_vgraph", "VGRAPH (static)", "VGRAPH (dynamic)", ALGORITHM_FAMILY_COLORS["vgraph"]),
    ("prm", "drm", "PRM (static)", "DRM (dynamic)", ALGORITHM_FAMILY_COLORS["prm"]),
    ("gvd", "dynamic_gvd", "GVD (static)", "GVD (dynamic)", ALGORITHM_FAMILY_COLORS["gvd"]),
    (
        "kings_graph", "dynamic_kings_graph",
        "King's graph (static)", "King's graph (dynamic)", ALGORITHM_FAMILY_COLORS["kings_graph"],
    ),
)
ALGORITHMS = tuple(name for family in ALGORITHM_FAMILIES for name in family[:2])


def load_timings(path):
    """Read individual measurements, rejecting invalid or duplicate samples."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"seed", "warehouse", "condition", "stage", "algorithm", "seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing timing columns in {path}: {sorted(required)}")
        rows = list(reader)

    seen = set()
    for row in rows:
        key = tuple(row[field] for field in (
            "seed", "warehouse", "condition", "stage", "algorithm",
        ))
        if key in seen:
            raise ValueError(f"Duplicate timing measurement: {key}")
        seen.add(key)
        match = re.fullmatch(r"stage_(\d+)", row["stage"])
        if match is None:
            raise ValueError(f"Unexpected stage name: {row['stage']}")
        row["stage"] = int(match[1])
        row["seconds"] = float(row["seconds"])
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError(f"Expected a finite positive runtime: {key}")
    return rows


def summarize_warehouse(rows, warehouse, *, metrics=("seconds",)):
    """Require matching scenario samples for all eight lines before averaging."""
    selected = [
        row for row in rows
        if row["warehouse"] == warehouse and row["algorithm"] in ALGORITHMS
    ]
    if not selected:
        raise ValueError(f"No measurements for {warehouse}")
    stages = sorted({row["stage"] for row in selected})
    scenarios = {(row["seed"], row["condition"]) for row in selected}
    expected = {(seed, condition, stage) for seed, condition in scenarios for stage in stages}

    groups = defaultdict(list)
    for algorithm in ALGORITHMS:
        measurements = [row for row in selected if row["algorithm"] == algorithm]
        actual = {(row["seed"], row["condition"], row["stage"]) for row in measurements}
        if actual != expected:
            raise ValueError(
                f"Incomplete comparison for {warehouse}/{algorithm}: "
                f"missing {len(expected - actual)} scenario-stage measurements."
            )
        for row in measurements:
            groups[algorithm, row["stage"]].append(row)

    summary = []
    for algorithm in ALGORITHMS:
        for stage in stages:
            measurements = groups[algorithm, stage]
            record = {
                "warehouse": warehouse,
                "stage": stage,
                "algorithm": algorithm,
                "sample_count": len(measurements),
                "seed_count": len({seed for seed, _ in scenarios}),
                "condition_count": len({condition for _, condition in scenarios}),
            }
            for metric in metrics:
                values = [row[metric] for row in measurements]
                record[f"{metric}_mean"] = mean(values)
                record[f"{metric}_std"] = stdev(values) if len(values) > 1 else ""
            summary.append(record)
    return summary


def lighter_color(color, amount=0.45):
    """Mix an opaque family color with white to obtain the static tint."""
    return tuple(value + amount * (1 - value) for value in to_rgb(color))


def plot_warehouse(
    summary, output_dir, linear_y=False, *, value_field="seconds_mean",
    ylabel="Mean computation time (s)", filename_prefix="computation_time",
    y_limits=(0, None),
):
    """Render stage means with one sample SD using the shared figure style."""
    warehouse = summary[0]["warehouse"]
    stages = sorted({row["stage"] for row in summary})
    values = {(row["algorithm"], row["stage"]): row[value_field] for row in summary}
    std_field = value_field.removesuffix("_mean") + "_std"
    # One observation has no sample SD: keep its mean and omit its error bar.
    deviations = {
        (row["algorithm"], row["stage"]): (
            row[std_field] if row[std_field] != "" else math.nan
        )
        for row in summary
    }

    # Follow the existing 16x10 result figures and shared right-margin legend.
    fig, axis = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=0.11, right=LEGEND_RIGHT, bottom=0.18, top=0.96)
    try:
        for static, incremental, static_label, incremental_label, color in ALGORITHM_FAMILIES:
            for algorithm, label, is_static in (
                (static, static_label, True), (incremental, incremental_label, False),
            ):
                line_color = lighter_color(color) if is_static else to_rgb(color)
                axis.errorbar(
                    stages,
                    [values[algorithm, stage] for stage in stages],
                    yerr=[deviations[algorithm, stage] for stage in stages],
                    label=label,
                    color=line_color,
                    ecolor=line_color,
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.2,
                    linestyle="--" if is_static else "-",
                    marker="o",
                    markerfacecolor=line_color,
                    markersize=7,
                    markeredgewidth=1.6,
                    linewidth=2.4,
                )

        if not linear_y:
            # Preserve the full SD; nonpositive endpoints continue below the
            # plot without drawing a false lower cap at a positive value.
            axis.set_yscale("log", nonpositive="clip")
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=100))
            axis.yaxis.set_minor_locator(NullLocator())
        else:
            axis.set_ylim(*y_limits)
        axis.set_xlabel("Scenario stage (1 = initial construction)", labelpad=14)
        axis.set_ylabel(ylabel, labelpad=14)
        axis.set_xticks(stages)
        axis.set_xlim(min(stages) - 0.2, max(stages) + 0.2)
        axis.tick_params(axis="both", which="major", labelsize=LEGEND_FONT_SIZE)
        axis.set_axisbelow(True)
        axis.grid(axis="y", which="major", color="0.88", linewidth=0.8)

        note = "Error bars: mean ± 1 SD across seeds and conditions."
        lower_limit, upper_limit = axis.get_ylim()
        if any(
            values[key] - deviation < lower_limit
            or values[key] + deviation > upper_limit
            for key, deviation in deviations.items()
        ):
            note += "\nSD bounds outside the displayed y-axis range are clipped."
        fig.text(0.11, 0.02, note, fontsize=LEGEND_FONT_SIZE - 6, va="bottom")

        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center left", bbox_to_anchor=LEGEND_POSITION,
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=2.2,
            labelspacing=0.7,
        )
        scale = "linear" if linear_y else "logarithmic"
        output_path = output_dir / f"{filename_prefix}_by_stage_{warehouse}.pdf"
        fig.savefig(
            output_path, format="pdf", bbox_inches="tight",
            metadata={
                "Title": f"{ylabel} by scenario stage: {warehouse}",
                "Subject": (
                    "Arithmetic means across seeds and occupancy/radius conditions; "
                    "error bars show plus or minus one sample standard deviation; "
                    f"stage 1 is initialization; {scale} y-axis. {note}"
                ),
                "Creator": "experiments.plot_stage_timings",
            },
        )
        return output_path
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="Evaluation run containing timing_results.csv")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "artifacts/visualization",
    )
    parser.add_argument("--warehouses", nargs="+", default=["10x20", "40x80"])
    parser.add_argument("--linear-y", action="store_true", help="Use a linear time axis instead of logarithmic")
    args = parser.parse_args()

    if args.results_dir is None:
        candidates = sorted((REPOSITORY_ROOT / "artifacts/results/evaluation_matrix_5_percent_change").glob("*/timing_results.csv"))
        if not candidates:
            parser.error("No timing_results.csv found; provide --results-dir.")
        args.results_dir = candidates[-1].parent

    rows = load_timings(args.results_dir / "timing_results.csv")
    # Validate every requested comparison before creating output files.
    summaries = [summarize_warehouse(rows, warehouse) for warehouse in args.warehouses]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": LEGEND_FONT_SIZE,
        "axes.labelsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print(f"Source: {args.results_dir / 'timing_results.csv'}")
    for summary in summaries:
        output_path = plot_warehouse(summary, args.output_dir, args.linear_y)
        print(f"Saved {output_path}")

    summary_path = args.output_dir / "computation_time_by_stage_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0][0]))
        writer.writeheader()
        writer.writerows(row for summary in summaries for row in summary)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
