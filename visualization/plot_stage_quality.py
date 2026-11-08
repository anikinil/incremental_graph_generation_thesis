"""Plot task-based and global roadmap quality across scenario stages.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/roadmap-matplotlib venv/bin/python -m visualization.plot_stage_quality

Use the latest evaluation matrix with a quality_results.csv by default and
write eight PDFs to artifacts/visualization: four metrics for 10x20 and 40x80.
Use --metrics to plot a subset, for example::

    MPLCONFIGDIR=/tmp/roadmap-matplotlib venv/bin/python -m visualization.plot_stage_quality \\
        --metrics data_structure_density geometric_roadmap_density

Each point is the arithmetic mean of the per-roadmap values across seeds
and occupancy/radius conditions at that stage. Error bars show plus or minus
one sample standard deviation across those same per-roadmap values.
Path redundancy is the
non-bridge path percentage; the raw CSV stores a fraction, converted here to
percent. The axes are linear, with route length scaled to the observed range
and percentages shown from 0 to 100 with headroom for markers at 100 percent.
Both density metrics use linear axes starting at zero.
The colors, static tints, line styles, markers, and layout use the same renderer
as plot_stage_timings. Plotted values, sample standard deviations, and sample
counts are exported as CSV.
"""

import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.presentation import LEGEND_FONT_SIZE
from visualization.plot_stage_timings import (
    REPOSITORY_ROOT,
    plot_warehouse,
    summarize_warehouse,
)


METRIC_SETTINGS = (
    (
        "average_task_route_length", "Mean task route length (units)",
        "task_route_length", (None, None),
    ),
    (
        "average_non_bridge_path_percentage", "Mean non-bridge path percentage (%)",
        "path_redundancy", (0, 105),
    ),
    (
        "data_structure_density", r"Mean data structure density (1/units$^2$)",
        "data_structure_density", (0, None),
    ),
    (
        "geometric_roadmap_density", r"Mean geometric roadmap density (units/units$^2$)",
        "geometric_roadmap_density", (0, None),
    ),
)
METRICS = tuple(metric for metric, *_ in METRIC_SETTINGS)


def load_quality(path):
    """Load individual roadmap averages and convert redundancy to percent."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"seed", "warehouse", "condition", "stage", "algorithm", *METRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing quality columns in {path}: {sorted(missing)}")
        rows = list(reader)

    seen = set()
    for row in rows:
        key = tuple(row[field] for field in (
            "seed", "warehouse", "condition", "stage", "algorithm",
        ))
        if key in seen:
            raise ValueError(f"Duplicate quality measurement: {key}")
        seen.add(key)
        match = re.fullmatch(r"stage_(\d+)", row["stage"])
        if match is None:
            raise ValueError(f"Unexpected stage name: {row['stage']}")
        row["stage"] = int(match[1])
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError(f"Expected a finite nonnegative {metric}: {key}")
        redundancy = row["average_non_bridge_path_percentage"]
        if redundancy > 1:
            raise ValueError(f"Expected redundancy as a fraction between 0 and 1: {key}")
        row["average_non_bridge_path_percentage"] = 100 * redundancy
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="Evaluation run containing quality_results.csv")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "artifacts/visualization",
    )
    parser.add_argument("--warehouses", nargs="+", default=["10x20", "40x80"])
    parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=METRICS,
        help="Quality metrics to plot (default: all four)",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        candidates = sorted((REPOSITORY_ROOT / "artifacts/results/evaluation_matrix_5_percent_change").glob("*/quality_results.csv"))
        if not candidates:
            parser.error("No quality_results.csv found; provide --results-dir.")
        args.results_dir = candidates[-1].parent

    rows = load_quality(args.results_dir / "quality_results.csv")
    summaries = [
        summarize_warehouse(rows, warehouse, metrics=METRICS)
        for warehouse in args.warehouses
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": LEGEND_FONT_SIZE,
        "axes.labelsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print(f"Source: {args.results_dir / 'quality_results.csv'}")
    for summary in summaries:
        for metric, ylabel, filename_prefix, y_limits in METRIC_SETTINGS:
            if metric not in args.metrics:
                continue
            output_path = plot_warehouse(
                summary, args.output_dir, linear_y=True,
                value_field=f"{metric}_mean", ylabel=ylabel,
                filename_prefix=filename_prefix, y_limits=y_limits,
            )
            print(f"Saved {output_path}")

    summary_path = args.output_dir / "roadmap_quality_by_stage_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0][0]))
        writer.writeheader()
        writer.writerows(row for summary in summaries for row in summary)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
