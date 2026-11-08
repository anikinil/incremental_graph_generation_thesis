
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import sys


ALGORITHM_PAIRS = [
    ("King's graph", "kings_graph", "dynamic_kings_graph"),
    ("PRM / DRM", "prm", "drm"),
    ("VGRAPH", "vgraph", "dynamic_vgraph"),
    ("GVD", "gvd", "dynamic_gvd"),
]



def load_timings(path):
    with path.open(newline="") as file:
        results = list(csv.DictReader(file))

    for result in results:
        result["seconds"] = float(result["seconds"])

    return results


def _calculate_averages(results):
    grouped_results = defaultdict(list)
    fields = ["warehouse", "condition", "algorithm", "seed"]

    # average the update times within each seed excluding initialization
    for result in results:
        if result["stage"] != "stage_001":
            key = tuple(result[field] for field in fields)
            grouped_results[key].append(result["seconds"])

    grouped_averages = defaultdict(list)

    for (warehouse, condition, algorithm, seed), values in grouped_results.items():
        grouped_averages[warehouse, condition, algorithm].append(mean(values))

    # calculate the mean and sample standard deviation across the seeds
    averages = {}
    for key, values in grouped_averages.items():
        averages[key] = {"seconds": mean(values), "seconds_std": stdev(values)}

    return averages 


def _scenario_order(scenario):
    warehouse, condition = scenario
    height, width = warehouse.split("x")
    return int(height), int(width), condition


def build_summary(results):
    averages = _calculate_averages(results)
    scenarios = sorted({(warehouse, condition) for warehouse, condition, algorithm in averages}, key=_scenario_order)
    comparisons = []

    for warehouse, condition in scenarios:
        occupancy, radius = condition.split("__")
        occupancy = float(occupancy.removeprefix("or").replace("_", ".")) * 100
        radius = float(radius.removeprefix("rr").replace("_", "."))

        for label, static_name, incremental_name in ALGORITHM_PAIRS:
            static = averages[warehouse, condition, static_name]
            incremental = averages.get((warehouse, condition, incremental_name))
            row = {"warehouse": warehouse, "occupancy_percent": occupancy, "robot_radius": radius, "algorithm": label}
            row["static_seconds"] = static["seconds"]
            row["static_seconds_std"] = static["seconds_std"]
            row["incremental_seconds"] = None
            row["incremental_seconds_std"] = None
            row["speedup"] = None

            if incremental is not None:
                row["incremental_seconds"] = incremental["seconds"]
                row["incremental_seconds_std"] = incremental["seconds_std"]
                row["speedup"] = static["seconds"] / incremental["seconds"]

            comparisons.append(row)

    return comparisons


def _format_time(seconds, seconds_std):
    if seconds is None:
        return r"\multicolumn{2}{c}{\textemdash}"
    return f"{seconds:.4f} & {seconds_std:.3f}"


def _get_time_column(rows, field):
    times = [row[field] for row in rows if row[field] is not None]
    deviations = [row[field + "_std"] for row in rows if row[field] is not None]
    integer_digits = len(f"{max(times):.4f}".split(".")[0])
    deviation_digits = len(f"{max(deviations):.3f}".split(".")[0])
    return rf"S[table-format={integer_digits}.4,round-mode=none] @{{\,\(\pm\)\,}} S[table-format={deviation_digits}.3,round-mode=none]"


def render_latex(rows):
    header = r"{\shortstack{$\mathrm{W}\times\mathrm{H}$\\(units)}} & {\shortstack{Occup.\\rate}} & {\shortstack{Radius\\(units)}} & Algorithm & \multicolumn{2}{c}{\shortstack{Static\\gener. (\si{\second})}} & \multicolumn{2}{c}{\shortstack{Increm.\\updates (\si{\second})}} & {\shortstack{Speedup\\($\times$)}} \\"
    static_column = _get_time_column(rows, "static_seconds")
    incremental_column = _get_time_column(rows, "incremental_seconds")

    lines = [rf"""{{\small\begin{{longtable}}{{l S[table-format=2.0{{\%}},table-align-text-after=false] S l {static_column} {incremental_column} S}}
\toprule
{header}
\midrule
\endhead
\bottomrule
\endfoot"""]

    previous_scenario = None
    for row in rows:
        scenario = row["warehouse"], row["occupancy_percent"], row["robot_radius"]
        if previous_scenario is not None and scenario != previous_scenario:
            lines.append(r"\midrule")

        height, width = row["warehouse"].split("x")
        if scenario != previous_scenario:
            row_span = len(ALGORITHM_PAIRS)
            dimensions = rf"\multirow{{{row_span}}}{{*}}{{${height}\times{width}$}}"
            occupancy = rf"\multicolumn{{1}}{{c}}{{\multirow{{{row_span}}}{{*}}{{{row['occupancy_percent']}\%}}}}"
            radius = rf"\multicolumn{{1}}{{c}}{{\multirow{{{row_span}}}{{*}}{{{row['robot_radius']}}}}}"
        else:
            dimensions = " "
            occupancy = " "
            radius = " "

        cells = [dimensions, occupancy, radius, row["algorithm"]]
        cells.append(_format_time(row["static_seconds"], row["static_seconds_std"]))
        cells.append(_format_time(row["incremental_seconds"], row["incremental_seconds_std"]))
        cells.append(r"{\textemdash}" if row["speedup"] is None else f"{row['speedup']:.2f}")
        lines.append(" & ".join(cells) + r" \\")
        previous_scenario = scenario

    lines.append("\\end{longtable}}\n")
    return "\n".join(lines)


def export_summary(results_dir, output_path=None):
    results = load_timings(results_dir / "timing_results.csv")
    summary = build_summary(results)
    table_path = output_path if output_path is not None else results_dir / "latex_tables" / "timing_summary.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_latex(summary))
    print(f"{table_path}: {len(summary)} rows")
    return table_path


def main():
    results_dir = Path(sys.argv[1])
    export_summary(results_dir)


if __name__ == "__main__":
    main()
