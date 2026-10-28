import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import sys


METRICS = ["average_task_route_length", "average_non_bridge_path_percentage", "geometric_roadmap_density", "data_structure_density"]
ALGORITHMS = [("VG", "vgraph"), ("DVG", "dynamic_vgraph"), ("PRM", "prm"), ("DRM", "drm"), ("GVD", "gvd"), ("Incr. GVD", "dynamic_gvd"), ("King's graph", "kings_graph"), ("Incr. King's graph", "dynamic_kings_graph")]


def load_quality(path):
    with path.open(newline="") as file:
        results = list(csv.DictReader(file))

    for result in results:
        for metric in METRICS:
            result[metric] = float(result[metric])
        result["average_non_bridge_path_percentage"] *= 100

    return results


def _calculate_averages(results):
    grouped_results = defaultdict(list)
    fields = ["warehouse", "condition", "algorithm", "seed"]

    # average the quality metrics within each seed, excluding initialization
    for result in results:
        if result["stage"] != "stage_001":
            key = tuple(result[field] for field in fields)
            grouped_results[key].append(result)

    grouped_averages = defaultdict(list)

    for (warehouse, condition, algorithm, seed), group in grouped_results.items():
        seed_average = {}
        for metric in METRICS:
            seed_average[metric] = mean(result[metric] for result in group)
        grouped_averages[warehouse, condition, algorithm].append(seed_average)

    # calculate the mean and sample standard deviation across the seeds
    averages = {}
    for key, group in grouped_averages.items():
        average = {}
        for metric in METRICS:
            values = [result[metric] for result in group]
            average[metric] = mean(values)
            average[metric + "_std"] = stdev(values)
        averages[key] = average

    return averages


def _scenario_order(scenario):
    warehouse, condition = scenario
    height, width = warehouse.split("x")
    return int(height), int(width), condition


def build_summary(results):
    averages = _calculate_averages(results)
    scenarios = sorted({(warehouse, condition) for warehouse, condition, algorithm in averages}, key=_scenario_order)
    summary = []

    for warehouse, condition in scenarios:
        occupancy, radius = condition.split("__")
        occupancy = float(occupancy.removeprefix("or").replace("_", ".")) * 100
        radius = float(radius.removeprefix("rr").replace("_", "."))

        for label, algorithm in ALGORITHMS:
            average = averages.get((warehouse, condition, algorithm), {})
            row = {"warehouse": warehouse, "occupancy_percent": occupancy, "robot_radius": radius, "algorithm": label}
            for metric in METRICS:
                row[metric] = average.get(metric)
                row[metric + "_std"] = average.get(metric + "_std")
            summary.append(row)

    return summary


def _format_metric(value, value_std):
    if value is None:
        return r"\multicolumn{2}{c}{\textemdash}"
    return f"{value:.4f} & {value_std:.3f}"


def _get_metric_column(rows, metric):
    values = [row[metric] for row in rows if row[metric] is not None]
    deviations = [row[metric + "_std"] for row in rows if row[metric] is not None]
    integer_digits = len(f"{max(values):.4f}".split(".")[0])
    deviation_digits = len(f"{max(deviations):.3f}".split(".")[0])
    return rf"S[table-format={integer_digits}.4,round-mode=none] @{{\,\(\pm\)\,}} S[table-format={deviation_digits}.3,round-mode=none]"


def render_latex(rows):
    metric_headers = [r"\multicolumn{2}{c}{\shortstack{Task route\\length (units)}}", r"\multicolumn{2}{c}{\shortstack{Non-bridge\\path (\%)}}", r"\multicolumn{2}{c}{\shortstack{Geom. roadmap\\density (units/units$^2$)}}", r"\multicolumn{2}{c}{\shortstack{Data structure\\density (1/units$^2$)}}"]
    header = r"{\shortstack{$\mathrm{W}\times\mathrm{H}$\\(units)}} & {\shortstack{Occup.\\rate}} & {\shortstack{Radius\\(units)}} & Algorithm & " + " & ".join(metric_headers) + r" \\"
    metric_columns = " ".join(_get_metric_column(rows, metric) for metric in METRICS)

    lines = [rf"""\begin{{landscape}}
\small
\begin{{longtable}}{{l S[table-format=2.0{{\%}},table-align-text-after=false] S l {metric_columns}}}
\toprule
{header}
\midrule
\endhead
\bottomrule
\endfoot"""]

    previous_scenario = None
    previous_height, previous_width = None, None
    scenario_count = 0
    for r_idx, row in enumerate(rows):
        scenario = row["warehouse"], row["occupancy_percent"], row["robot_radius"]
        if previous_scenario is not None and scenario != previous_scenario:
            scenario_count += 1
            lines.append(r"\midrule")
            if scenario_count % 3 == 0:
                lines.append(r"\newpage")

        height, width = row["warehouse"].split("x")
        
        if previous_height is not None and (previous_height != height or previous_width != width):
            print(f"Test: {height}x{width} vs {previous_height}x{previous_width}, row index: {r_idx}")
            lines.append(r"\midrule")
        
        if r_idx % 8 == 0:
            dimensions = rf"\multirow{{8}}{{*}}{{${height}\times{width}$}}"
            occupancy = rf"\multicolumn{{1}}{{c}}{{\multirow{{8}}{{*}}{{{row['occupancy_percent']}\%}}}}"
            radius = rf"\multicolumn{{1}}{{c}}{{\multirow{{8}}{{*}}{{{row['robot_radius']}}}}}"
        else:
            dimensions = " "
            occupancy = " "
            radius = " "
        
        cells = [dimensions, occupancy, radius, row["algorithm"]]
        for metric in METRICS:
            cells.append(_format_metric(row[metric], row[metric + "_std"]))
        lines.append(" & ".join(cells) + r" \\")
        previous_scenario = scenario
        previous_height, previous_width = height, width

    lines.append("\\end{longtable}\n\\end{landscape}\n")
    return "\n".join(lines)


def export_summary(results_dir):
    results = load_quality(results_dir / "quality_results.csv")
    summary = build_summary(results)
    table_path = results_dir / "latex_tables" / "quality_summary.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_latex(summary))
    print(f"{table_path}: {len(summary)} rows")


def main():
    results_dir = Path(sys.argv[1])
    export_summary(results_dir)


if __name__ == "__main__":
    main()
