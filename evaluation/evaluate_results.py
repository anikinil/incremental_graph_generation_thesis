import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

from evaluation.roadmap_quality_evaluation import average_task_metrics, evaluate_roadmap, evaluate_tasks, _load_roadmap_data


METRICS = ["average_task_route_length", "average_non_bridge_path_percentage", "geometric_roadmap_density", "data_structure_density"]


def _get_roadmap_runs(results_dir):
    runs = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        for warehouse_dir in sorted(seed_dir.iterdir()):
            for condition_dir in sorted(warehouse_dir.iterdir()):
                roadmap_run = max((condition_dir / "roadmaps").iterdir())
                runs.append({
                    "seed": seed_dir.name.removeprefix("seed_"),
                    "warehouse": warehouse_dir.name,
                    "condition": condition_dir.name,
                    "roadmap_run": roadmap_run,
                })
    return runs


def _evaluate_roadmap_run(run):
    results = []

    for roadmap in _load_roadmap_data(run["roadmap_run"]):
        result = {
            "seed": run["seed"],
            "warehouse": run["warehouse"],
            "condition": run["condition"],
            "stage": roadmap["stage"],
            "algorithm": roadmap["algorithm"],
        }

        # evaluate tasks, compute metric averages, and add to result
        average_task_metrics_result = average_task_metrics(evaluate_tasks(roadmap))
        result.update(average_task_metrics_result)

        # evaluate the roadmap itself and compute metric averages, and add to result
        roadmap_result = evaluate_roadmap(roadmap)
        result.update(roadmap_result)

        results.append(result)

    return results


def _load_timing_results(run):
    results = []

    with (run["roadmap_run"] / "timings.csv").open() as file:
        reader = csv.DictReader(file)
        for timing in reader:
            results.append({
                "seed": run["seed"],
                "warehouse": run["warehouse"],
                "condition": run["condition"],
                "stage": timing["scenario"],
                "algorithm": timing["algorithm"],
                "seconds": float(timing["seconds"]),
            })

    return results



def _calculate_averages(results, metrics, group_fields):
    grouped_results = defaultdict(list)

    # group results by the specified fields
    for result in results:
        key = tuple(result[field] for field in group_fields)
        grouped_results[key].append(result)

    # calculate averages for each group
    averages = []
    for key, group in grouped_results.items():
        average = dict(zip(group_fields, key))
        average["seed_count"] = len(group)
        for metric in metrics:
            values = [result[metric] for result in group]
            average[metric] = float(np.mean(values))
            average[f"{metric}_std"] = float(np.std(values, ddof=1))
        averages.append(average)

    return averages


def _save_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_results(results_dir):

    results_dir = Path(results_dir)
    quality_results = []
    timing_results = []
    runs = _get_roadmap_runs(results_dir)

    print("\n------------------------------------------------")
    print("\nEvaluating roadmap quality and collecting timing measurements...\n")
    for run in runs:
        quality_results.extend(_evaluate_roadmap_run(run))
        timing_results.extend(_load_timing_results(run))
        print(run["roadmap_run"])

    # save roadmap quality metrics for every seed, condition, stage and algorithm
    result_fields = ["seed", "warehouse", "condition", "stage", "algorithm", *METRICS]
    _save_csv(results_dir / "quality_results.csv", quality_results, result_fields)

    # average metrics over the seeds for each scenario, condition, stage and algorithm
    group_fields = ["warehouse", "condition", "stage", "algorithm"]
    quality_averages = _calculate_averages(quality_results, METRICS, group_fields)

    # save the quality averages together with their standard deviations
    averages_fields = ["warehouse", "condition", "stage", "algorithm", "seed_count", *METRICS, *[f"{metric}_std" for metric in METRICS]]
    _save_csv(results_dir / "quality_averages.csv", quality_averages, averages_fields)

    # save the timing measurements for every seed, condition, stage and algorithm
    timing_result_fields = ["seed", "warehouse", "condition", "stage", "algorithm", "seconds"]
    _save_csv(results_dir / "timing_results.csv", timing_results, timing_result_fields)

    # average timing measurements over the seeds (the stage )
    group_fields = ["warehouse", "condition", "stage", "algorithm"]
    timing_averages = _calculate_averages(timing_results, ["seconds"], group_fields)

    # save the timing averages together with their standard deviations
    timing_average_fields = ["warehouse", "condition", "stage", "algorithm", "seed_count", "seconds", "seconds_std"]
    _save_csv(results_dir / "timing_averages.csv", timing_averages, timing_average_fields)
    print(f"\nSaved quality and timing CSVs to {results_dir}")


def main():
    evaluate_results(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
