
from pathlib import Path
import sys

from experiments.evaluate_results import evaluate_results
from experiments.run_evaluation_matrix import run_evaluation_matrix


def main():
    config_paths = sys.argv[1:] or [
        "experiments/evaluation_matrix_one_obstacle_change.toml",
        "experiments/evaluation_matrix_5_percent_change.toml"
    ]

    for config_path in config_paths:
        results_dir = run_evaluation_matrix(Path(config_path))
        evaluate_results(results_dir)


if __name__ == "__main__":
    main()
